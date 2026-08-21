"""Write the ARTIS output files: adata.txt, transitiondata.txt, phixsdata_v2.txt, compositiondata.txt."""

import argparse
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

from artisatomic.base import atomic_weights
from artisatomic.base import elsymbols
from artisatomic.base import hc_in_ev_cm
from artisatomic.base import ion_log_path
from artisatomic.base import log_and_print
from artisatomic.base import roman_numerals
from artisatomic.iondata import IonData
from artisatomic.ionhandlers import drop_handlers


def clear_files(args: argparse.Namespace) -> None:
    """Truncate the output files and write the phixs header, which the ions are appended to."""
    outdir = Path(args.output_folder)
    with (
        (outdir / "adata.txt").open("w", encoding="utf-8"),
        (outdir / "transitiondata.txt").open("w", encoding="utf-8"),
        (outdir / "phixsdata_v2.txt").open("w", encoding="utf-8") as fphixs,
    ):
        fphixs.write(f"{args.nphixspoints:d}\n")
        fphixs.write(f"{args.phixsnuincrement:14.7e}\n")


# A transition this strong is an electric dipole line, whatever the level names say. Below these
# values, one that breaks the delta J rule is taken to be a forbidden line that the source listed
# with its own small strength, which agrees with the label instead of contradicting it.
#
# The two need their own values because they are not the same quantity. f has no units; A is a
# rate in s-1 that spans many decades. Neither cut is a law of physics: a weak E1 line can sit
# well below either (an intercombination line, or one between high levels), and a strong M1 or E2
# line in a highly charged ion can sit above the A cut. They are where the two populations part in
# this corpus -- forbidden lines end near f ~ 1e-6 and A ~ 1e2 (QUB's Co III peaks at 14 s-1),
# and the lines that contradict their own J labels start at f = 7.8e-4 (F III's smallest), only
# eight times above the cut. A mislabelled line weaker than that falls on the wrong side, and
# that is the accepted cost.
min_f_asserts_e1 = 1e-4
min_a_asserts_e1 = 1e5


def strength_asserts_e1(dftransitions_ion: pl.DataFrame) -> pl.Expr:
    """Whether the source's own numbers claim the transition is an electric dipole line.

    f where the reader gives one, else the Einstein A. Both need a size, not merely a value: a
    forbidden line has an A, and a reader that tabulates f gives one to its forbidden lines too,
    so "greater than zero" says nothing about which kind of transition this is.
    """
    if "f" in dftransitions_ion.columns:
        return pl.col("f").fill_null(0.0).abs() > min_f_asserts_e1

    return pl.col("A").fill_null(0.0).abs() > min_a_asserts_e1


def add_level_ids_forbidden(dfenergylevels_ion: pl.DataFrame, dftransitions_ion: pl.DataFrame) -> pl.DataFrame:
    """Fill in whichever of lowerlevel, upperlevel and forbidden a reader did not supply.

    Readers that key their transitions by level name (namefrom/nameto) get the level ids joined
    on here; readers that already supply ids keep them.

    Forbidden here means "not an electric dipole transition", and two of the E1 selection rules
    are checked, each on whichever levels carry what it needs:

    - Laporte: E1 changes the parity, so two levels of the same parity cannot be E1.
    - Delta J: E1 has |J_upper - J_lower| <= 1, and J = 0 -> J = 0 is forbidden outright.

    Either rule is sufficient on its own, so the two are OR-ed. Neither rule can fire on a level
    that does not carry the quantum number it needs. A data set that gives no J thus keeps the
    behaviour it had before this code read J at all.

    The delta J rule yields to a transition the source made strong enough to be E1, as
    strength_asserts_e1() judges it. Some files disagree with their own J labels: CMFGEN's
    provisional F III set splits a term by a nominal 0.8 cm-1 and then shares the term's f over
    all the J pairs, so it lists delta J = 2 lines with f as large as 0.116, and O II reaches
    0.695. To call such a line forbidden would give a strong line the forbidden collision
    approximation, which is worse than the label it corrects. write_output_files() logs those
    instead. A weak line is not evidence of anything, so the J labels decide it.

    A null parity means the level has no definite parity, either because it merges sub-levels of
    both parities (CMFGEN's '1___' and '2s2_13w_2W') or because the reader could not read one
    from the level name. Null is the whole of that convention: polars resolves null == anything
    to null rather than to true, so an absent parity cannot match another absent one, and no
    sentinel number is needed for readers to spell it. A null J means the same, and both are
    resolved the same way: anything a reader could not give a number for -- a NaN, a string
    pandas inferred from a blank column -- casts to null and disables only its own rule.
    """
    if dftransitions_ion.is_empty():
        return dftransitions_ion

    if "upperlevel" not in dftransitions_ion.columns:
        dftransitions_ion = dftransitions_ion.join(
            dfenergylevels_ion.select(pl.col("levelid").alias("upperlevel"), pl.col("levelname").alias("nameto")),
            on="nameto",
        )

    if "lowerlevel" not in dftransitions_ion.columns:
        dftransitions_ion = dftransitions_ion.join(
            dfenergylevels_ion.select(pl.col("levelid").alias("lowerlevel"), pl.col("levelname").alias("namefrom")),
            on="namefrom",
        )

    if "forbidden" not in dftransitions_ion.columns:
        # null for anything that is not a number, so a NaN cannot compare equal to itself and a
        # string column cannot raise against the numbers it is compared with
        knownparity = pl.col("parity").cast(pl.Int64, strict=False)
        hasj = "j" in dfenergylevels_ion.columns
        knownj = pl.col("j").cast(pl.Float64, strict=False) if hasj else pl.lit(None, dtype=pl.Float64)

        assertse1 = strength_asserts_e1(dftransitions_ion)

        dftransitions_ion = (
            dftransitions_ion.join(
                dfenergylevels_ion.select(
                    pl.col("levelid").alias("lowerlevel"),
                    knownparity.alias("lower_parity"),
                    knownj.alias("lower_j"),
                ),
                on="lowerlevel",
            )
            .join(
                dfenergylevels_ion.select(
                    pl.col("levelid").alias("upperlevel"),
                    knownparity.alias("upper_parity"),
                    knownj.alias("upper_j"),
                ),
                on="upperlevel",
            )
            # The delta J rule is kept as its own column, because write_output_files() reports
            # the transitions that break it while the source still gives them an f.
            .with_columns(
                breaksdeltaj=(
                    ((pl.col("lower_j") - pl.col("upper_j")).abs() > 1)
                    | ((pl.col("lower_j") == 0) & (pl.col("upper_j") == 0))
                ).fill_null(False)
            )
            # Each rule gets fill_null(False) on its own, before the or. A null would otherwise
            # spread and make forbidden itself null, which "{forbidden:d}" cannot format, and a
            # level with no J would undo what the two parities had already settled.
            .with_columns(
                forbidden=(pl.col("lower_parity") == pl.col("upper_parity")).fill_null(False)
                | (pl.col("breaksdeltaj") & ~assertse1)
            )
            .drop("lower_j", "upper_j")
        )
    return dftransitions_ion


def log_deltaj_contradictions(flog, dftransitions_ion: pl.DataFrame, ionstr: str) -> None:
    """Report the transitions whose J labels and oscillator strength contradict each other.

    A transition with |delta J| > 1, or with J = 0 at both ends, is not an electric dipole
    transition. A large oscillator strength says that it is. add_level_ids_forbidden() lets the
    oscillator strength win, so the transition stays permitted, and this reports how often a data
    set needed that.

    CMFGEN's provisional F III set is the known example: it splits a term by a nominal 0.8 cm-1
    and shares the term's f over all the J pairs, so a delta J = 2 line can carry f = 0.116.
    """
    if "breaksdeltaj" not in dftransitions_ion.columns:
        return

    hasf = "f" in dftransitions_ion.columns
    strengthcol = "f" if hasf else "A"
    minstrength = min_f_asserts_e1 if hasf else min_a_asserts_e1
    # the same test the rule used, so this reports exactly the transitions it let through
    contradictions = dftransitions_ion.filter(pl.col("breaksdeltaj") & strength_asserts_e1(dftransitions_ion))
    if contradictions.is_empty():
        return

    largest = contradictions[strengthcol].abs().max()
    log_and_print(
        flog,
        f"WARNING: {contradictions.height:d} transitions of {ionstr} break the delta J rule but"
        f" carry {strengthcol} > {minstrength:g} (largest {largest:.3g}). The level names and the"
        f" {strengthcol} values of this data set disagree. The {strengthcol} values are used, so"
        f" these stay permitted.",
    )


def resolve_coll_str(dftransitions_ion: pl.DataFrame) -> pl.DataFrame:
    """Turn the joined upsilon column into coll_str, and let a negative one set the forbidden flag.

    A negative upsilon is not a collision strength. It is the reader's mark for "this pair is
    forbidden and I have no value", and readhillierdata writes -2 for the J pairs within a term.
    So it sets the flag rather than the parities alone. A merged term has no parity, and the pair
    would then come out permitted. Those pairs carry no A, so van Regemorter would get an
    oscillator strength of zero, which is no collisional coupling at all. The -2 asks instead for
    Axelrod's approximation.

    coll_str then repeats what the flag says: -2 forbidden, -1 unknown. Only a missing upsilon
    reaches the -1, because a negative one has already made the flag true.
    """
    return (
        dftransitions_ion.with_columns(forbidden=pl.col("forbidden") | (pl.col("upsilon") < 0.0).fill_null(False))
        .with_columns(
            coll_str=pl.when(pl.col("upsilon") >= 0.0)
            .then(pl.col("upsilon"))
            .otherwise(pl.when(pl.col("forbidden")).then(-2.0).otherwise(-1.0))
        )
        .drop("upsilon")
    )


def write_output_files(atomic_number: int, iondatalist: list[IonData], args: argparse.Namespace) -> None:
    """Append one element's ions to adata.txt, transitiondata.txt and phixsdata_v2.txt.

    Every non-top ion's photoionization_targetfractions must already be filled in by
    resolve_photoion_targetfractions() (in iondata), which this does not call itself. Writing an
    ion that still needs resolving would silently drop its cross sections, so that is rejected.
    """
    outdir = Path(args.output_folder)
    log_folder = outdir / args.output_folder_logs

    # a level's photoionisation threshold reaches into the ion above it, so keep the whole
    # element to hand rather than only the ion being written
    iondata_of_ion_stage = {iondata.ion_stage: iondata for iondata in iondatalist}

    for iondata in iondatalist:
        ion_stage = iondata.ion_stage
        upsilondict = iondata.upsilondict
        transition_count_of_level_name = iondata.transition_count_of_level_name
        ionstr = f"{elsymbols[atomic_number]} {roman_numerals[ion_stage]}"

        with ion_log_path(log_folder, atomic_number, ion_stage).open("a", encoding="utf-8") as flog:
            log_and_print(flog, f"\n===========> Z={atomic_number} {ionstr} output:")

            dfenergylevels_ion = iondata.dfenergylevels
            dftransitions_ion = iondata.dftransitions

            if dftransitions_ion.is_empty():
                unused_upsilon_transitions = set()
            else:
                dftransitions_ion = add_level_ids_forbidden(dfenergylevels_ion, dftransitions_ion)
                log_deltaj_contradictions(flog, dftransitions_ion, ionstr)
                unused_upsilon_transitions = set(upsilondict.keys()).difference(
                    dftransitions_ion[["lowerlevel", "upperlevel"]].iter_rows(named=False)
                )

            log_and_print(
                flog, f"Adding in {len(unused_upsilon_transitions):d} extra transitions with only upsilon values"
            )

            if unused_upsilon_transitions:
                dfupsilon_only_transitions = pl.DataFrame(
                    list(unused_upsilon_transitions),
                    schema=(("lowerlevel", pl.Int64), ("upperlevel", pl.Int64)),
                    orient="row",
                ).with_columns(A=0.0)
                for id_lower, id_upper in dfupsilon_only_transitions[["lowerlevel", "upperlevel"]].iter_rows(
                    named=False
                ):
                    namefrom = dfenergylevels_ion["levelname"][id_upper]
                    nameto = dfenergylevels_ion["levelname"][id_lower]

                    transition_count_of_level_name[namefrom] += 1
                    transition_count_of_level_name[nameto] += 1

                dfupsilon_only_transitions = add_level_ids_forbidden(dfenergylevels_ion, dfupsilon_only_transitions)
                dftransitions_ion = pl.concat([dftransitions_ion, dfupsilon_only_transitions], how="diagonal_relaxed")

            if not dftransitions_ion.is_empty():
                # a left join rather than a per-row map_elements(): this runs over every
                # transition of the ion (2.6M of them for the cmfgen set), where a Python callback
                # per row costs more than the whole rest of the write. upsilondict's keys are
                # unique, so the join cannot duplicate rows, and maintain_order keeps the frame in
                # the order the reader produced it.
                dfupsilon = pl.DataFrame(
                    [(lower, upper, upsilon) for (lower, upper), upsilon in upsilondict.items()],
                    schema={"lowerlevel": pl.Int64, "upperlevel": pl.Int64, "upsilon": pl.Float64},
                    orient="row",
                )
                dftransitions_ion = (
                    dftransitions_ion.with_columns(
                        pl.col("lowerlevel").cast(pl.Int64), pl.col("upperlevel").cast(pl.Int64)
                    )
                    .join(dfupsilon, on=["lowerlevel", "upperlevel"], how="left", maintain_order="left")
                    .pipe(resolve_coll_str)
                )

            with (outdir / "adata.txt").open("a", encoding="utf-8") as fatommodels:
                write_adata(
                    fatommodels,
                    atomic_number,
                    ion_stage,
                    dfenergylevels_ion,
                    iondata.ionization_energy_ev,
                    transition_count_of_level_name,
                    flog,
                )

            dftransitions_ion = (
                dftransitions_ion
                if dftransitions_ion.is_empty()
                else dftransitions_ion.sort(by=("lowerlevel", "upperlevel"))
            )
            log_degenerate_transitions(flog, dfenergylevels_ion, dftransitions_ion)
            with (outdir / "transitiondata.txt").open("a", encoding="utf-8") as ftransitiondata:
                write_transition_data(
                    ftransitiondata,
                    atomic_number,
                    ion_stage,
                    dftransitions_ion,
                    flog,
                )

            if not iondata.is_top_ion and not args.nophixs:
                # an ion with cross sections but no targets has not been through
                # resolve_photoion_targetfractions(), and every one of its tables would be
                # dropped without a word by write_phixs_data()
                if len(iondata.photoionization_crosssections) > 0 and not iondata.photoionization_targetfractions:
                    msg = (
                        f"Z={atomic_number} ion_stage={ion_stage} has photoionization cross sections but no target"
                        " fractions: call resolve_photoion_targetfractions() before write_output_files()"
                    )
                    raise ValueError(msg)

                with (outdir / "phixsdata_v2.txt").open("a", encoding="utf-8") as fphixs:
                    write_phixs_data(
                        fphixs,
                        atomic_number,
                        ion_stage,
                        iondata.photoionization_crosssections,
                        iondata.photoionization_targetfractions,
                        fill_missing_phixs_thresholds(iondata, iondata_of_ion_stage.get(ion_stage + 1), flog),
                        args,
                        flog,
                    )


def write_adata(
    fatommodels,
    atomic_number: int,
    ion_stage: int,
    dfenergylevels: pl.DataFrame,
    ionization_energy: float,
    transition_count_of_level_name,
    flog,
) -> None:
    """Append one ion's level list to adata.txt.

    Level ids are zero-based in memory but numbered from one in the output. Each level line ends
    with the level's name as a free-text comment; artistools reads it back as everything after the
    fourth field, so it must not be padded.
    """
    log_and_print(flog, f"Writing {dfenergylevels.height} levels to 'adata.txt'")
    fatommodels.write(f"{atomic_number:12d}{ion_stage:12d}{dfenergylevels.height:12d}{ionization_energy:15.7f}\n")

    # every reader names its own levels, and that name is the whole level comment
    for energylevel in dfenergylevels.iter_rows(named=True):
        levelname = energylevel.get("levelname", "")
        transitioncount = transition_count_of_level_name.get(levelname, 0)

        # level ids are zero-based in memory, but the output format numbers them from one
        fatommodels.write(
            f"{energylevel['levelid'] + 1:5d} {hc_in_ev_cm * float(energylevel['energyabovegsinpercm']):19.16f} {float(energylevel['g']):8.3f} {transitioncount:4d} {levelname:}\n"
        )

    fatommodels.write("\n")


def log_degenerate_transitions(flog, dfenergylevels_ion: pl.DataFrame, dftransitions_ion: pl.DataFrame) -> None:
    """Report the transitions whose two levels have the same energy, which ARTIS drops.

    ARTIS gives every transition a frequency of (E_upper - E_lower) / h and skips the row where
    that is not positive (input.cc), so a pair of levels the source gives the same energy is
    silently lost, along with any collision strength it carried. Nothing here can fix that: a line
    of zero frequency has no place in the radiation field. It should not happen quietly, though.
    """
    if dftransitions_ion.is_empty() or "energyabovegsinpercm" not in dfenergylevels_ion.columns:
        return

    energy = dfenergylevels_ion.select("levelid", "energyabovegsinpercm")
    degenerate = (
        dftransitions_ion.join(
            energy.select(pl.col("levelid").alias("lowerlevel"), pl.col("energyabovegsinpercm").alias("e_lower")),
            on="lowerlevel",
        )
        .join(
            energy.select(pl.col("levelid").alias("upperlevel"), pl.col("energyabovegsinpercm").alias("e_upper")),
            on="upperlevel",
        )
        .filter(pl.col("e_lower") == pl.col("e_upper"))
    )
    if degenerate.is_empty():
        return

    withcollstr = degenerate.filter(pl.col("coll_str") > 0.0).height if "coll_str" in degenerate.columns else 0
    log_and_print(
        flog,
        f"WARNING: {degenerate.height:d} transitions connect two levels of the same energy"
        f" ({withcollstr:d} of them with a collision strength). ARTIS gives every transition a"
        " frequency from the level energies and drops the ones that come out at zero, so these"
        " are written but not used.",
    )


def write_transition_data(
    ftransitiondata,
    atomic_number: int,
    ion_stage: int,
    dftransitions_ion: pl.DataFrame,
    flog,
) -> None:
    """Append one ion's transitions to transitiondata.txt.

    Level ids are zero-based in memory but numbered from one in the output, and every transition
    is written with the lower id first.
    """
    log_and_print(flog, f"Writing {dftransitions_ion.height} transitions to 'transitiondata.txt'")

    # ARTIS reads the two ids as lower then upper, so a reversed pair would be a different
    # transition. Checked over the whole frame before the header goes out, so a bad row fails
    # before anything is written rather than leaving a block whose count overstates its rows.
    # Not an assert: this guards written output and must survive python -O.
    if not dftransitions_ion.is_empty():
        misordered = dftransitions_ion.filter(pl.col("lowerlevel") >= pl.col("upperlevel"))
        if not misordered.is_empty():
            levelid_lower, levelid_upper = misordered.select("lowerlevel", "upperlevel").row(0)
            msg = (
                f"Z={atomic_number} ion_stage={ion_stage} has {misordered.height} transitions that are not"
                f" ordered with the lower level id first, e.g. {levelid_lower} -> {levelid_upper}"
            )
            raise ValueError(msg)

    ftransitiondata.write(f"{atomic_number:7d}{ion_stage:7d}{dftransitions_ion.height:12d}\n")

    if not dftransitions_ion.is_empty():
        # writelines() over a generator, not a write() per row: this loop runs 2.6M times for the
        # cmfgen set, where the per-call overhead is a measurable part of the whole run
        # level ids are zero-based in memory, but the output format numbers them from one
        ftransitiondata.writelines(
            f"{levelid_lower + 1:4d} {levelid_upper + 1:4d} {float(A):11.5e} {coll_str:9.2e} {forbidden:d}\n"
            for levelid_lower, levelid_upper, A, coll_str, forbidden in dftransitions_ion[
                ["lowerlevel", "upperlevel", "A", "coll_str", "forbidden"]
            ].iter_rows()
        )

    ftransitiondata.write("\n")

    num_forbidden_transitions = (
        0 if dftransitions_ion.is_empty() else dftransitions_ion.filter(pl.col("forbidden")).height
    )

    num_collision_strengths_applied = (
        0 if dftransitions_ion.is_empty() else dftransitions_ion.filter(pl.col("coll_str") > 0).height
    )

    log_and_print(
        flog,
        f"  output {dftransitions_ion.height:d} transitions of which {num_forbidden_transitions:d} are forbidden and"
        f" {num_collision_strengths_applied:d} have collision strengths",
    )


def fill_missing_phixs_thresholds(iondata: IonData, upperiondata: IonData | None, flog) -> npt.NDArray[np.float64]:
    """Work out a photoionisation threshold for the levels whose reader gave none.

    Returns the ion's threshold array with its unreadable entries replaced, and the rest as they
    were. This is the same quantity ARTIS derives in get_phixs_threshold(), where a level's
    epsilon carries the ionization energies of every stage below it:

        threshold = ionization energy of this ion + E(target level) - E(this level)

    with both level energies above their own ion's ground state. The target is the first one, as
    ARTIS uses phixstargetindex 0 for a level's continuum edge (input.cc).

    A reader marks a threshold it does not have in two ways, and both count as missing here: NaN,
    which is what the arrays start as, and a negative value, which readqubdata writes to say that
    the threshold comes from the level energies rather than from its cross-section table.

    A threshold that does not come out positive is left alone: the level is then at or above the
    continuum, which is not something a photoionisation edge can describe.
    """
    thresholds = iondata.photoionization_thresholds_ev.copy()
    missing = [
        levelid for levelid, threshold in enumerate(thresholds) if not np.isfinite(threshold) or threshold <= 0.0
    ]
    if not missing or upperiondata is None:
        return thresholds
    if "energyabovegsinpercm" not in iondata.dfenergylevels.columns:
        return thresholds
    if "energyabovegsinpercm" not in upperiondata.dfenergylevels.columns:
        return thresholds

    energy_ev = [hc_in_ev_cm * energy for energy in iondata.dfenergylevels["energyabovegsinpercm"]]
    upper_energy_ev = [hc_in_ev_cm * energy for energy in upperiondata.dfenergylevels["energyabovegsinpercm"]]

    filled = 0
    for levelid in missing:
        if levelid >= len(iondata.photoionization_targetfractions) or levelid >= len(energy_ev):
            continue
        targetlist = iondata.photoionization_targetfractions[levelid]
        if not targetlist:
            continue
        targetlevelid = targetlist[0][0]
        if targetlevelid >= len(upper_energy_ev):
            continue

        threshold_ev = iondata.ionization_energy_ev + upper_energy_ev[targetlevelid] - energy_ev[levelid]
        if threshold_ev > 0.0:
            thresholds[levelid] = threshold_ev
            filled += 1

    if filled:
        log_and_print(
            flog,
            f"Worked out a photoionization threshold for {filled} levels whose reader gave none,"
            " from the ionization energy and the two level energies, as ARTIS does",
        )
    return thresholds


def write_phixs_data(
    fphixs,
    atomic_number: int,
    ion_stage: int,
    photoionization_crosssections: npt.NDArray[np.float64],
    photoionization_targetfractions: list[list[tuple[int, float]]],
    photoionization_thresholds_ev: npt.NDArray[np.float64],
    args,
    flog,
) -> None:
    """Append one ion's photoionization cross sections to phixsdata_v2.txt.

    Every level with targets is written. Level ids, of this ion and of the upper ion's targets,
    are zero-based in memory but numbered from one in the output.

    The threshold energy is written for information only. ARTIS reads the column into a variable
    it does not use (input.cc) and always takes the threshold from the difference of the level
    energies instead, in get_phixs_threshold(). A level whose threshold could not be determined
    therefore has a perfectly usable cross section table, and dropping it would discard real data
    over a number the consumer ignores; such a level is written with a threshold of zero.
    """
    # the target fractions are filled in per level by the caller, but a reader that found no
    # photoionization data at all returns the cross-section and threshold arrays still empty, so
    # bound the ids by what those arrays actually hold rather than indexing off the end
    levelids_to_write = [
        levelid
        for levelid, targetlist in enumerate(photoionization_targetfractions)
        if targetlist and levelid < len(photoionization_crosssections) and levelid < len(photoionization_thresholds_ev)
    ]
    nothreshold = sum(1 for levelid in levelids_to_write if not np.isfinite(photoionization_thresholds_ev[levelid]))

    log_and_print(flog, f"Writing {len(levelids_to_write)} phixs tables to 'phixsdata_v2.txt'")
    if nothreshold:
        log_and_print(
            flog,
            f"{nothreshold} of them have no threshold energy, and are written with a threshold of"
            " zero. ARTIS takes the threshold from the level energies, so their cross sections are"
            " used in full.",
        )
    flog.write(
        f"Downsampling cross sections assuming T={args.optimaltemperature} Kelvin, "
        f"nphixspoints={args.nphixspoints}, phixsnuincrement={args.phixsnuincrement}\n"
    )

    # only for a ground state that is actually being written: a level with no targets is
    # deliberately skipped (match_hydrogenic_phixs does this for a ground state at or above the
    # ionization energy), and that is not an error
    if 0 in levelids_to_write and photoionization_crosssections[0][0] == 0.0:
        msg = f"Z={atomic_number} ion_stage={ion_stage} ground state has zero photoionization cross section"
        log_and_print(flog, f"ERROR: {msg}")
        raise ValueError(msg)

    # level ids (of this ion and of the upper ion's photoionisation targets) are zero-based in
    # memory, but the output format numbers them from one
    for lowerlevelid in levelids_to_write:
        targetlist = photoionization_targetfractions[lowerlevelid]
        # zero where the reader could not determine one; ARTIS derives it from the level energies
        threshold_ev = photoionization_thresholds_ev[lowerlevelid]
        if not np.isfinite(threshold_ev):
            threshold_ev = 0.0
        if len(targetlist) == 1 and targetlist[0][1] > 0.99:
            upperionlevelid = targetlist[0][0]

            fphixs.write(
                f"{atomic_number:12d}{ion_stage + 1:12d}{upperionlevelid + 1:8d}{ion_stage:12d}{lowerlevelid + 1:8d}{threshold_ev:16.6E}\n"
            )
        else:
            fphixs.write(
                f"{atomic_number:12d}{ion_stage + 1:12d}{-1:8d}{ion_stage:12d}{lowerlevelid + 1:8d}{threshold_ev:16.6E}\n"
            )
            fphixs.write(f"{len(targetlist):8d}\n")
            probability_sum = 0.0
            for upperionlevelid, targetprobability in targetlist:
                fphixs.write(f"{upperionlevelid + 1:8d}{targetprobability:12f}\n")
                probability_sum += targetprobability
            if abs(probability_sum - 1.0) > 0.00001:
                msg = (
                    f"Z={atomic_number} ion_stage={ion_stage} level id {lowerlevelid}: phixs target fractions"
                    f" sum to {probability_sum:.5f} != 1.0 ({targetlist})"
                )
                raise ValueError(msg)

        # one writelines() per table rather than a write() per point: nphixspoints lines per level,
        # 1.5M of them for the cmfgen set
        fphixs.writelines(f"{crosssection:16.8E}\n" for crosssection in photoionization_crosssections[lowerlevelid])


def write_compositionfile(ion_handlers: list[tuple[int, list[tuple[int, str]]]], args: argparse.Namespace) -> None:
    """Write compositiondata.txt, listing each element's contiguous range of ion stages."""
    print("Writing compositiondata.txt")
    with (Path(args.output_folder) / "compositiondata.txt").open("w", encoding="utf-8") as fcomp:
        fcomp.write(f"{len(ion_handlers):d}\n")
        fcomp.write("0\n0\n")
        for atomic_number, listions in ion_handlers:
            listions_nohandlers: list[int] = drop_handlers(listions)
            ion_stage_min: int = 0
            ion_stage_max: int = 0
            nions: int = 0
            if listions_nohandlers:
                ion_stage_min = min(listions_nohandlers)
                ion_stage_max = max(listions_nohandlers)
                # the file gives only the range, so a gap in it would silently claim ions that were
                # never written. Not an assert: this guards written output and must survive -O.
                missing = [
                    ion_stage
                    for ion_stage in range(ion_stage_min, ion_stage_max + 1)
                    if ion_stage not in listions_nohandlers
                ]
                if missing:
                    msg = (
                        f"Missing ion stages {missing} for Z={atomic_number} between {ion_stage_min}"
                        f" and {ion_stage_max}"
                    )
                    raise ValueError(msg)
                nions = ion_stage_max - ion_stage_min + 1

            fcomp.write(
                f"{atomic_number:d}  {nions:d}  {ion_stage_min:d}  {ion_stage_max:d}  "
                f"-1 0.0 {atomic_weights[atomic_number]:.4f}\n"
            )
