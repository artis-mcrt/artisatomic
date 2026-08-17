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


def add_level_ids_forbidden(dfenergylevels_ion: pl.DataFrame, dftransitions_ion: pl.DataFrame) -> pl.DataFrame:
    """Fill in whichever of lowerlevel, upperlevel and forbidden a reader did not supply.

    Readers that key their transitions by level name (namefrom/nameto) get the level ids joined
    on here; readers that already supply ids keep them. A transition is forbidden when both of
    its levels have a known parity and it is the same one.

    A negative parity means the level has no definite parity, either because it merges sub-levels
    of both parities (CMFGEN's '1___' and '2s2_13w_2W') or because the reader could not read one
    from the level name. Those never count as a match: equal parity is only evidence of
    forbiddenness when the parities are real. Anything a reader could not give a whole number
    for -- a null, a NaN, a string pandas inferred from a blank column -- means the same thing,
    so the column is normalised to Int64 first and unreadable values join the negatives.
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
        # -1 for anything that is not a whole number: a NaN would otherwise compare equal to
        # itself and pass the >= 0 test, and a null would make forbidden itself null, which
        # write_transition_data's "{forbidden:d}" cannot format.
        knownparity = pl.col("parity").cast(pl.Int64, strict=False).fill_null(-1)
        dftransitions_ion = (
            dftransitions_ion.join(
                dfenergylevels_ion.select(pl.col("levelid").alias("lowerlevel"), knownparity.alias("lower_parity")),
                on="lowerlevel",
            )
            .join(
                dfenergylevels_ion.select(pl.col("levelid").alias("upperlevel"), knownparity.alias("upper_parity")),
                on="upperlevel",
            )
            .with_columns(forbidden=(pl.col("lower_parity") == pl.col("upper_parity")) & (pl.col("lower_parity") >= 0))
        )
    return dftransitions_ion


def write_output_files(atomic_number: int, iondatalist: list[IonData], args: argparse.Namespace) -> None:
    """Append one element's ions to adata.txt, transitiondata.txt and phixsdata_v2.txt.

    Every non-top ion's photoionization_targetfractions must already be filled in by
    resolve_photoion_targetfractions() (in iondata), which this does not call itself. Writing an
    ion that still needs resolving would silently drop its cross sections, so that is rejected.
    """
    outdir = Path(args.output_folder)
    log_folder = outdir / args.output_folder_logs

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
                    .with_columns(
                        # no upsilon for this transition: -2 marks it forbidden, -1 unknown
                        coll_str=pl.col("upsilon").fill_null(pl.when(pl.col("forbidden")).then(-2.0).otherwise(-1.0))
                    )
                    .drop("upsilon")
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
                        iondata.photoionization_thresholds_ev,
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

    Only levels with both targets and a threshold energy are written; the rest are counted and
    reported. Level ids, of this ion and of the upper ion's targets, are zero-based in memory but
    numbered from one in the output.
    """
    # a level gets a table only if it has targets and a threshold energy. The threshold arrays
    # start as NaN and are filled in only for levels that got a table, so NaN means "no data".
    # A negative value is written deliberately (readqubdata): ARTIS reads it as "no value given"
    # and takes the threshold from the difference of the level energies instead.
    # the target fractions are filled in per level by the caller, but a reader that found no
    # photoionization data at all returns the cross-section and threshold arrays still empty, so
    # bound the ids by what those arrays actually hold rather than indexing off the end
    levelids_with_targets = [
        levelid
        for levelid, targetlist in enumerate(photoionization_targetfractions)
        if targetlist and levelid < len(photoionization_crosssections) and levelid < len(photoionization_thresholds_ev)
    ]
    levelids_to_write = [
        levelid for levelid in levelids_with_targets if not np.isnan(photoionization_thresholds_ev[levelid])
    ]
    skipped_no_threshold = len(levelids_with_targets) - len(levelids_to_write)

    log_and_print(flog, f"Writing {len(levelids_to_write)} phixs tables to 'phixsdata_v2.txt'")
    flog.write(
        f"Downsampling cross sections assuming T={args.optimaltemperature} Kelvin, "
        f"nphixspoints={args.nphixspoints}, phixsnuincrement={args.phixsnuincrement}\n"
    )

    # only for a ground state that is actually being written: a level with no targets, or none
    # left after the threshold filter, is deliberately skipped (match_hydrogenic_phixs does this
    # for a ground state at or above the ionization energy), and that is not an error
    if 0 in levelids_to_write and photoionization_crosssections[0][0] == 0.0:
        msg = f"Z={atomic_number} ion_stage={ion_stage} ground state has zero photoionization cross section"
        log_and_print(flog, f"ERROR: {msg}")
        raise ValueError(msg)

    # level ids (of this ion and of the upper ion's photoionisation targets) are zero-based in
    # memory, but the output format numbers them from one
    for lowerlevelid in levelids_to_write:
        targetlist = photoionization_targetfractions[lowerlevelid]
        threshold_ev = photoionization_thresholds_ev[lowerlevelid]
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

    if skipped_no_threshold > 0:
        log_and_print(
            flog,
            f"Skipped {skipped_no_threshold} levels with no photoionization threshold energy",
        )


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
