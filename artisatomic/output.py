"""Write the ARTIS output files: adata.txt, transitiondata.txt, phixsdata_v2.txt, compositiondata.txt."""

import argparse
import sys
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
    on here; readers that already supply ids keep them. A transition is forbidden when its two
    levels have the same parity.
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
        dftransitions_ion = (
            dftransitions_ion.join(
                dfenergylevels_ion.select(
                    pl.col("levelid").alias("lowerlevel"), pl.col("parity").alias("lower_parity")
                ),
                on="lowerlevel",
            )
            .join(
                dfenergylevels_ion.select(
                    pl.col("levelid").alias("upperlevel"), pl.col("parity").alias("upper_parity")
                ),
                on="upperlevel",
            )
            .with_columns(forbidden=pl.col("lower_parity") == pl.col("upper_parity"))
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
                dftransitions_ion = dftransitions_ion.with_columns(
                    pl.struct(["lowerlevel", "upperlevel", "forbidden"])
                    .map_elements(
                        lambda row, upsilondict=upsilondict: upsilondict.get(  # type: ignore[misc]
                            (row["lowerlevel"], row["upperlevel"]),
                            -2.0 if row["forbidden"] else -1.0,
                        ),
                        return_dtype=pl.Float64,
                    )
                    .alias("coll_str")
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

    ftransitiondata.write(f"{atomic_number:7d}{ion_stage:7d}{dftransitions_ion.height:12d}\n")

    if not dftransitions_ion.is_empty():
        for levelid_lower, levelid_upper, A, coll_str, forbidden in dftransitions_ion[
            ["lowerlevel", "upperlevel", "A", "coll_str", "forbidden"]
        ].iter_rows():
            assert levelid_lower < levelid_upper

            # level ids are zero-based in memory, but the output format numbers them from one
            ftransitiondata.write(
                f"{levelid_lower + 1:4d} {levelid_upper + 1:4d} {float(A):11.5e} {coll_str:9.2e} {forbidden:d}\n"
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

    if len(photoionization_crosssections) >= 1 and photoionization_crosssections[0][0] == 0.0:
        log_and_print(flog, "ERROR: ground state has zero photoionization cross section")
        sys.exit()

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
                print(f"STOP! phixs fractions sum to {probability_sum:.5f} != 1.0")
                print(targetlist)
                print(f"level id {lowerlevelid}")
                sys.exit()

        for crosssection in photoionization_crosssections[lowerlevelid]:
            fphixs.write(f"{crosssection:16.8E}\n")

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
                assert all(ion_stage in listions_nohandlers for ion_stage in range(ion_stage_min, ion_stage_max + 1)), (
                    f"Missing ion stages for Z={atomic_number} between {ion_stage_min} and {ion_stage_max}"
                )
                nions = ion_stage_max - ion_stage_min + 1

            fcomp.write(
                f"{atomic_number:d}  {nions:d}  {ion_stage_min:d}  {ion_stage_max:d}  "
                f"-1 0.0 {atomic_weights[atomic_number]:.4f}\n"
            )
