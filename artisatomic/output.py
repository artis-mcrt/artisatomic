"""Write the ARTIS output files: adata.txt, transitiondata.txt, phixsdata_v2.txt, compositiondata.txt."""

import argparse
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

from artisatomic.base import atomic_weights
from artisatomic.base import check_ion_stages_contiguous
from artisatomic.base import comment_lines
from artisatomic.base import creation_time_utc
from artisatomic.base import drop_handlers
from artisatomic.base import elsymbols
from artisatomic.base import hc_in_ev_cm
from artisatomic.base import ion_label
from artisatomic.base import IonLog
from artisatomic.base import log_and_print
from artisatomic.base import log_comment
from artisatomic.base import log_path
from artisatomic.base import roman_numerals
from artisatomic.base import transition_count_of_level
from artisatomic.iondata import IonData
from artisatomic.phixs import PHIXS_TARGET_FRACTION_CUT

# The file comments. Each one must explain the format of its file to a reader who has only that
# file. A blank line here gives a line with only a # in the file. clear_files() gives each name in
# braces its value.
file_comment_end = """
COMMENTS
A line that starts with # is a comment. ARTIS skips a comment only directly before a header line.
{commentrule}A comment block before {blockposition} of each ion gives the handler, the source of the data and its
source files. A source file name in a comment omits the compression extension (.zst, .gz or .xz) that
the file on the disk can have.
"""

adata_file_comment = (
    """\
adata.txt: the energy levels of each ion, for the radiative transfer code ARTIS.
artisatomic (https://github.com/artis-mcrt/artisatomic) wrote this file at {creationtime} (UTC).

FORMAT
The file has one block for each ion. A blank line ends each block. ARTIS reads the file from the top
to the bottom one time. The blocks must therefore be in the order of compositiondata.txt. That order is
the elements in the order of that file, and the ion stages of an element from the lowest to the highest.
  header line:  Z  ion_stage  nlevels  ionisation_energy
  level line:   level_number  energy  g  ntransitions  level_name    (nlevels lines)

Z                  atomic number
ion_stage          1 for the neutral atom, 2 for the first ion, and so on
nlevels            count of the level lines of the block
ionisation_energy  energy from the ground level of this ion to the ground level of the next ion [eV]
level_number       position of the level line in its block. The first level of each ion has the number 1
                   (not 0). transitiondata.txt and phixsdata_v2.txt name a level by this number.
energy             energy above the ground level of the ion [eV]
g                  statistical weight
ntransitions       count of the lines of transitiondata.txt that name this level as lower or upper level
level_name         free text to the end of the line. ARTIS does not read it.

NUMBERS START AT 1
ion_stage and level_number start at 1, and not at 0. The level numbers start again at 1 for each ion.
"""
    + file_comment_end
)

transitiondata_file_comment = (
    """\
transitiondata.txt: the bound-bound transitions of each ion, for the radiative transfer code ARTIS.
artisatomic (https://github.com/artis-mcrt/artisatomic) wrote this file at {creationtime} (UTC).

FORMAT
The file has one block for each ion, in the order of adata.txt. A blank line ends each block.
  header line:      Z  ion_stage  ntransitions
  transition line:  lower  upper  A  coll_str  forbidden    (ntransitions lines)

Z             atomic number
ion_stage     1 for the neutral atom, 2 for the first ion, and so on
ntransitions  count of the transition lines of the block
lower, upper  level_number of the lower level and of the upper level in adata.txt. The first level of
              an ion has the number 1 (not 0).
A             Einstein A coefficient of the spontaneous decay [s^-1]
coll_str      effective collision strength (upsilon) at the temperature closest to {electrontemperature} K
              (option -electrontemperature).
              -1: no value for a permitted transition. ARTIS then uses the formula of van Regemorter.
              -2: no value for a forbidden transition. ARTIS then uses the approximation of Axelrod.
forbidden     1 for a forbidden transition, 0 for a permitted transition

NUMBERS START AT 1
ion_stage, lower and upper start at 1, and not at 0. The numbers -1 and -2 in coll_str are not level numbers.
"""
    + file_comment_end
)

phixsdata_file_comment = (
    """\
phixsdata_v2.txt: the photoionisation cross sections of each level, for the radiative transfer code ARTIS.
artisatomic (https://github.com/artis-mcrt/artisatomic) wrote this file at {creationtime} (UTC).

FORMAT
line 1:  npoints, the count of points of each table (option -nphixspoints)
line 2:  step, the distance between two points as a fraction of the threshold frequency
         (option -phixsnuincrement)
Then the file has one table for each level that has a cross section.
  header line:  Z  upper_ion_stage  upper_level  lower_ion_stage  lower_level  threshold_energy
  target list:  only if upper_level is -1. One line with ntargets, then ntargets lines:
                upper_level  fraction
  table:        npoints lines with one cross section each

Z                 atomic number
lower_ion_stage   ion stage of the ion that absorbs the photon (1 for the neutral atom)
lower_level       level_number of that level in adata.txt. The first level of an ion has the number 1 (not 0).
upper_ion_stage   lower_ion_stage + 1
upper_level       level_number in adata.txt of the level of the upper ion that the photoionisation
                  goes to. -1 means more than one such level, and the target list then comes next.
fraction          fraction of the cross section that goes to this upper_level. The fractions of a
                  target list sum to 1 (see the target cut in OPTIONS OF THIS RUN).
threshold_energy  [eV]. ARTIS does not use it. It takes the threshold from the level energies.
cross section     [Mb] for the frequency nu_threshold * (1 + i * step). The first line of a table is
                  i = 0, and the last line is i = npoints - 1. Each value is an average over the
                  frequency bin around its frequency. The bin of i = 0 starts at the threshold.

NUMBERS START AT 1
lower_ion_stage, upper_ion_stage, lower_level and upper_level start at 1, and not at 0. An upper_level
of -1 is not a level number. Only the point number i of a table starts at 0.

OPTIONS OF THIS RUN
artisatomic downsamples each table of its data source to the grid of this file. It keeps the
recombination rate constant at T={optimaltemperature} K (option -optimaltemperature).
{hydrogenicrule}
A target level with less than {targetcut:.0%} of the cross section of a level is not in the target list of that
level. The fractions of the other target levels then sum to 1.
"""
    + file_comment_end
)


def clear_files(args: argparse.Namespace) -> None:
    """Start the output files again, each with its file comment. The writer appends the ions after it.

    The file comment gives the format of the file and the creation time in UTC. It also gives the
    options of the run that apply to all ions of the file. The three files of a run get the same
    time (see base.creation_time_utc() for the time of a checksum run).

    ARTIS skips a comment line before the first header line of an ion in adata.txt and
    transitiondata.txt. It reads the first two numbers of phixsdata_v2.txt with no comment skip,
    so the file comment of that file comes after them.

    The option --nophixs writes no phixsdata_v2.txt. The run removes the file of an earlier run in
    the same folder, because its level ids belong to that run's adata.txt.
    """
    outdir = Path(args.output_folder)
    # one time for the three files of the run
    creationtime = creation_time_utc()
    with (outdir / "adata.txt").open("w", encoding="utf-8") as fatommodels:
        fatommodels.writelines(
            comment_lines(
                adata_file_comment.format(
                    creationtime=creationtime, commentrule="", blockposition="the header line"
                ).splitlines()
            )
        )

    with (outdir / "transitiondata.txt").open("w", encoding="utf-8") as ftransitiondata:
        ftransitiondata.writelines(
            comment_lines(
                transitiondata_file_comment.format(
                    creationtime=creationtime,
                    electrontemperature=args.electrontemperature,
                    commentrule="",
                    blockposition="the header line",
                ).splitlines()
            )
        )

    if args.nophixs:
        (outdir / "phixsdata_v2.txt").unlink(missing_ok=True)
        return

    with (outdir / "phixsdata_v2.txt").open("w", encoding="utf-8") as fphixs:
        fphixs.write(f"{args.nphixspoints:d}\n")
        fphixs.write(f"{args.phixsnuincrement:14.7e}\n")
        if args.nlevels_hydrogenic_for_unknown_phixs > 0:
            hydrogenicrule = (
                "An ion whose handler gives no cross section can get a hydrogenic estimate for its lowest\n"
                f"{args.nlevels_hydrogenic_for_unknown_phixs} levels (option -nlevels_hydrogenic_for_unknown_phixs)."
                " The comment block of such an ion\n"
                "names the estimate as its source. The top ion of an element gets no table. An ion of a handler\n"
                "whose level names give no principal quantum number gets no table also."
            )
        else:
            hydrogenicrule = (
                "The hydrogenic estimate is off (option -nlevels_hydrogenic_for_unknown_phixs 0). An ion whose\n"
                "handler gives no cross section has no table."
            )
        fphixs.writelines(
            comment_lines(
                phixsdata_file_comment.format(
                    creationtime=creationtime,
                    optimaltemperature=args.optimaltemperature,
                    hydrogenicrule=hydrogenicrule,
                    targetcut=PHIXS_TARGET_FRACTION_CUT,
                    commentrule="No comment can come before line 1 or line 2, because ARTIS reads those two"
                    " numbers with no comment skip.\n",
                    blockposition="the first table",
                ).splitlines()
            )
        )


# A transition this strong is an electric dipole line, whatever the level names say. Below these
# values, the code reads a line that breaks the delta J rule as a forbidden line that the source
# listed with its own small strength, which agrees with the label.
#
# The two need their own values because they are not the same quantity. f has no units. A is a
# rate in s-1 that spans many decades. Neither cut is a law of physics. A weak E1 line can lie
# well below either cut (an intercombination line, or one between high levels). A strong M1 or E2
# line in a highly charged ion can lie above the A cut.
#
# The cuts lie where the two populations separate in these data sets. Forbidden lines end near
# f ~ 1e-6 and A ~ 1e2 (the QUB Co III data peaks at 14 s-1). The lines that contradict their own J
# labels start at f = 7.8e-4 (F III's smallest), only eight times above the cut. A mislabelled
# line weaker than that falls on the wrong side, and that is the accepted cost.
min_f_asserts_e1 = 1e-4
min_a_asserts_e1 = 1e5


def strength_asserts_e1(dftransitions_ion: pl.DataFrame) -> pl.Expr:
    """Whether the source's own numbers claim the transition is an electric dipole line.

    The test uses f where the reader gives one, else the Einstein A. Both need a size, not only a
    value. A forbidden line has an A, and a reader that tabulates f gives one to its forbidden
    lines too. "Greater than zero" therefore says nothing about the kind of transition.
    """
    if "f" in dftransitions_ion.columns:
        return pl.col("f").fill_null(0.0).abs() > min_f_asserts_e1

    return pl.col("A").fill_null(0.0).abs() > min_a_asserts_e1


def add_level_ids_forbidden(dfenergylevels_ion: pl.DataFrame, dftransitions_ion: pl.DataFrame) -> pl.DataFrame:
    """Fill in whichever of lowerlevel, upperlevel and forbidden a reader did not supply.

    For readers that key their transitions by level name (namefrom/nameto), this function joins
    the level ids on. Readers that already supply ids keep them.

    Forbidden here means "not an electric dipole transition". The function checks two of the E1
    selection rules, each on the levels that carry the quantum number it needs:

    - Laporte: E1 changes the parity, so two levels of the same parity cannot be E1.
    - Delta J: E1 has |J_upper - J_lower| <= 1, and J = 0 -> J = 0 is forbidden outright.

    Either rule is sufficient on its own, so the function ORs the two. Neither rule can fire on a
    level that does not carry the quantum number it needs. A data set that gives no J gets the
    Laporte rule only.

    The delta J rule yields to a transition the source made strong enough to be E1, as
    strength_asserts_e1() judges it. Some files disagree with their own J labels. CMFGEN's
    provisional F III set splits a term by a nominal 0.8 cm-1 and then shares the term's f over
    all the J pairs. It therefore lists delta J = 2 lines with f as large as 0.116, and O II
    reaches 0.695.

    To call such a line forbidden would give a strong line the forbidden collision approximation,
    which is worse than the label it corrects. Instead, log_deltaj_contradictions() reports
    those. A weak line proves nothing, so the J labels decide it.

    A null parity means the level has no definite parity. Either the level merges sub-levels of
    both parities (CMFGEN's '1___' and '2s2_13w_2W'), or the reader could not read one from the
    level name. Null is the whole of that convention. polars resolves null == anything to null
    and not to true, so an absent parity cannot match another absent one. Readers need no
    sentinel number to spell it.

    A null J means the same. The casts below give null for a text value that a reader could not
    parse as a number. A NaN J stays NaN, and every comparison with NaN gives false. Each case
    disables only its own rule.
    """
    if dftransitions_ion.is_empty():
        return dftransitions_ion

    # the name-keyed joins keep the reader's row order (maintain_order) and fail on a height
    # change, as the id-keyed joins below do. An inner join drops a transition whose name matches
    # no level, and gives no message. A level name that two levels share multiplies its rows. Both
    # mean the reader's level list and transition list disagree. Not an assert: this guards
    # written output.
    for idcolumn, namecolumn in (("upperlevel", "nameto"), ("lowerlevel", "namefrom")):
        if idcolumn in dftransitions_ion.columns:
            continue
        height_before = dftransitions_ion.height
        dftransitions_ion = dftransitions_ion.join(
            dfenergylevels_ion.select(pl.col("levelid").alias(idcolumn), pl.col("levelname").alias(namecolumn)),
            on=namecolumn,
            maintain_order="left",
        )
        if dftransitions_ion.height != height_before:
            msg = (
                f"the {namecolumn} join changed the transition count from {height_before:d} to"
                f" {dftransitions_ion.height:d}: a level name in the transitions matches no level, or more than one"
            )
            raise ValueError(msg)

    if "forbidden" not in dftransitions_ion.columns:
        # The cast gives null for a text value, so a string column cannot raise in a comparison
        # with numbers. A NaN survives the float cast, and a comparison with NaN gives false.
        knownparity = pl.col("parity").cast(pl.Int64, strict=False)
        hasj = "j" in dfenergylevels_ion.columns
        knownj = pl.col("j").cast(pl.Float64, strict=False) if hasj else pl.lit(None, dtype=pl.Float64)

        assertse1 = strength_asserts_e1(dftransitions_ion)

        height_before = dftransitions_ion.height
        dftransitions_ion = dftransitions_ion.join(
            dfenergylevels_ion.select(
                pl.col("levelid").alias("lowerlevel"),
                knownparity.alias("lower_parity"),
                knownj.alias("lower_j"),
            ),
            on="lowerlevel",
            maintain_order="left",
        ).join(
            dfenergylevels_ion.select(
                pl.col("levelid").alias("upperlevel"),
                knownparity.alias("upper_parity"),
                knownj.alias("upper_j"),
            ),
            on="upperlevel",
            maintain_order="left",
        )
        # An inner join drops a transition whose id names no level, and gives no message.
        # adata.txt counts the transition anyway. An id-keyed reader that emits such an id has a
        # bug, so this fails rather than warns. Not an assert: this guards written output.
        if dftransitions_ion.height != height_before:
            msg = (
                f"{height_before - dftransitions_ion.height:d} transitions name a level id that is not one of the"
                f" {dfenergylevels_ion.height:d} levels"
            )
            raise ValueError(msg)

        dftransitions_ion = (
            dftransitions_ion
            # The delta J rule gets its own column, because log_deltaj_contradictions() reports
            # the transitions that break it while the source still gives them an f.
            .with_columns(
                breaksdeltaj=(
                    ((pl.col("lower_j") - pl.col("upper_j")).abs() > 1)
                    | ((pl.col("lower_j") == 0) & (pl.col("upper_j") == 0))
                ).fill_null(False)
            )
            # Each rule gets fill_null(False) on its own, before the or. A null would otherwise
            # spread and make forbidden itself null, which "{forbidden:d}" cannot format. A level
            # with no J would also undo what the two parities had already settled.
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
    transition. A large oscillator strength says that it is. The function add_level_ids_forbidden()
    lets the oscillator strength win, so the transition stays permitted. This function reports how
    often a data set needed that.

    CMFGEN's provisional F III set is the known example. It splits a term by a nominal 0.8 cm-1
    and shares the term's f over all the J pairs. A delta J = 2 line can then carry f = 0.116.
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
    log_comment(
        flog,
        ("transitiondata",),
        f"WARNING: {contradictions.height:d} transitions of {ionstr} break the delta J rule but"
        f" carry {strengthcol} > {minstrength:g} (largest {largest:.3g}). The level names and the"
        f" {strengthcol} values of this data set disagree. The output keeps the {strengthcol} values, so"
        f" these transitions stay permitted.",
    )


def resolve_coll_str(dftransitions_ion: pl.DataFrame) -> pl.DataFrame:
    """Turn the joined upsilon column into coll_str, and let a negative one set the forbidden flag.

    A negative upsilon is not a collision strength. It is the reader's mark for "this pair is
    forbidden and I have no value". readhillierdata writes -2 for the J pairs within a term. The
    mark therefore sets the flag, and not the parities alone: a merged term has no parity, so the
    parities alone would leave the pair permitted.

    Those pairs carry no A, so the van Regemorter formula (van Regemorter 1962, ApJ, 136, 906-915,
    doi:10.1086/147445) would give an oscillator strength of zero, which is no collisional coupling
    at all. The -2 asks instead for the approximation of Axelrod (1980, PhD thesis, University of
    California, Santa Cruz).

    coll_str then repeats what the flag says: -2 forbidden, -1 unknown. Only a missing upsilon
    reaches the -1, because a negative one has already made the flag true.
    """
    # the fill_null keeps the outer condition free of nulls: polars 1.44.0 sends a null-condition
    # row past the inner when to the innermost otherwise (pola-rs/polars#28498)
    return (
        dftransitions_ion.with_columns(forbidden=pl.col("forbidden") | (pl.col("upsilon") < 0.0).fill_null(False))
        .with_columns(
            coll_str=pl.when(pl.col("upsilon").fill_null(-1.0) >= 0.0)
            .then(pl.col("upsilon"))
            .otherwise(pl.when(pl.col("forbidden")).then(-2.0).otherwise(-1.0))
        )
        .drop("upsilon")
    )


def write_output_files(atomic_number: int, iondatalist: list[IonData], args: argparse.Namespace) -> None:
    """Append one element's ions to adata.txt, transitiondata.txt and phixsdata_v2.txt.

    Call this function one time for each element. It adds its comment lines to the lists of each
    IonData, so a second call would write those lines two times.

    resolve_photoion_targetfractions() (in iondata) must already have filled in every non-top
    ion's photoionization_targetfractions. This function does not call it. An ion that still
    needs the resolve pass would lose its cross sections without a message, so this function
    rejects it.
    """
    outdir = Path(args.output_folder)

    # A level's photoionisation threshold reaches into the ion above it, so keep the whole
    # element available and not only the current ion.
    iondata_of_ion_stage = {iondata.ion_stage: iondata for iondata in iondatalist}

    for iondata in iondatalist:
        ion_stage = iondata.ion_stage
        upsilondict = iondata.upsilondict
        ionstr = f"{elsymbols[atomic_number]} {roman_numerals[ion_stage]}"
        ionlabel = ion_label(atomic_number, ion_stage)

        # the "source:" line is one of the recorded comment lines, because a reader knows its source
        titlelines = (ionlabel, f"handler: {iondata.handler}")

        with log_path(outdir).open("a", encoding="utf-8") as logstream:
            # the comment lines of the read pass, so the lines of this pass go to the same lists
            flog = IonLog(logstream, iondata.comments)
            log_and_print(flog, f"\n===========> {ionlabel} output:")

            dfenergylevels_ion = iondata.dfenergylevels
            dftransitions_ion = iondata.dftransitions

            # One frame of the upsilon pairs for the whole ion. The anti join below finds the
            # pairs with no transition, and the left join after it attaches the values.
            dfupsilon = pl.DataFrame(
                [(lower, upper, upsilon) for (lower, upper), upsilon in upsilondict.items()],
                schema={"lowerlevel": pl.Int64, "upperlevel": pl.Int64, "upsilon": pl.Float64},
                orient="row",
            )

            if dftransitions_ion.is_empty():
                # a reader with no transitions gives a frame with no columns, which the joins in
                # add_level_ids_forbidden() cannot take. No transition uses an upsilon pair then,
                # so the upsilon-only mechanism below still writes the ion's collision strengths.
                dfupsilon_only_transitions = dfupsilon.select("lowerlevel", "upperlevel")
            else:
                dftransitions_ion = add_level_ids_forbidden(dfenergylevels_ion, dftransitions_ion).with_columns(
                    pl.col("lowerlevel").cast(pl.Int64), pl.col("upperlevel").cast(pl.Int64)
                )
                log_deltaj_contradictions(flog, dftransitions_ion, ionstr)
                # an anti join, not set.difference() over iter_rows(): that built a Python tuple
                # for each of the millions of transitions of a cmfgen ion
                dfupsilon_only_transitions = dfupsilon.select("lowerlevel", "upperlevel").join(
                    dftransitions_ion.select("lowerlevel", "upperlevel"), on=["lowerlevel", "upperlevel"], how="anti"
                )

            addedtext = (
                f"artisatomic added {dfupsilon_only_transitions.height:d} transitions with A = 0, for level pairs that"
                " have a collision strength but no transition in the data source."
            )
            # a count of zero tells nothing about the data, so only the log gets it
            if dfupsilon_only_transitions.is_empty():
                log_and_print(flog, addedtext)
            else:
                log_comment(flog, ("transitiondata",), addedtext)

            if not dfupsilon_only_transitions.is_empty():
                dfupsilon_only_transitions = dfupsilon_only_transitions.with_columns(A=0.0)
                dfupsilon_only_transitions = add_level_ids_forbidden(dfenergylevels_ion, dfupsilon_only_transitions)
                dftransitions_ion = pl.concat([dftransitions_ion, dfupsilon_only_transitions], how="diagonal_relaxed")

            if not dftransitions_ion.is_empty():
                # A left join and not a per-row map_elements(). This runs over every transition
                # of the ion (2.6M of them for the cmfgen set). A Python callback for each row
                # would cost more than the whole rest of the write. The keys of upsilondict are
                # unique, so the join cannot duplicate rows. maintain_order keeps the frame in the
                # order the reader produced it.
                dftransitions_ion = dftransitions_ion.join(
                    dfupsilon, on=["lowerlevel", "upperlevel"], how="left", maintain_order="left"
                ).pipe(resolve_coll_str)

            # the counts come from the final transition frame, so they include the upsilon-only
            # rows and agree with transitiondata.txt whatever the reader counted
            transition_counts = transition_count_of_level(dftransitions_ion, dfenergylevels_ion.height)
            with (outdir / "adata.txt").open("a", encoding="utf-8") as fatommodels:
                write_adata(
                    fatommodels,
                    atomic_number,
                    ion_stage,
                    dfenergylevels_ion,
                    iondata.ionization_energy_ev,
                    transition_counts,
                    flog,
                    titlelines,
                )

            # maintain_order: a reader can give one level pair several rows with different A
            # values (readkuruczdata keeps the distinct lines that share a pair). The default
            # sort places tied rows in an unspecified order that can change with the polars
            # version. The checksum tests compare the whole file.
            dftransitions_ion = (
                dftransitions_ion
                if dftransitions_ion.is_empty()
                else dftransitions_ion.sort(by=("lowerlevel", "upperlevel"), maintain_order=True)
            )
            log_degenerate_transitions(flog, dfenergylevels_ion, dftransitions_ion)
            with (outdir / "transitiondata.txt").open("a", encoding="utf-8") as ftransitiondata:
                write_transition_data(
                    ftransitiondata,
                    atomic_number,
                    ion_stage,
                    dftransitions_ion,
                    flog,
                    titlelines,
                )

            if not iondata.is_top_ion and not args.nophixs:
                # An ion with cross sections but no targets has not passed through
                # resolve_photoion_targetfractions(). write_phixs_data() would drop every one of
                # its tables without a message.
                if len(iondata.photoionization_crosssections) > 0 and not iondata.photoionization_targetfractions:
                    msg = (
                        f"Z={atomic_number} ion_stage={ion_stage} has photoionisation cross sections but no target"
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
                        titlelines,
                    )


def write_comment_block(fout, table: str, titlelines: Iterable[str], flog) -> None:
    """Write the comment block of one ion: the title lines, the source line, then the other recorded lines.

    titlelines name the ion and the handler. The other lines are the lines that the log holds for
    this file (see base.log_comment()). One of them must start with "source:", and no more than
    one. A function that gives data records that line, so a block with no such line means that a
    reader does not state its source.

    ARTIS skips a comment line only directly before the data header line of an ion in adata.txt
    and transitiondata.txt. In phixsdata_v2.txt it skips one only directly before the data header
    line of a table (get_noncommentline() in input.h). Inside the data of an ion ARTIS counts the
    lines, so a comment there would move every later line. A writer must therefore call this
    function directly before it writes such a data header line, and at no other position.

    The function writes only the title lines for a log that is not an IonLog.
    """
    if not isinstance(flog, IonLog):
        fout.writelines(comment_lines(titlelines))
        return

    recorded = flog.comments[table]
    sourcelines = [line for line in recorded if line.startswith("source:")]
    # not an assert: this guards written output and must survive python -O
    if len(sourcelines) != 1:
        msg = f"The {table} comment block of {' '.join(titlelines)} needs one source line but has {len(sourcelines)}"
        raise ValueError(msg)
    otherlines = [line for line in recorded if not line.startswith("source:")]
    fout.writelines(comment_lines([*titlelines, *sourcelines, *otherlines]))


def write_adata(
    fatommodels,
    atomic_number: int,
    ion_stage: int,
    dfenergylevels: pl.DataFrame,
    ionization_energy: float,
    transition_counts: list[int],
    flog,
    titlelines: Iterable[str] = (),
) -> None:
    """Append one ion's level list to adata.txt.

    Level ids are zero-based in memory but numbered from one in the output. transition_counts has
    one entry for each level id, in id order. Each level line ends with the level's name as free
    text. The artistools package reads it back as everything after the fourth field, so the writer
    must not pad it. The comment block of the ion comes before the header line (see
    write_comment_block()).
    """
    log_and_print(flog, f"artisatomic writes {dfenergylevels.height} levels to adata.txt.")
    write_comment_block(fatommodels, "adata", titlelines, flog)
    fatommodels.write(f"{atomic_number:12d}{ion_stage:12d}{dfenergylevels.height:12d}{ionization_energy:15.7f}\n")

    # the level name is the whole free text of the level line. A frame with no levelname column
    # gets an empty text.
    dfout = (
        dfenergylevels if "levelname" in dfenergylevels.columns else dfenergylevels.with_columns(levelname=pl.lit(""))
    )
    # Level ids are zero-based in memory. The output format numbers them from one.
    # writelines() with a generator, as in write_transition_data(). A write() call for each level
    # costs more than the formatting.
    fatommodels.writelines(
        # + 0.0 turns a -0.0 ground-state energy into 0.0, so the file never shows a minus sign
        f"{levelid + 1:5d} {hc_in_ev_cm * float(energyabovegsinpercm) + 0.0:19.16f} {float(g):8.3f} {transition_counts[levelid]:4d} {levelname:}\n"
        for levelid, energyabovegsinpercm, g, levelname in dfout.select(
            "levelid", "energyabovegsinpercm", "g", "levelname"
        ).iter_rows(named=False)
    )

    fatommodels.write("\n")


def log_degenerate_transitions(flog, dfenergylevels_ion: pl.DataFrame, dftransitions_ion: pl.DataFrame) -> None:
    """Report the transitions whose upper level is not above the lower one, which ARTIS drops.

    ARTIS gives every transition a frequency of (E_upper - E_lower) / h, and it skips the row
    where that is not positive (input.cc). A pair of levels with the same energy in the source
    thus loses its transition, and any collision strength it carried. Nothing here can fix that:
    a line of zero frequency has no place in the radiation field. This function reports it.

    A level list that is not in energy order gives the same loss for a pair whose lower id has
    the higher energy. The function reports that case too, on its own line.
    """
    if dftransitions_ion.is_empty() or "energyabovegsinpercm" not in dfenergylevels_ion.columns:
        return

    energy = dfenergylevels_ion.select("levelid", "energyabovegsinpercm")
    notabove = (
        dftransitions_ion.join(
            energy.select(pl.col("levelid").alias("lowerlevel"), pl.col("energyabovegsinpercm").alias("e_lower")),
            on="lowerlevel",
        )
        .join(
            energy.select(pl.col("levelid").alias("upperlevel"), pl.col("energyabovegsinpercm").alias("e_upper")),
            on="upperlevel",
        )
        .filter(pl.col("e_lower") >= pl.col("e_upper"))
    )
    if notabove.is_empty():
        return

    degenerate = notabove.filter(pl.col("e_lower") == pl.col("e_upper"))
    if not degenerate.is_empty():
        withcollstr = degenerate.filter(pl.col("coll_str") > 0.0).height if "coll_str" in degenerate.columns else 0
        log_comment(
            flog,
            ("transitiondata",),
            f"WARNING: {degenerate.height:d} transitions connect two levels of the same energy"
            f" ({withcollstr:d} of them with a collision strength). ARTIS computes the frequency of"
            " each transition from the level energies and drops a transition with a frequency of zero."
            " The output file has these transitions, but ARTIS does not use them.",
        )

    inverted = notabove.height - degenerate.height
    if inverted:
        log_comment(
            flog,
            ("transitiondata",),
            f"WARNING: {inverted:d} transitions have a lower level id whose energy is above the upper"
            " level's. The level list is not in energy order. ARTIS drops a transition with a"
            " negative frequency. The output file has these transitions, but ARTIS does not use them.",
        )


def write_transition_data(
    ftransitiondata,
    atomic_number: int,
    ion_stage: int,
    dftransitions_ion: pl.DataFrame,
    flog,
    titlelines: Iterable[str] = (),
) -> None:
    """Append one ion's transitions to transitiondata.txt.

    Level ids are zero-based in memory but numbered from one in the output. The writer lists the
    lower id of every transition first. The comment block of the ion comes before the header line
    (see write_comment_block()).
    """
    log_and_print(flog, f"artisatomic writes {dftransitions_ion.height} transitions to transitiondata.txt.")

    # ARTIS reads the two ids as lower then upper, so a reversed pair would be a different
    # transition. The check runs over the whole frame before the header goes out. A bad row
    # therefore fails before the writer writes any row, and no block has a count that overstates
    # its rows. Not an assert: this guards written output and must survive python -O.
    if not dftransitions_ion.is_empty():
        misordered = dftransitions_ion.filter(pl.col("lowerlevel") >= pl.col("upperlevel"))
        if not misordered.is_empty():
            levelid_lower, levelid_upper = misordered.select("lowerlevel", "upperlevel").row(0)
            msg = (
                f"Z={atomic_number} ion_stage={ion_stage} has {misordered.height} transitions that do not"
                f" name the lower level id first, e.g. {levelid_lower} -> {levelid_upper}"
            )
            raise ValueError(msg)

    # sums, and not a filtered copy of the frame for each count: a cmfgen ion has 2.6M rows
    num_forbidden_transitions, num_collision_strengths_applied = (
        (0, 0)
        if dftransitions_ion.is_empty()
        else dftransitions_ion.select(pl.col("forbidden").sum(), (pl.col("coll_str") > 0).sum()).row(0)
    )

    log_and_print(
        flog,
        f"  output {dftransitions_ion.height:d} transitions of which {num_forbidden_transitions:d} are forbidden and"
        f" {num_collision_strengths_applied:d} have collision strengths",
    )
    # The data header line gives the count of transitions, so the comment block gives the other
    # two. Two counts of zero tell nothing about the data. log_comment() and not add_comment(),
    # so a user finds each line of a comment block in the log file also.
    if num_forbidden_transitions or num_collision_strengths_applied:
        log_comment(
            flog,
            ("transitiondata",),
            f"{num_forbidden_transitions:d} transitions are forbidden, and {num_collision_strengths_applied:d}"
            " transitions have collision strengths.",
        )

    write_comment_block(ftransitiondata, "transitiondata", titlelines, flog)
    ftransitiondata.write(f"{atomic_number:7d}{ion_stage:7d}{dftransitions_ion.height:12d}\n")

    if not dftransitions_ion.is_empty():
        # One %-format over the column lists, not an f-string for each row from iter_rows(). This
        # runs 2.6M times for the cmfgen set and was the largest single cost of the build. The two
        # forms give the same bytes for every value. Level ids are zero-based in memory, but the
        # output format numbers them from one.
        columns = [
            (dftransitions_ion["lowerlevel"] + 1).to_list(),
            (dftransitions_ion["upperlevel"] + 1).to_list(),
            dftransitions_ion["A"].to_list(),
            dftransitions_ion["coll_str"].to_list(),
            dftransitions_ion["forbidden"].to_list(),
        ]
        ftransitiondata.writelines(map("%4d %4d %11.5e %9.2e %d\n".__mod__, zip(*columns, strict=True)))

    ftransitiondata.write("\n")


def threshold_is_known(threshold_ev: float) -> bool:
    """Whether a photoionisation threshold is a usable value: finite and positive.

    A reader marks a threshold it does not have as NaN. A non-positive value is not a
    threshold either, so the filler and the writer both treat it as missing.
    """
    return bool(np.isfinite(threshold_ev)) and threshold_ev > 0.0


def fill_missing_phixs_thresholds(iondata: IonData, upperiondata: IonData | None, flog) -> npt.NDArray[np.float64]:
    """Work out a photoionisation threshold for the levels whose reader gave none.

    Returns the ion's threshold array with its unreadable entries replaced, and the rest as they
    were. This is the same quantity ARTIS derives in get_phixs_threshold(), where a level's
    epsilon carries the ionisation energies of every stage below it:

        threshold = ionisation energy of this ion + E(target level) - E(this level)

    with both level energies above their own ion's ground state. The target is the first one, as
    ARTIS uses phixstargetindex 0 for a level's continuum edge (input.cc).

    A reader marks a threshold it does not have as NaN, which is the initial value of the arrays.
    A non-positive value also counts as missing here, so the function fills a mis-computed
    threshold and does not write it to the output.

    The function does not change a threshold that does not come out positive. The level is then
    at or above the continuum, which a photoionisation edge cannot describe.
    """
    thresholds = iondata.photoionization_thresholds_ev.copy()
    missing = [levelid for levelid, threshold in enumerate(thresholds) if not threshold_is_known(threshold)]
    if not missing or upperiondata is None:
        return thresholds
    if "energyabovegsinpercm" not in iondata.dfenergylevels.columns:
        return thresholds
    if "energyabovegsinpercm" not in upperiondata.dfenergylevels.columns:
        return thresholds

    # to_numpy() and not a Python loop over the Series. The ion can have 10^5 levels, and the
    # loop below reads only the missing ones.
    energy_ev = hc_in_ev_cm * iondata.dfenergylevels["energyabovegsinpercm"].to_numpy()
    upper_energy_ev = hc_in_ev_cm * upperiondata.dfenergylevels["energyabovegsinpercm"].to_numpy()

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

        threshold_ev = float(iondata.ionization_energy_ev + upper_energy_ev[targetlevelid] - energy_ev[levelid])
        if threshold_ev > 0.0:
            thresholds[levelid] = threshold_ev
            filled += 1

    if filled:
        log_comment(
            flog,
            ("phixsdata",),
            f"artisatomic computed the threshold energy of {filled} tables, because the reader gave none. The threshold"
            " comes from the ionisation energy and the two level energies, as in ARTIS.",
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
    titlelines: Iterable[str] = (),
) -> None:
    """Append one ion's photoionisation cross sections to phixsdata_v2.txt.

    The comment block of the ion comes before the header line of its first table (see
    write_comment_block()).

    The writer writes every level with targets. Level ids, of this ion and of the upper ion's
    targets, are zero-based in memory but numbered from one in the output.

    The threshold energy is for information only. ARTIS reads the column into a variable it does
    not use (input.cc). It always takes the threshold from the difference of the level energies
    instead, in get_phixs_threshold(). A level with an unknown threshold therefore has a usable
    cross section table. To drop it would discard real data because of a number that the consumer
    ignores. The writer writes such a level with a threshold of zero.
    """
    # The caller fills in the target fractions for each level. A reader that found no
    # photoionisation data at all returns empty cross section and threshold arrays. Bound the ids
    # by the length of those arrays, so the loop does not index past the end.
    levelids_to_write = [
        levelid
        for levelid, targetlist in enumerate(photoionization_targetfractions)
        if targetlist and levelid < len(photoionization_crosssections) and levelid < len(photoionization_thresholds_ev)
    ]
    # the same test as fill_missing_phixs_thresholds(): a non-positive threshold is not one
    nothreshold = sum(
        1 for levelid in levelids_to_write if not threshold_is_known(photoionization_thresholds_ev[levelid])
    )

    log_and_print(flog, f"artisatomic writes {len(levelids_to_write)} cross section tables to phixsdata_v2.txt.")
    if nothreshold:
        log_comment(
            flog,
            ("phixsdata",),
            f"{nothreshold} tables have no threshold energy, so the output gives them a threshold of"
            " zero. ARTIS then takes the threshold from the level energies and uses their cross sections"
            " in full.",
        )
    flog.write(
        f"Downsample of the cross sections with T={args.optimaltemperature} Kelvin, "
        f"nphixspoints={args.nphixspoints}, phixsnuincrement={args.phixsnuincrement}\n"
    )

    # Only for a ground state that the writer writes. The writer skips a level with no targets on
    # purpose, and that is not an error. An example is match_hydrogenic_phixs, which does this
    # for a ground state at or above the ionisation energy.
    if 0 in levelids_to_write and photoionization_crosssections[0][0] == 0.0:
        msg = f"Z={atomic_number} ion_stage={ion_stage} ground state has zero photoionisation cross section"
        log_and_print(flog, f"ERROR: {msg}")
        raise ValueError(msg)

    # ARTIS gives each entry of a target list its own target level and fraction (input.cc). A
    # repeated target level would get two fractions of the same recombination, and a bad
    # fraction sum would change the recombination total. The checks run over every level before
    # the first write, so a bad level fails before part of the ion goes out. Not an assert:
    # this guards written output and must survive python -O.
    for lowerlevelid in levelids_to_write:
        targetlist = photoionization_targetfractions[lowerlevelid]
        targetcounts = Counter(upperionlevelid for upperionlevelid, _ in targetlist)
        duplicates = sorted(upperionlevelid for upperionlevelid, count in targetcounts.items() if count > 1)
        if duplicates:
            msg = (
                f"Z={atomic_number} ion_stage={ion_stage} level id {lowerlevelid}: phixs target level ids"
                f" {duplicates} occur more than one time ({targetlist})"
            )
            log_and_print(flog, f"ERROR: {msg}")
            raise ValueError(msg)
        # the single-target form below implies a fraction of 1.0, so a single fraction must
        # be 1.0 as well
        probability_sum = sum(fraction for _, fraction in targetlist)
        if abs(probability_sum - 1.0) > 0.00001:
            msg = (
                f"Z={atomic_number} ion_stage={ion_stage} level id {lowerlevelid}: phixs target fractions"
                f" sum to {probability_sum:.5f} != 1.0 ({targetlist})"
            )
            log_and_print(flog, f"ERROR: {msg}")
            raise ValueError(msg)

    # An ion with no table gets no block. A block must come directly before a table header line.
    if levelids_to_write:
        write_comment_block(fphixs, "phixsdata", titlelines, flog)

    # level ids (of this ion and of the upper ion's photoionisation targets) are zero-based in
    # memory, but the output format numbers them from one
    for lowerlevelid in levelids_to_write:
        targetlist = photoionization_targetfractions[lowerlevelid]
        # zero where the reader could not determine one; ARTIS derives it from the level energies
        threshold_ev = photoionization_thresholds_ev[lowerlevelid]
        if not threshold_is_known(threshold_ev):
            threshold_ev = 0.0
        # the check above makes a single fraction 1.0, so one target takes the short form
        if len(targetlist) == 1:
            upperionlevelid = targetlist[0][0]

            fphixs.write(
                f"{atomic_number:12d}{ion_stage + 1:12d}{upperionlevelid + 1:8d}{ion_stage:12d}{lowerlevelid + 1:8d}{threshold_ev:16.6E}\n"
            )
        else:
            fphixs.write(
                f"{atomic_number:12d}{ion_stage + 1:12d}{-1:8d}{ion_stage:12d}{lowerlevelid + 1:8d}{threshold_ev:16.6E}\n"
            )
            fphixs.write(f"{len(targetlist):8d}\n")
            fphixs.writelines(
                f"{upperionlevelid + 1:8d}{targetprobability:12f}\n"
                for upperionlevelid, targetprobability in targetlist
            )

        # One write() for each table with a %-format over the points, not a generator for each
        # point. There are nphixspoints lines for each level, 1.5M of them for the cmfgen set.
        fphixs.write("".join(map("%16.8E\n".__mod__, photoionization_crosssections[lowerlevelid].tolist())))


def write_compositionfile(ion_handlers: list[tuple[int, list[tuple[int, str]]]], args: argparse.Namespace) -> None:
    """Write compositiondata.txt, which lists each element's contiguous range of ion stages."""
    print("Writing compositiondata.txt")
    check_ion_stages_contiguous(ion_handlers)
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
                nions = ion_stage_max - ion_stage_min + 1

            fcomp.write(
                f"{atomic_number:d}  {nions:d}  {ion_stage_min:d}  {ion_stage_max:d}  "
                f"-1 0.0 {atomic_weights[atomic_number]:.4f}\n"
            )
