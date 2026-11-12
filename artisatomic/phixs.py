"""Downsample photoionisation cross section tables and estimate hydrogenic ones where none exist."""

import math
import typing as t
from collections.abc import Callable
from functools import partial

import numpy as np
import numpy.typing as npt
import polars as pl

from artisatomic import readhillierdata
from artisatomic.base import elsymbols
from artisatomic.base import h_over_kb_in_K_sec
from artisatomic.base import hc_in_ev_angstrom
from artisatomic.base import hc_in_ev_cm
from artisatomic.base import log_and_print
from artisatomic.base import log_comment
from artisatomic.base import log_source
from artisatomic.base import output_xgrid
from artisatomic.base import parallel_map
from artisatomic.base import path_in_data_folder
from artisatomic.base import phixs_nu_cubed_tail
from artisatomic.base import ryd_to_hz


def match_hydrogenic_phixs(
    atomic_number: int,
    energy_levels: pl.DataFrame,
    ionization_energy_ev: float,
    ion_handler: str,
    get_level_valence_n: Callable[[str], int | None] | None,
    args,
    flog,
) -> tuple[npt.NDArray[np.float64], list[list[tuple[int, float]]], npt.NDArray[np.float64]]:
    """Estimate photoionisation cross sections for a data set that supplies none.

    This applies to any handler, not to one source only. The function assigns a hydrogenic cross
    section to each of the -nlevels_hydrogenic_for_unknown_phixs lowest levels by energy. It
    scales the cross section to that level's own ionisation threshold, with the upper ion's ground
    state as the only target.

    That option defaults to 100, so the estimate is on unless the user sets it to 0. The option
    bounds the levels considered and not the tables produced. A level at or above the ionisation
    energy gets no table but still counts towards the limit. The function sorts the levels by
    energy here, because a reader can keep its file's order.

    The caller reaches this function only for an ion whose handler returned no cross sections at
    all. An estimate therefore never replaces or extends real data. The granularity is the whole
    ion. An ion whose handler covered even one level keeps exactly the levels that the handler
    covered. The other levels get no photoionisation and no hydrogenic estimate.

    get_level_valence_n is the handler's own level-name parser. The handler registry in
    iondata.py holds it. None means that the handler has no parser. The ion then gets no
    estimate, and this function writes a warning.

    The parser returns None for a name it cannot read. Such a level gets no estimate, and the
    log file records it. The hydrogenic tables cover n = 1 to max_hyd_gaunt_n only. A level
    outside that range also gets no estimate, and the function does not read past the table.
    """
    if get_level_valence_n is None:
        log_and_print(
            flog,
            f"WARNING: no hydrogenic photoionisation cross sections, because no parser gives the principal"
            f" quantum number of a {ion_handler} level",
        )
        return np.empty((0, args.nphixspoints)), [], np.empty(0)

    log_and_print(
        flog,
        f"artisatomic uses hydrogenic photoionisation cross sections for Z={atomic_number} {elsymbols[atomic_number]}",
    )
    # This loads the tables on the first call. The range test below reads max_hyd_gaunt_n, which
    # is -1 before the load, and the loop would then skip every level as out of range.
    readhillierdata.read_hyd_phixsdata()

    photoionization_crosssections = np.zeros((energy_levels.height, args.nphixspoints))
    photoionization_targetfractions: list[list[tuple[int, float]]] = [[] for _ in range(energy_levels.height)]
    photoionization_thresholds_ev = np.full(energy_levels.height, np.nan)
    phixstables = {}
    # The lowest levels by energy, whatever order the reader kept them in. The stable sort keeps
    # levels of one energy in id order, and the code indexes every array here by level id.
    lowest_levels = energy_levels.sort("energyabovegsinpercm", maintain_order=True).head(
        args.nlevels_hydrogenic_for_unknown_phixs
    )
    # the counts of the levels that get no table, for one summary line in the comment block
    levels_above_ionization = 0
    levels_without_n = 0
    levels_outside_tables = 0
    for level in lowest_levels.iter_rows(named=True):
        levelindex = level["levelid"]
        en_ev = hc_in_ev_cm * level["energyabovegsinpercm"]
        threshold_ev = ionization_energy_ev - en_ev
        if threshold_ev <= 0.0:
            # level lies above the ionisation energy, so there is nothing to ionise from
            levels_above_ionization += 1
            continue

        n = get_level_valence_n(level["levelname"])
        if n is None:
            log_and_print(
                flog,
                f"WARNING: level name '{level['levelname']}' has no principal quantum number, so the level"
                " gets no hydrogenic cross section",
            )
            levels_without_n += 1
            continue
        if n < 1 or n > readhillierdata.max_hyd_gaunt_n:
            log_and_print(
                flog,
                f"WARNING: n={n} of level '{level['levelname']}' is outside the hydrogenic tables"
                f" (1 to {readhillierdata.max_hyd_gaunt_n}), so the level gets no hydrogenic cross section",
            )
            levels_outside_tables += 1
            continue

        photoionization_thresholds_ev[levelindex] = threshold_ev
        lambda_angstrom = hc_in_ev_angstrom / threshold_ev
        # get_hydrogenic_n_phixstable() already scales by the effective charge, since its
        # scale factor 7.91 / (E_threshold / Ryd) / n is the Kramers result 7.91 * n / Z_eff^2
        # (Kramers 1923, Phil. Mag., 46, 836-871, doi:10.1080/14786442308565244)
        phixstables[levelindex] = readhillierdata.get_hydrogenic_n_phixstable(lambda_angstrom=lambda_angstrom, n=n)
        photoionization_targetfractions[levelindex] = [(0, 1.0)]  # the upper ion's ground state

    # ARTIS stops for an ion with a table for an excited level and none for the ground level
    # (input.cc). The ground level is the first of the levels in energy order.
    if phixstables and lowest_levels["levelid"][0] not in phixstables:
        log_comment(
            flog,
            ("adata",),
            "WARNING: the ground level gets no hydrogenic photoionisation table, so artisatomic gives no table to any"
            " level of this ion. ARTIS needs a table for the ground level if an excited level has one.",
        )
        return np.empty((0, args.nphixspoints)), [], np.empty(0)

    reduced_phixs_dict = reduce_phixs_tables(
        phixstables,
        args.optimaltemperature,
        args.nphixspoints,
        args.phixsnuincrement,
        label=f"Z={atomic_number} {elsymbols[atomic_number]} hydrogenic estimate",
        interpolation=args.phixsinterpolation,
    )
    for levelindex, reduced_phixs_table in reduced_phixs_dict.items():
        photoionization_crosssections[levelindex] = reduced_phixs_table

    # only an ion that got a table names the estimate as its source
    if reduced_phixs_dict:
        gauntpath = path_in_data_folder(readhillierdata.hyd_gaunt_filename(), readhillierdata.hillier_folder)
        log_source(
            flog,
            ("phixsdata",),
            "the cross sections",
            "the hydrogenic estimate of artisatomic. It is the cross section of Kramers, H. A. (1923),"
            " Phil. Mag., 46, 836-871, doi:10.1080/14786442308565244, with the Gaunt factors of CMFGEN in"
            f" {gauntpath}",
        )

        skipped = levels_above_ionization + levels_without_n + levels_outside_tables
        if skipped:
            log_comment(
                flog,
                ("phixsdata",),
                f"{skipped} of the lowest {lowest_levels.height} levels got no hydrogenic table:"
                f" {levels_above_ionization} are at or above the ionisation energy, {levels_without_n} have a level"
                f" name with no principal quantum number, and {levels_outside_tables} have an n outside the"
                " hydrogenic tables.",
            )

    return photoionization_crosssections, photoionization_targetfractions, photoionization_thresholds_ev


# a target below this share of the level's total drops out of the target list, with its route.
# Every multi-route target of the CMFGEN test sets is above 3%, so the value only acts on the QUB Co data.
PHIXS_TARGET_FRACTION_CUT = 0.02


class PhixsRoutes[TargetType](t.NamedTuple):
    """The one output table of a level and the targets of its routes, from combine_phixs_routes()."""

    table: npt.NDArray[np.float64]
    # the kept targets with their fractions, and the open targets with their factors
    fractions: list[tuple[TargetType, float]]
    factors: list[tuple[TargetType, float]]
    # the open targets below the cut
    dropped: list[tuple[TargetType, float]]


def combine_phixs_routes[TargetType](
    routes: list[tuple[TargetType, npt.NDArray[np.float64]]], fractioncut: float = PHIXS_TARGET_FRACTION_CUT
) -> PhixsRoutes[TargetType]:
    """Combine the routes of one level into one table and the fractions of its targets.

    Each route is a target and its reduced table. The output format carries one table for each
    level. ARTIS reads that table at the ratio of the frequency to the edge of each target, then
    applies the fraction of the target. The table is therefore the sum of the reduced tables.
    Each reduced table is on the ratio grid of its own route, which is the grid that ARTIS reads
    for that target. A target then gets its share of the total shape.

    The factor of a target is the sum of its reduced table. That sum is the integral of the
    cross section over the output grid, with the weights that build the table. A route whose
    table is zero everywhere is closed and drops out. A target below fractioncut of the factor
    sum drops out with its route. The strongest target always stays.

    The sum is exact for routes of one shape. For routes of different shapes it spreads the
    error over the targets, so no target gets a zero where its own route is open. With no open
    route the table is zero and the fraction list is empty.
    """
    if not routes:
        msg = "combine_phixs_routes() needs at least one route"
        raise ValueError(msg)
    openroutes = [(target, reduced, float(reduced.sum())) for target, reduced in routes if reduced.any()]
    factors = [(target, factor) for target, _, factor in openroutes]
    if not openroutes:
        return PhixsRoutes(np.zeros_like(routes[0][1]), [], [], [])
    factor_sum = sum(factor for _, factor in factors)
    largest = max(factor for _, factor in factors)
    keptroutes = [
        (target, reduced, factor)
        for target, reduced, factor in openroutes
        if factor == largest or factor / factor_sum > fractioncut
    ]
    kepttargets = {target for target, _, _ in keptroutes}
    keptfactor_sum = sum(factor for _, _, factor in keptroutes)
    # a new array: the caller hands out the reduced tables, so this function must not mutate them
    return PhixsRoutes(
        np.sum([reduced for _, reduced, _ in keptroutes], axis=0),
        [(target, factor / keptfactor_sum) for target, _, factor in keptroutes],
        factors,
        [(target, factor) for target, factor in factors if target not in kepttargets],
    )


def reduce_phixs_tables[KeyType](
    dicttables: dict[KeyType, npt.NDArray[np.float64]],
    optimaltemperature: float,
    nphixspoints: int,
    phixsnuincrement: float,
    label: str | None = None,
    interpolation: t.Literal["linear", "classic"] = "linear",
) -> dict[KeyType, npt.NDArray[np.float64]]:
    """Downsample each 2D table of (energy, cross section) points into a 1D array.

    The energies must be in Rydberg. The function reads the first (lowest) energy point as the
    threshold energy.

    The result keeps the key type: callers index the tables by level name or by level id.

    label names the source of the tables, for example "Z=27 Co II phot_data_A". A key alone does
    not say which ion or which file the table came from. The messages of the worker name both.
    """
    print(f"Processing {len(dicttables.keys()):d} phixs tables")

    # One call reduces many tables onto one grid. The worker gets that grid and builds none.
    xgrid = output_xgrid(nphixspoints, phixsnuincrement)

    return dict(
        zip(
            dicttables.keys(),
            parallel_map(
                partial(
                    reduce_phixs_tables_worker, optimaltemperature, xgrid, label=label, interpolation=interpolation
                ),
                dicttables.values(),
                dicttables.keys(),
            ),
            strict=True,
        )
    )


gauss_legendre_5 = tuple(zip(*np.polynomial.legendre.leggauss(5), strict=True))


def weight_moments(start: npt.NDArray[np.float64], width: npt.NDArray[np.float64], a: float) -> npt.NDArray[np.float64]:
    """Integrate E^2 exp(-a s) s^k over each segment, for k = 0, 1 and 2.

    E = start + s is the energy, and s goes from 0 to the width of the segment. The result has one
    row for each k.

    A segment narrower than 0.25 kT takes a Gauss-Legendre rule of five points, with an error below
    1e-12. A wider segment takes the exact integral. Each rule fails where the other works. Five
    points miss the weight at the low end of a segment that is many kT wide. The exact form loses
    digits in a narrow segment.
    """
    moments = np.zeros((3, len(width)))
    for node, nodeweight in gauss_legendre_5:
        offset = 0.5 * (node + 1.0) * width
        weight = 0.5 * nodeweight * width * (start + offset) ** 2 * np.exp(-a * offset)
        moments[0] += weight
        weight *= offset
        moments[1] += weight
        moments[2] += weight * offset

    wide = a * width > 0.25
    # a very high temperature has no wide segment, and a power of its a underflows to zero
    if not wide.any():
        return moments
    # exponential[k] is the integral of s^k exp(-a s): k! / a^(k+1) (1 - exp(-t) (1 + t + ... + t^k / k!)),
    # with t = a * width
    twidth = a * width[wide]
    exp_minus_t = np.exp(-twidth)
    exponential = []
    partialsum = np.zeros_like(twidth)
    term = np.ones_like(twidth)
    for k in range(5):
        if k > 0:
            term = term * twidth / k
        partialsum += term
        exponential.append(math.factorial(k) / a ** (k + 1) * (1.0 - exp_minus_t * partialsum))
    # E^2 = start^2 + 2 start s + s^2
    for k in range(3):
        moments[k, wide] = (
            start[wide] ** 2 * exponential[k] + 2.0 * start[wide] * exponential[k + 1] + exponential[k + 2]
        )
    return moments


def reduce_phixs_tables_worker(
    optimaltemperature: float,
    xgrid: npt.NDArray[np.float64],
    tablein: np.ndarray,
    key: object = None,
    label: str | None = None,
    interpolation: t.Literal["linear", "classic"] = "linear",
) -> np.ndarray:
    """Downsample one cross section table onto the output's nu/nu_edge grid.

    ARTIS builds a cross section function from the output points. The output values make the
    integral of that function equal to the integral of the input, with the weight
    nu^2 exp(-h nu / k T). That weight preserves the recombination rate at optimaltemperature, and
    not the cross section itself.

    In linear mode, ARTIS joins the points with straight lines, and its recombination integral ends
    at the last point. Each value is the average of the input with the weight times the basis
    function of its point. That function is 1 at the point and falls linearly to 0 at each
    neighbour point. The basis functions sum to 1, so the averages preserve the integral.

    In classic mode, ARTIS holds each value constant from its point to the next point. Each value
    is the average of the input over that interval. The recombination integral of ARTIS ends at the
    last point also. ARTIS uses the last value above that point only, where it also scales the
    nu^-3 tail.

    Each exponential has the low end of its interval or of its basis function as the reference
    energy. A constant factor cancels in an average, and the absolute weight underflows to zero for
    h nu / k T > 745. That is every point above about 64 eV at -optimaltemperature 1000.

    xgrid is the nu/nu_edge grid of the output, which reduce_phixs_tables() builds once for the
    whole batch. It holds one point more than the output, to close the last interval of classic mode.

    key is the key of the table in the dict that reduce_phixs_tables() received, for example a
    level name. label names the source of the batch. The messages below name both.
    parallel_map() maps a batch of tables, so the caller cannot catch an error for one table and
    add the key itself.
    """
    labeltext = "" if label is None else f" The tables come from {label}."
    keytext = "" if key is None else f" The key of the table is {key!r}."
    npoints = len(xgrid) - 1

    if len(tablein) == 0:
        return np.zeros(npoints)

    threshold = tablein[0][0]
    # tablein is an array of pairs (energy, phixs cross section). Split it once, because numpy
    # re-slices a strided view of a 2D array every time.
    energies = np.ascontiguousarray(tablein[:, 0])
    sigmas = np.ascontiguousarray(tablein[:, 1])
    invalid = np.flatnonzero(~np.isfinite(sigmas) | (sigmas < 0.0))
    if invalid.size:
        row = int(invalid[0])
        msg = f"A source cross section is not finite and non-negative at row {row + 1}: {sigmas[row]}.{labeltext}{keytext}"
        raise ValueError(msg)
    # A zero threshold means that the level has no cross section.
    if threshold == 0.0:
        return np.zeros(npoints)
    # not an assert: np.searchsorted() and np.interp() below both give a wrong result for a
    # table that decreases in energy. This function also reads the first energy as the threshold.
    if np.any(np.diff(energies) < 0.0):
        msg = (
            f"The energy column of a photoionisation table decreases. The table shape is {tablein.shape}"
            f" and the first energy is {threshold:.6e} Ryd.{labeltext}{keytext}"
        )
        raise ValueError(msg)

    # the ends of the intervals between the points, in Rydberg
    edges = (xgrid if interpolation == "classic" else xgrid[:-1]) * threshold
    if energies[-1] < edges[-1]:
        # The nu^-3 tail above the table is not linear. It gets 50 points for each step of the output.
        ntail = math.ceil(50 * (edges[-1] - energies[-1]) / (edges[1] - edges[0]))
        tail = np.linspace(energies[-1], edges[-1], num=ntail + 1)[1:]
        sigmas = np.append(sigmas, phixs_nu_cubed_tail(sigmas[-1], energies[-1], tail))
        energies = np.append(energies, tail)

    # The grid holds each table sample and each edge. The cross section is then linear on each
    # segment between two grid points, and each segment belongs to one interval. A table can give
    # two cross sections at one energy (a step). The samples keep their order, and an edge at the
    # energy of a sample would split the step.
    newedges = edges[~np.isin(edges, energies)]
    grid = np.concatenate((energies, newedges))
    sigma = np.concatenate((sigmas, np.interp(newedges, energies, sigmas)))
    order = np.argsort(grid, kind="stable")
    order = order[: np.searchsorted(grid[order], edges[-1], side="right")]
    grid = grid[order]
    sigma = sigma[order]

    start = grid[:-1]
    width = np.diff(grid)
    slope = np.divide(np.diff(sigma), width, out=np.zeros_like(width), where=width > 0.0)
    # a step at the last edge gives a segment of zero width that starts there
    interval = np.minimum(np.searchsorted(edges, start, side="right") - 1, len(edges) - 2)
    low = edges[interval]
    high = edges[interval + 1]
    a = h_over_kb_in_K_sec * ryd_to_hz / optimaltemperature

    def point_integrals(moment0: npt.NDArray[np.float64], moment1: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Add the segment integrals of each point. moment1 has the extra factor s of its segment."""
        if interpolation == "classic":
            return np.bincount(interval, weights=moment0, minlength=npoints)
        # the basis function of the low point is (high - E) / (high - low), and E = start + s
        to_low_point = ((high - start) * moment0 - moment1) / (high - low)
        to_high_point = ((start - low) * moment0 + moment1) / (high - low)
        # the reference energy of a point is the point below it, which is one interval below `low`
        step_factor = np.append(1.0, np.exp(-a * np.diff(edges)))
        return step_factor * np.bincount(interval, weights=to_low_point, minlength=npoints) + np.bincount(
            interval + 1, weights=to_high_point, minlength=npoints
        )

    # Large grid steps can overflow the weights even when every source value is finite. An average
    # is then NaN or infinite, and the check below stops the run, so numpy gives no warning here.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        weight0, weight1, weight2 = weight_moments(start, width, a) * np.exp(-a * (start - low))
        integralnosigma = point_integrals(weight0, weight1)
        # the cross section on a segment is sigma[:-1] + slope * s
        integralwithsigma = point_integrals(
            sigma[:-1] * weight0 + slope * weight1, sigma[:-1] * weight1 + slope * weight2
        )
        averages = integralwithsigma / integralnosigma
    if not (np.all(integralnosigma > 0.0) and np.all(np.isfinite(averages)) and np.all(averages >= 0.0)):
        msg = (
            f"A photoionisation bin average is not a finite number of 0 or more. The table shape is"
            f" {tablein.shape}, the threshold energy is {threshold:.6e} Ryd, the smallest weighted"
            f" integral is {integralwithsigma.min():.6e} and the smallest weight integral is"
            f" {integralnosigma.min():.6e}. Check the cross sections, -phixsnuincrement and"
            f" -optimaltemperature.{labeltext}{keytext}"
        )
        raise ValueError(msg)
    return averages
