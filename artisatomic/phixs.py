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
        interpolation=getattr(args, "phixsinterpolation", "linear"),
        label=f"Z={atomic_number} {elsymbols[atomic_number]} hydrogenic estimate",
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
    interpolation: str = "linear",
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


gauss_legendre_3 = ((-np.sqrt(0.6), 5.0 / 9.0), (0.0, 8.0 / 9.0), (np.sqrt(0.6), 5.0 / 9.0))


def weighted_segment_moments(energy, sigma, temperature: float):
    """Integrate the source and its first moment over each linear segment.

    The weight is E^2 exp(-a s), where s starts at the low end of each segment.
    Each result has the zeroth and first moment, first without sigma and then with sigma.
    """
    start = energy[:-1]
    width = np.diff(energy)
    sigma_start = sigma[:-1]
    slope = np.divide(np.diff(sigma), width, out=np.zeros_like(width), where=width > 0.0)
    a = h_over_kb_in_K_sec * ryd_to_hz / temperature
    weight0, weight1, source0, source1 = (np.zeros_like(width) for _ in range(4))
    for node, nodeweight in gauss_legendre_3:
        offset = 0.5 * (node + 1.0) * width
        weight = 0.5 * nodeweight * width * (start + offset) ** 2 * np.exp(-a * offset)
        source = weight * (sigma_start + slope * offset)
        weight0 += weight
        weight1 += weight * offset
        source0 += source
        source1 += source * offset

    # The exact expression loses digits for a narrow segment. Three quadrature points miss a wide exponential.
    wide = a * width > 0.25
    if not np.any(wide):
        return weight0, weight1, source0, source1
    twidth = a * width[wide]
    exp_minus_t = np.exp(-twidth)
    moments = []
    partialsum = np.zeros_like(twidth)
    term = np.ones_like(twidth)
    for k in range(5):
        if k > 0:
            term = term * twidth / k
        partialsum += term
        moments.append(math.factorial(k) / a ** (k + 1) * (1.0 - exp_minus_t * partialsum))
    energy0 = start[wide]
    weighted = [energy0**2 * moments[k] + 2 * energy0 * moments[k + 1] + moments[k + 2] for k in range(3)]
    weight0[wide], weight1[wide] = weighted[:2]
    source0[wide] = sigma_start[wide] * weighted[0] + slope[wide] * weighted[1]
    source1[wide] = sigma_start[wide] * weighted[1] + slope[wide] * weighted[2]
    return weight0, weight1, source0, source1


def reduce_phixs_tables_worker(
    optimaltemperature: float,
    xgrid: npt.NDArray[np.float64],
    tablein: np.ndarray,
    key: object = None,
    label: str | None = None,
    interpolation: str = "linear",
) -> np.ndarray:
    """Reduce a source table with the cross section representation that ARTIS uses.

    Linear mode averages the source against each linear basis function of the output grid.
    The basis functions sum to one. Their weighted averages therefore preserve the recombination integral.
    Classic mode averages each interval from one stored point to the next.
    Both modes integrate from the threshold to the last stored point, as ARTIS does.

    Each basis function uses its own reference energy in the exponential weight.
    This keeps a high threshold or a low temperature from causing numerical underflow.
    The final value in classic mode sets only the tail beyond the integration range.
    The extra point of xgrid closes no part of the ARTIS recombination integral.
    """
    if interpolation not in {"linear", "classic"}:
        msg = f"Unknown photoionisation interpolation mode: {interpolation}"
        raise ValueError(msg)
    npoints = len(xgrid) - 1
    if npoints < 2:
        msg = "A photoionisation table needs at least two points for the ARTIS recombination integral."
        raise ValueError(msg)
    labeltext = "" if label is None else f" The tables come from {label}."
    keytext = "" if key is None else f" The key of the table is {key!r}."
    if len(tablein) == 0 or tablein[0][0] == 0.0:
        return np.zeros(npoints)
    threshold = tablein[0, 0]
    energies = np.ascontiguousarray(tablein[:, 0])
    sigmas = np.ascontiguousarray(tablein[:, 1])
    if np.any(np.diff(energies) < 0.0):
        msg = f"The energy column of a photoionisation table decreases.{labeltext}{keytext}"
        raise ValueError(msg)
    edges = xgrid[:-1] * threshold
    # Extra points describe the power-law tail after the last source point.
    uniform = np.append(np.linspace(edges[:-1], edges[1:], num=50, endpoint=False, axis=-1), edges[-1])
    uniform_sigma = np.interp(uniform, energies, sigmas)
    beyond = uniform > energies[-1]
    uniform_sigma[beyond] = phixs_nu_cubed_tail(sigmas[-1], energies[-1], uniform[beyond])
    nsource = np.searchsorted(energies, edges[-1], side="right")
    # Repeated source energies describe a step. Keep both limits in their source order.
    keep_uniform = ~np.isin(uniform, energies[:nsource])
    grid = np.concatenate((energies[:nsource], uniform[keep_uniform]))
    sigma = np.concatenate((sigmas[:nsource], uniform_sigma[keep_uniform]))
    order = np.argsort(grid, kind="stable")
    grid, sigma = grid[order], sigma[order]
    interval = np.minimum(np.searchsorted(edges, grid[:-1], side="right") - 1, npoints - 2)
    a = h_over_kb_in_K_sec * ryd_to_hz / optimaltemperature
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        weight0, weight1, source0, source1 = weighted_segment_moments(grid, sigma, optimaltemperature)
        if interpolation == "classic":
            scale = np.exp(-a * (grid[:-1] - edges[interval]))
            denominator = np.bincount(interval, weights=weight0 * scale, minlength=npoints - 1)
            numerator = np.bincount(interval, weights=source0 * scale, minlength=npoints - 1)
            last = np.interp(edges[-1], energies, sigmas)
            if edges[-1] > energies[-1]:
                last = phixs_nu_cubed_tail(sigmas[-1], energies[-1], edges[-1])
            result = np.append(numerator / denominator, last)
        else:
            width = edges[interval + 1] - edges[interval]
            right_start = (grid[:-1] - edges[interval]) / width
            left_scale = np.exp(-a * (grid[:-1] - edges[np.maximum(interval - 1, 0)]))
            right_scale = np.exp(-a * (grid[:-1] - edges[interval]))
            denominator = np.bincount(
                interval, weights=((1 - right_start) * weight0 - weight1 / width) * left_scale, minlength=npoints
            ) + np.bincount(
                interval + 1, weights=(right_start * weight0 + weight1 / width) * right_scale, minlength=npoints
            )
            numerator = np.bincount(
                interval, weights=((1 - right_start) * source0 - source1 / width) * left_scale, minlength=npoints
            ) + np.bincount(
                interval + 1, weights=(right_start * source0 + source1 / width) * right_scale, minlength=npoints
            )
            result = numerator / denominator
    if not (np.all(denominator > 0.0) and np.all(np.isfinite(result)) and np.all(result >= 0.0)):
        msg = (
            "A photoionisation bin average is not a finite number of 0 or more."
            f" Check the cross sections, -phixsnuincrement and -optimaltemperature.{labeltext}{keytext}"
        )
        raise ValueError(msg)
    return result
