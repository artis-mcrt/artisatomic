"""Downsample photoionisation cross section tables and estimate hydrogenic ones where none exist."""

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
from artisatomic.base import leveltuples_to_pldataframe
from artisatomic.base import log_and_print
from artisatomic.base import log_comment
from artisatomic.base import output_xgrid
from artisatomic.base import parallel_map
from artisatomic.base import path_for_log
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
    lowest_levels = (
        leveltuples_to_pldataframe(energy_levels)
        .sort("energyabovegsinpercm", maintain_order=True)
        .head(args.nlevels_hydrogenic_for_unknown_phixs)
    )
    for level in lowest_levels.iter_rows(named=True):
        levelindex = level["levelid"]
        en_ev = hc_in_ev_cm * level["energyabovegsinpercm"]
        threshold_ev = ionization_energy_ev - en_ev
        if threshold_ev <= 0.0:
            # level lies above the ionisation energy, so there is nothing to ionise from
            continue

        n = get_level_valence_n(level["levelname"])
        if n is None:
            log_and_print(
                flog,
                f"WARNING: level name '{level['levelname']}' has no principal quantum number, so the level"
                " gets no hydrogenic cross section",
            )
            continue
        if n < 1 or n > readhillierdata.max_hyd_gaunt_n:
            log_and_print(
                flog,
                f"WARNING: n={n} of level '{level['levelname']}' is outside the hydrogenic tables"
                f" (1 to {readhillierdata.max_hyd_gaunt_n}), so the level gets no hydrogenic cross section",
            )
            continue

        photoionization_thresholds_ev[levelindex] = threshold_ev
        lambda_angstrom = hc_in_ev_angstrom / threshold_ev
        # get_hydrogenic_n_phixstable() already scales by the effective charge, since its
        # scale factor 7.91 / (E_threshold / Ryd) / n is the Kramers result 7.91 * n / Z_eff^2
        # (Kramers 1923, Phil. Mag., 46, 836-871, doi:10.1080/14786442308565244)
        phixstables[levelindex] = readhillierdata.get_hydrogenic_n_phixstable(lambda_angstrom=lambda_angstrom, n=n)
        photoionization_targetfractions[levelindex] = [(0, 1.0)]  # the upper ion's ground state

    reduced_phixs_dict = reduce_phixs_tables(
        phixstables,
        args.optimaltemperature,
        args.nphixspoints,
        args.phixsnuincrement,
        label=f"Z={atomic_number} {elsymbols[atomic_number]} hydrogenic estimate",
    )
    for levelindex, reduced_phixs_table in reduced_phixs_dict.items():
        photoionization_crosssections[levelindex] = reduced_phixs_table

    # only an ion that got a table names the estimate as its source
    if reduced_phixs_dict:
        gauntpath = path_for_log(readhillierdata.hyd_gaunt_filename(), relative_to=readhillierdata.hillier_datadir)
        log_comment(
            flog,
            ("phixsdata",),
            "source: the hydrogenic estimate of artisatomic. It is the cross section of Kramers, H. A. (1923),"
            " Phil. Mag., 46, 836-871, doi:10.1080/14786442308565244, with the Gaunt factors of CMFGEN in"
            f" {gauntpath}",
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
                partial(reduce_phixs_tables_worker, optimaltemperature, xgrid, label=label),
                dicttables.values(),
                dicttables.keys(),
            ),
            strict=True,
        )
    )


def trapezoid_with_widths(arr_y: npt.NDArray[np.float64], arr_dx: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Integrate each row of arr_y with the trapezoid rule, over the sample widths arr_dx.

    np.trapezoid computes the widths from the x values at each call. The caller integrates two
    functions over one set of x values, so it computes the widths once. The result of this
    function is bit-identical to the result of np.trapezoid.
    """
    return np.sum(arr_dx * (arr_y[:, 1:] + arr_y[:, :-1]) / 2.0, axis=1)


def reduce_phixs_tables_worker(
    optimaltemperature: float,
    xgrid: npt.NDArray[np.float64],
    tablein: np.ndarray,
    key: object = None,
    label: str | None = None,
) -> np.ndarray:
    """Downsample one cross section table onto the output's nu/nu_edge grid.

    Each output point is the average of the input over that point's frequency bin, with the
    weight nu^2 exp(-h (nu - nu_low) / k T). The weight preserves the recombination rate at
    optimaltemperature, and not the cross section itself.

    nu_low is the lowest frequency of the bin. The constant factor exp(h nu_low / k T) cancels in
    the ratio of the two integrals, so the subtraction leaves the average unchanged. It also keeps
    the weight at 1.0 or below. The absolute weight underflows to zero for h nu / k T > 745, which
    is every bin above about 64 eV at -optimaltemperature 1000.

    xgrid is the nu/nu_edge grid of the output, which reduce_phixs_tables() builds once for the
    whole batch. It holds one point more than the output, to close the last bin.

    key is the key of the table in the dict that reduce_phixs_tables() received, for example a
    level name. label names the source of the batch. The messages below name both.
    parallel_map() maps a batch of tables, so the caller cannot catch an error for one table and
    add the key itself.
    """
    minus_h_over_kb_t = -h_over_kb_in_K_sec / optimaltemperature
    labeltext = "" if label is None else f" The tables come from {label}."
    keytext = "" if key is None else f" The key of the table is {key!r}."
    nphixspoints = len(xgrid) - 1

    # An empty table has no threshold to scale the grid, and a zero threshold would divide by
    # zero. Both therefore mean "no cross section", and neither raises an index error on tablein[0].
    if len(tablein) == 0 or tablein[0][0] == 0.0:
        return np.zeros(nphixspoints)

    threshold_old_ryd = tablein[0][0]
    # tablein is an array of pairs (energy, phixs cross section). Split it once, because numpy
    # re-slices a strided view of a 2D array every time.
    tablein_energyryd = np.ascontiguousarray(tablein[:, 0])
    tablein_sigma = np.ascontiguousarray(tablein[:, 1])
    # not an assert: np.searchsorted() and np.interp() below both give a wrong result for a
    # table that decreases in energy. This function also reads the first energy as the threshold.
    if np.any(np.diff(tablein_energyryd) < 0.0):
        msg = (
            f"The energy column of a photoionisation table decreases. The table shape is {tablein.shape}"
            f" and the first energy is {threshold_old_ryd:.6e} Ryd.{labeltext}{keytext}"
        )
        raise ValueError(msg)

    table_energy_last = tablein_energyryd[-1]
    table_sigma_last = tablein_sigma[-1]

    def weighted_averages(arr_energyryd: np.ndarray, arr_sigma_megabarns: np.ndarray) -> np.ndarray:
        """Average the cross section of each row over that row's bin, with the weight above.

        Each row holds the samples of one bin in energy order. The first sample gives nu_low.
        """
        arr_nu = arr_energyryd * ryd_to_hz
        integrand_vals = arr_nu**2 * np.exp(minus_h_over_kb_t * (arr_nu - arr_nu[:, :1]))
        # The two integrals cover one set of x values, so compute the sample widths once.
        arr_dx = np.diff(arr_energyryd, axis=1)
        integralnosigma = trapezoid_with_widths(integrand_vals, arr_dx)
        integralwithsigma = trapezoid_with_widths(arr_sigma_megabarns * integrand_vals, arr_dx)
        # The weight is positive, so integralnosigma is positive. A negative cross section is the
        # only way to get a negative integralwithsigma, and the input must not contain one.
        if np.any(integralwithsigma < 0.0) or np.any(integralnosigma <= 0.0):
            msg = (
                f"A photoionisation bin integral is not positive. The table shape is {tablein.shape},"
                f" the threshold energy is {threshold_old_ryd:.6e} Ryd, the smallest weighted integral is"
                f" {integralwithsigma.min():.6e} and the smallest weight integral is"
                f" {integralnosigma.min():.6e}.{labeltext}{keytext}"
            )
            raise ValueError(msg)
        return integralwithsigma / integralnosigma

    # x is nu/nu_edge. The interval edges depend only on the grid, so compute all of them at once.
    arr_enlow = 0.5 * (xgrid[np.maximum(np.arange(nphixspoints) - 1, 0)] + xgrid[:-1]) * threshold_old_ryd
    arr_enhigh = 0.5 * (xgrid[:-1] + xgrid[1:]) * threshold_old_ryd
    # The table is in energy order, so a bisection finds the samples of each interval. The code
    # does not rebuild a boolean mask over the whole column for each output point.
    arr_startindex = np.searchsorted(tablein_energyryd, arr_enlow, side="left")
    arr_endindex = np.searchsorted(tablein_energyryd, arr_enhigh, side="right")
    arr_nsamples_table = arr_endindex - arr_startindex

    # An interval gets an interpolated point at each edge that its own samples do not reach.
    arr_first = tablein_energyryd[np.minimum(arr_startindex, len(tablein_energyryd) - 1)]
    arr_add_low = (arr_nsamples_table == 0) | (((arr_first - arr_enlow) / arr_enlow) > 1e-20)
    arr_last = np.where(arr_nsamples_table > 0, tablein_energyryd[np.maximum(arr_endindex - 1, 0)], arr_enlow)
    arr_add_high = ((arr_enhigh - arr_last) / arr_last) > 1e-20
    arr_nsamples = arr_nsamples_table + arr_add_low + arr_add_high

    # Three groups of intervals, and the code does each group at once:
    # - past: the whole interval lies above the table, and the two edges are the only samples;
    # - dense: the table gives 50 samples or more, and the code keeps them;
    # - the rest: the code resamples the interval onto 51 points.
    arr_past = arr_enlow > table_energy_last
    arr_dense = (arr_nsamples >= 50) & ~arr_past
    arr_resample = ~(arr_past | arr_dense)

    arr_sigma_out = np.empty(nphixspoints)

    if np.any(arr_past):
        # assume power law decay after the last point
        edges_energyryd = np.stack([arr_enlow[arr_past], arr_enhigh[arr_past]], axis=1)
        edges_sigma = phixs_nu_cubed_tail(table_sigma_last, table_energy_last, edges_energyryd)
        arr_sigma_out[arr_past] = weighted_averages(edges_energyryd, edges_sigma)

    if np.any(arr_resample):
        # 51 points from one bin edge to the other, so the integrals cover the whole bin
        grid_energyryd = np.linspace(arr_enlow[arr_resample], arr_enhigh[arr_resample], num=51, axis=-1)
        # np.interp holds the last cross section constant past the table's end. Apply the same
        # power-law decay that the interval edges use, so a bin that straddles the table end
        # does not overweight its tail.
        # np.asarray() only names the type of the interpolated grid. np.interp() returns that
        # array of float64 already, so the call copies nothing.
        grid_sigma = np.asarray(np.interp(grid_energyryd, tablein_energyryd, tablein_sigma), dtype=np.float64)
        # Almost every table reaches past the highest resampled energy, so the power law applies
        # to no point at all. The grid increases along both axes, so its last value is its
        # largest one. That scalar test keeps the power law off the whole grid in that case.
        if grid_energyryd[-1, -1] > table_energy_last:
            beyond = grid_energyryd > table_energy_last
            grid_sigma[beyond] = phixs_nu_cubed_tail(table_sigma_last, table_energy_last, grid_energyryd[beyond])
        arr_sigma_out[arr_resample] = weighted_averages(grid_energyryd, grid_sigma)

    # Each dense interval keeps its own samples, so the number of samples changes from one
    # interval to the next. Such intervals are rare, and this loop handles them one at a time.
    for i in np.flatnonzero(arr_dense):
        enlow = arr_enlow[i]
        enhigh = arr_enhigh[i]
        sample_energyryd = tablein_energyryd[arr_startindex[i] : arr_endindex[i]]
        sample_sigma = tablein_sigma[arr_startindex[i] : arr_endindex[i]]
        if arr_add_low[i]:
            # np.interp and not scipy: scipy is an optional extra of this package
            sample_energyryd = np.concatenate(([enlow], sample_energyryd))
            sample_sigma = np.concatenate(([np.interp(enlow, tablein_energyryd, tablein_sigma)], sample_sigma))
        if arr_add_high[i]:
            new_crosssection = (
                np.interp(enhigh, tablein_energyryd, tablein_sigma)
                if enhigh <= table_energy_last
                else phixs_nu_cubed_tail(table_sigma_last, table_energy_last, enhigh)
            )
            sample_energyryd = np.concatenate((sample_energyryd, [enhigh]))
            sample_sigma = np.concatenate((sample_sigma, [new_crosssection]))

        arr_sigma_out[i] = weighted_averages(sample_energyryd[np.newaxis, :], sample_sigma[np.newaxis, :])[0]

    return arr_sigma_out
