"""Downsample photoionisation cross section tables and estimate hydrogenic ones where none exist."""

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
from artisatomic.base import parallel_map
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
    ion log records it. The hydrogenic tables cover n = 1 to max_hyd_gaunt_n only. A level
    outside that range also gets no estimate, and the function does not read past the table.
    """
    # stdout only, as before: the tested log files must not change for an ion that gets the
    # same estimate as before. A skipped level below is new information, so that goes to the log.
    if get_level_valence_n is None:
        print(
            f"WARNING: no hydrogenic photoionization cross sections, because no parser gives the principal"
            f" quantum number of a {ion_handler} level"
        )
        return np.empty((0, args.nphixspoints)), [], np.empty(0)

    print(f"using hydrogenic photoionization cross sections for Z={atomic_number} {elsymbols[atomic_number]}")
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
                f"WARNING: no principal quantum number found in level name '{level['levelname']}', so the level"
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
        phixstables[levelindex] = readhillierdata.get_hydrogenic_n_phixstable(lambda_angstrom=lambda_angstrom, n=n)
        photoionization_targetfractions[levelindex] = [(0, 1.0)]  # the upper ion's ground state

    reduced_phixs_dict = reduce_phixs_tables(
        phixstables, args.optimaltemperature, args.nphixspoints, args.phixsnuincrement
    )
    for levelindex, reduced_phixs_table in reduced_phixs_dict.items():
        photoionization_crosssections[levelindex] = reduced_phixs_table

    return photoionization_crosssections, photoionization_targetfractions, photoionization_thresholds_ev


def reduce_phixs_tables[KeyType](
    dicttables: dict[KeyType, npt.NDArray[np.float64]],
    optimaltemperature: float,
    nphixspoints: int,
    phixsnuincrement: float,
) -> dict[KeyType, npt.NDArray[np.float64]]:
    """Downsample each 2D table of (energy, cross section) points into a 1D array.

    The energy unit does not matter. The function reads the first (lowest) energy point as the
    threshold energy.

    The result keeps the key type: callers index the tables by level name or by level id.
    """
    print(f"Processing {len(dicttables.keys()):d} phixs tables")

    return dict(
        zip(
            dicttables.keys(),
            parallel_map(
                partial(
                    reduce_phixs_tables_worker,
                    optimaltemperature,
                    nphixspoints,
                    phixsnuincrement,
                ),
                dicttables.values(),
            ),
            strict=True,
        )
    )


# This function downsamples the photoionisation cross section table to a regular grid. It keeps
# the recombination rate integral constant if the temperature matches.
def reduce_phixs_tables_worker(
    optimaltemperature: float,
    nphixspoints: int,
    phixsnuincrement: float,
    tablein: np.ndarray,
) -> np.ndarray:
    """Downsample one cross section table onto the output's nu/nu_edge grid.

    Each output point is the average of the input over that point's frequency bin, with the
    weight nu^2 exp(-h nu / k T). The weight preserves the recombination rate at
    optimaltemperature, and not the cross section itself.
    """
    minus_h_over_kb_t = -h_over_kb_in_K_sec / optimaltemperature

    xgrid = np.linspace(1.0, 1.0 + phixsnuincrement * (nphixspoints + 1), num=nphixspoints + 1, endpoint=False)

    # An empty table has no threshold to scale the grid, and a zero threshold would divide by
    # zero. Both therefore mean "no cross section", and neither raises an index error on tablein[0].
    if len(tablein) == 0 or tablein[0][0] == 0.0:
        return np.zeros(nphixspoints)

    threshold_old_ryd = tablein[0][0]
    # tablein is an array of pairs (energy, phixs cross section). Split it once. The loop below
    # reads both columns for each output point, and numpy re-slices a strided view of a 2D array
    # every time.
    tablein_energyryd = np.ascontiguousarray(tablein[:, 0])
    tablein_sigma = np.ascontiguousarray(tablein[:, 1])
    table_energy_last = tablein_energyryd[-1]
    table_sigma_last = tablein_sigma[-1]

    arr_sigma_out = np.empty(nphixspoints)
    # x is nu/nu_edge

    # the interval edges depend only on the grid, so compute all of them at once
    arr_enlow = 0.5 * (xgrid[np.maximum(np.arange(nphixspoints) - 1, 0)] + xgrid[:-1]) * threshold_old_ryd
    arr_enhigh = 0.5 * (xgrid[:-1] + xgrid[1:]) * threshold_old_ryd
    # The table is in energy order (the old code called interp1d with assume_sorted). A bisection
    # therefore finds each interval's slice, and the loop does not rebuild a boolean mask over the
    # whole column for each point.
    arr_startindex = np.searchsorted(tablein_energyryd, arr_enlow, side="left")
    arr_endindex = np.searchsorted(tablein_energyryd, arr_enhigh, side="right")

    for i in range(nphixspoints):
        enlow = arr_enlow[i]
        enhigh = arr_enhigh[i]

        # start of interval interpolated point, input data points, and end of interval interpolated point
        sample_energyryd = tablein_energyryd[arr_startindex[i] : arr_endindex[i]]
        sample_sigma = tablein_sigma[arr_startindex[i] : arr_endindex[i]]

        if len(sample_energyryd) == 0 or ((sample_energyryd[0] - enlow) / enlow) > 1e-20:
            if i == 0 and len(sample_energyryd) != 0:
                print(
                    f"adding first point {enlow:.4e} {sample_energyryd[0]:.4e} {(sample_energyryd[0] - enlow) / enlow:.4e}"
                )
            if enlow <= table_energy_last:
                # np.interp, not scipy's interp1d: identical linear interpolation (verified
                # bit-for-bit) without a scipy call per interval edge
                new_crosssection = np.interp(enlow, tablein_energyryd, tablein_sigma)
                if new_crosssection < 0:
                    print("negative extrap")
            else:
                # assume power law decay after last point
                new_crosssection = table_sigma_last * (table_energy_last / enlow) ** 3
            sample_energyryd = np.concatenate(([enlow], sample_energyryd))
            sample_sigma = np.concatenate(([new_crosssection], sample_sigma))

        if ((enhigh - sample_energyryd[-1]) / sample_energyryd[-1]) > 1e-20:
            if enhigh <= table_energy_last:
                new_crosssection = np.interp(enhigh, tablein_energyryd, tablein_sigma)
                if new_crosssection < 0:
                    print("negative extrap")
            else:
                new_crosssection = (
                    table_sigma_last * (table_energy_last / enhigh) ** 3
                )  # assume power law decay after last point

            sample_energyryd = np.concatenate((sample_energyryd, [enhigh]))
            sample_sigma = np.concatenate((sample_sigma, [new_crosssection]))

        nsamples = len(sample_energyryd)

        if nsamples >= 50 or enlow > table_energy_last:
            arr_energyryd = sample_energyryd
            arr_sigma_megabarns = sample_sigma
        else:
            # 51 points from one bin edge to the other, so the integrals below cover the whole
            # bin. With endpoint=False the last two percent of every resampled bin were missing.
            arr_energyryd = np.linspace(enlow, enhigh, num=51)
            # np.interp holds the last cross section constant past the table's end. Apply the
            # same power-law decay that the interval edges above use, so a bin that straddles
            # the table end does not overweight its tail.
            arr_sigma_megabarns = np.where(
                arr_energyryd > table_energy_last,
                table_sigma_last * (table_energy_last / arr_energyryd) ** 3,
                np.interp(arr_energyryd, tablein_energyryd, tablein_sigma),
            )

        # the recombination-rate weight nu^2 exp(-h nu / k T), evaluated over the whole interval at
        # once. np.vectorize() called math.exp() once per sample, which dominated this function.
        arr_nu = arr_energyryd * ryd_to_hz
        integrand_vals = arr_nu**2 * np.exp(minus_h_over_kb_t * arr_nu)
        if np.any(integrand_vals):
            sigma_integrand_vals = arr_sigma_megabarns * integrand_vals

            integralnosigma = np.trapezoid(integrand_vals, arr_energyryd)
            integralwithsigma = np.trapezoid(sigma_integrand_vals, arr_energyryd)

        else:
            integralnosigma = 1.0
            integralwithsigma = np.average(arr_sigma_megabarns)

        if integralwithsigma > 0 and integralnosigma > 0:
            arr_sigma_out[i] = integralwithsigma / integralnosigma
        elif integralwithsigma == 0:
            arr_sigma_out[i] = 0.0
        else:
            print("Math error: ", i, nsamples, integralwithsigma, integralnosigma)
            print(np.column_stack([sample_energyryd, sample_sigma]))
            arr_sigma_out[i] = 0.0

    return arr_sigma_out
