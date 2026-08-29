"""Downsample photoionization cross-section tables and estimate hydrogenic ones where none exist."""

from functools import partial

import numpy as np
import numpy.typing as npt
import polars as pl

from artisatomic import readfacdata
from artisatomic import readfloers25data
from artisatomic import readhillierdata
from artisatomic import readkuruczdata
from artisatomic import readqubdata
from artisatomic import readtanakajpltdata
from artisatomic.base import elsymbols
from artisatomic.base import hc_in_ev_angstrom
from artisatomic.base import hc_in_ev_cm
from artisatomic.base import parallel_map


def match_hydrogenic_phixs(
    atomic_number: int, energy_levels: pl.DataFrame, ionization_energy_ev: float, ion_handler: str, args
) -> tuple[npt.NDArray[np.float64], list[list[tuple[int, float]]], npt.NDArray[np.float64]]:
    """Estimate photoionization cross sections for a data set that supplies none.

    Applies to any handler, not just one source: a hydrogenic cross section is assigned to each of
    the lowest -nlevels_hydrogenic_for_unknown_phixs levels, scaled to that level's own ionisation
    threshold, with the upper ion's ground state as the only target. That option defaults to 100,
    so this is on unless it is set to 0. It bounds the levels considered rather than the tables
    produced: a level at or above the ionization energy is skipped but still counts towards it.

    The caller only reaches this for an ion whose handler returned no cross sections at all, so
    real data is never replaced or extended by an estimate. The granularity is the whole ion: an
    ion whose handler covered even one level keeps exactly the levels that handler covered, and
    the rest are left without photoionization rather than filled in hydrogenically.
    """
    dict_get_n_func = {
        "tanakajplt": readtanakajpltdata.get_level_valence_n,
        "kurucz": readkuruczdata.get_level_valence_n,
        "fac": readfacdata.get_level_valence_n,
        "floers25calibwithforbidden": readfloers25data.get_level_valence_n,
        "floers25calib": readfloers25data.get_level_valence_n,
        "floers25uncalib": readfloers25data.get_level_valence_n,
        "qub_data": readqubdata.get_level_valence_n,
    }
    if ion_handler not in dict_get_n_func:
        print(
            f"WARNING: Can't assign hydrogenic photoionization cross sections because I don't know how to find principle quantum numbers for {ion_handler} levels"
        )
        return np.empty((0, args.nphixspoints)), [], np.empty(0)

    get_n = dict_get_n_func[ion_handler]
    print(f"using hydrogenic photoionization cross sections for Z={atomic_number} {elsymbols[atomic_number]}")

    photoionization_crosssections = np.zeros((energy_levels.height, args.nphixspoints))
    photoionization_targetfractions: list[list[tuple[int, float]]] = [[] for _ in range(energy_levels.height)]
    photoionization_thresholds_ev = np.full(energy_levels.height, np.nan)
    phixstables = {}
    for levelindex, level in enumerate(energy_levels.iter_rows(named=True)):
        if levelindex >= args.nlevels_hydrogenic_for_unknown_phixs:
            break
        en_ev = hc_in_ev_cm * level["energyabovegsinpercm"]
        threshold_ev = ionization_energy_ev - en_ev
        if threshold_ev <= 0.0:
            # level lies above the ionization energy, so there is nothing to ionize from
            continue
        photoionization_thresholds_ev[levelindex] = threshold_ev
        lambda_angstrom = hc_in_ev_angstrom / threshold_ev

        n = get_n(level["levelname"])
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

    Units don't matter, but the first (lowest) energy point is assumed to be the threshold energy

    The key type is preserved: callers index the tables by level name or by level id.
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


# this method downsamples the photoionization cross section table to a
# regular grid while keeping the recombination rate integral constant
# (assuming that the temperature matches)
def reduce_phixs_tables_worker(
    optimaltemperature: float,
    nphixspoints: int,
    phixsnuincrement: float,
    tablein: np.ndarray,
) -> np.ndarray:
    """Downsample one cross-section table onto the output's nu/nu_edge grid.

    Each output point is the average of the input over that point's frequency bin, weighted by
    nu^2 exp(-h nu / k T) so that the recombination rate at optimaltemperature is preserved
    rather than the cross section itself.
    """
    ryd_to_hz = 3289841960250880.5
    h_over_kb_in_K_sec = 4.799243073366221e-11
    minus_h_over_kb_t = -h_over_kb_in_K_sec / optimaltemperature

    xgrid = np.linspace(1.0, 1.0 + phixsnuincrement * (nphixspoints + 1), num=nphixspoints + 1, endpoint=False)

    # an empty table has no threshold to scale the grid by, and a zero threshold would divide by
    # zero, so both mean "no cross section" rather than an index error on tablein[0]
    if len(tablein) == 0 or tablein[0][0] == 0.0:
        return np.zeros(nphixspoints)

    threshold_old_ryd = tablein[0][0]
    # tablein is an array of pairs (energy, phixs cross section). Split once: the loop below reads
    # both columns per output point, and a strided view of a 2D array is re-sliced every time.
    tablein_energyryd = np.ascontiguousarray(tablein[:, 0])
    tablein_sigma = np.ascontiguousarray(tablein[:, 1])
    table_energy_last = tablein_energyryd[-1]
    table_sigma_last = tablein_sigma[-1]

    arr_sigma_out = np.empty(nphixspoints)
    # x is nu/nu_edge

    # the interval edges depend only on the grid, so compute all of them at once
    arr_enlow = 0.5 * (xgrid[np.maximum(np.arange(nphixspoints) - 1, 0)] + xgrid[:-1]) * threshold_old_ryd
    arr_enhigh = 0.5 * (xgrid[:-1] + xgrid[1:]) * threshold_old_ryd
    # the table is sorted by energy (interp1d was called with assume_sorted), so each interval's
    # slice can be bisected rather than rebuilding a boolean mask over the whole column per point
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
            nsteps = 50  # was 500
            arr_energyryd = np.linspace(enlow, enhigh, num=nsteps, endpoint=False)
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
