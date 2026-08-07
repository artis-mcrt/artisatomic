"""Downsample photoionization cross-section tables and estimate hydrogenic ones where none exist."""

import math
from functools import partial

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy import integrate  # pyright: ignore[reportMissingTypeStubs]
from scipy import interpolate  # pyright: ignore[reportMissingTypeStubs]

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

    dictout = dict(
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

    return dictout


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

    # proportional to recombination rate
    # nu0 = 1e16
    # fac = math.exp(h_over_kb_in_K_sec * nu0 / optimaltemperature)

    def integrand(nu):
        """Weight for averaging the cross section: proportional to the recombination rate."""
        return (nu**2) * math.exp(-h_over_kb_in_K_sec * nu / optimaltemperature)

    # def integrand_vec(nu_list):
    #    return [(nu ** 2) * math.exp(- h_over_kb_in_K_sec * (nu - nu0) / optimaltemperature)
    #            for nu in nu_list]

    integrand_vec = np.vectorize(integrand)

    xgrid = np.linspace(1.0, 1.0 + phixsnuincrement * (nphixspoints + 1), num=nphixspoints + 1, endpoint=False)

    # for key in keylist:
    #   tablein = dicttables[key]
    # # filter zero points out of the table
    # firstnonzeroindex = 0
    # for i, point in enumerate(tablein):
    #     if point[1] != 0.:
    #         firstnonzeroindex = i
    #         break
    # if firstnonzeroindex != 0:
    #     tablein = tablein[firstnonzeroindex:]

    # table says zero threshold, so avoid divide by zero
    if tablein[0][0] == 0.0:
        return np.zeros(nphixspoints)

    threshold_old_ryd = tablein[0][0]
    # tablein is an array of pairs (energy, phixs cross section)

    # nu0 = tablein[0][0] * ryd_to_hz

    arr_sigma_out = np.empty(nphixspoints)
    # x is nu/nu_edge

    sigma_interp = interpolate.interp1d(tablein[:, 0], tablein[:, 1], kind="linear", assume_sorted=True)

    for i, _ in enumerate(xgrid[:-1]):
        iprevious = max(i - 1, 0)
        enlow = 0.5 * (xgrid[iprevious] + xgrid[i]) * threshold_old_ryd
        enhigh = 0.5 * (xgrid[i] + xgrid[i + 1]) * threshold_old_ryd

        # start of interval interpolated point, input data points, and end of interval interpolated point
        samples_in_interval = tablein[(enlow <= tablein[:, 0]) & (tablein[:, 0] <= enhigh)]

        if len(samples_in_interval) == 0 or ((samples_in_interval[0, 0] - enlow) / enlow) > 1e-20:
            if i == 0 and len(samples_in_interval) != 0:
                print(
                    f"adding first point {enlow:.4e} {samples_in_interval[0, 0]:.4e} {(samples_in_interval[0, 0] - enlow) / enlow:.4e}"
                )
            if enlow <= tablein[-1][0]:
                new_crosssection = sigma_interp(enlow)
                if new_crosssection < 0:
                    print("negative extrap")
            else:
                # assume power law decay after last point
                new_crosssection = tablein[-1][1] * (tablein[-1][0] / enlow) ** 3
            samples_in_interval = np.vstack([[enlow, new_crosssection], samples_in_interval])

        if (
            len(samples_in_interval) == 0
            or ((enhigh - samples_in_interval[-1, 0]) / samples_in_interval[-1, 0]) > 1e-20
        ):
            if enhigh <= tablein[-1][0]:
                new_crosssection = sigma_interp(enhigh)
                if new_crosssection < 0:
                    print("negative extrap")
            else:
                new_crosssection = (
                    tablein[-1][1] * (tablein[-1][0] / enhigh) ** 3
                )  # assume power law decay after last point

            samples_in_interval = np.vstack([samples_in_interval, [enhigh, new_crosssection]])

        nsamples = len(samples_in_interval)

        # integralnosigma, err = integrate.fixed_quad(integrand_vec, enlow, enhigh, n=250)
        # integralwithsigma, err = integrate.fixed_quad(
        #    lambda x: sigma_interp(x) * integrand_vec(x), enlow, enhigh, n=250)

        # this is incredibly fast, but maybe not accurate
        # integralnosigma, err = integrate.quad(integrand, enlow, enhigh, epsrel=1e-2)
        # integralwithsigma, err = integrate.quad(
        #    lambda x: sigma_interp(x) * integrand(x), enlow, enhigh, epsrel=1e-2)

        if nsamples >= 50 or enlow > tablein[-1][0]:
            arr_energyryd = samples_in_interval[:, 0]
            arr_sigma_megabarns = samples_in_interval[:, 1]
        else:
            nsteps = 50  # was 500
            arr_energyryd = np.linspace(enlow, enhigh, num=nsteps, endpoint=False)
            arr_sigma_megabarns = np.interp(arr_energyryd, tablein[:, 0], tablein[:, 1])
        assert isinstance(arr_sigma_megabarns, np.ndarray)

        integrand_vals = integrand_vec(arr_energyryd * ryd_to_hz)
        if np.any(integrand_vals):
            sigma_integrand_vals = [
                sigma * integrand_val for sigma, integrand_val in zip(arr_sigma_megabarns, integrand_vals, strict=True)
            ]

            integralnosigma = integrate.trapezoid(integrand_vals, arr_energyryd)
            integralwithsigma = integrate.trapezoid(sigma_integrand_vals, arr_energyryd)

        else:
            integralnosigma = 1.0
            integralwithsigma = np.average(arr_sigma_megabarns)

        if integralwithsigma > 0 and integralnosigma > 0:
            arr_sigma_out[i] = integralwithsigma / integralnosigma
        elif integralwithsigma == 0:
            arr_sigma_out[i] = 0.0
        else:
            print("Math error: ", i, nsamples, integralwithsigma, integralnosigma)
            print(samples_in_interval)
            arr_sigma_out[i] = 0.0
            # sys.exit()

    return arr_sigma_out
