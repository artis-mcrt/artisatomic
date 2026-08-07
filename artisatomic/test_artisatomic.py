#!/usr/bin/env python3
"""Tests for the artisatomic readers, parsers and output writers."""

import io
import typing as t

import numpy as np
import polars as pl
import pytest

from artisatomic import add_handler_if_not_set
from artisatomic import hc_in_ev_cm
from artisatomic import interpret_configuration
from artisatomic import leveltuples_to_pldataframe
from artisatomic import match_hydrogenic_phixs
from artisatomic import PYDIR
from artisatomic import readfacdata
from artisatomic import readfloers25data
from artisatomic import readhillierdata
from artisatomic import readhillierdata as rhd
from artisatomic import readkuruczdata
from artisatomic import readqubdata
from artisatomic import readtanakajpltdata
from artisatomic import reduce_phixs_tables_worker
from artisatomic import write_adata
from artisatomic import write_phixs_data


def test_interpret_term():
    """LS terms are read from level names, reporting unknown rather than raising on unreadable ones."""
    assert readhillierdata.get_term_as_tuple("3d5(6S)4s(7S)4d6De") == (6, 2, 0)
    assert readhillierdata.get_term_as_tuple("3d6_3P2e") == (3, 1, 0)

    # names with no L character must report "unknown" rather than raising UnboundLocalError
    for unreadable in ("e2x", "o12", "3d5", "12"):
        assert readhillierdata.get_term_as_tuple(unreadable) == (-1, -1, -1)


def test_get_parity_from_config():
    """Parity is the sum of l over the occupied orbitals, skipping parent terms and merge markers."""
    from artisatomic import get_parity_from_config

    # sum of l over the occupied orbitals, mod 2
    assert get_parity_from_config("3d7") == 0  # 2 * 7 = 14
    assert get_parity_from_config("3d64s2") == 0  # 2 * 6 = 12
    assert get_parity_from_config("5s2.5p5") == 1  # 0 * 2 + 1 * 5 = 5

    # parent terms in parentheses are not occupied orbitals and must be skipped, not parsed
    assert get_parity_from_config("3s23p63d7(4F)") == 0  # 0*2 + 1*6 + 2*7 = 20
    assert get_parity_from_config("3d6(5D)4s_6De") == 0  # 2 * 6 = 12

    # closed shells with two-digit occupations: a truncated '4f1' reading would give the
    # wrong (odd) parity here, since 3*14 is even but 3*1 is odd
    assert get_parity_from_config("4f145d96s2") == 0  # 3*14 + 2*9 + 0 = 60

    # CMFGEN packs a shell's high-l levels into one level whose orbital letter is a merge marker,
    # not a real l ('5z' would be l=22, '13w' l=19). It spans several l of both parities, so only
    # the real orbitals decide the parity.
    assert get_parity_from_config("2s2_2p3(4So)5z_5Z") == 1  # 2s2 + 2p3 = 3, the 5z contributes none
    assert get_parity_from_config("2s2_13w_2W") == 0  # 2s2 = 0, the 13w contributes none


def test_interpret_configuration():
    """Level names split into orbitals and term, including the ambiguous two-digit n and occupation cases."""
    assert interpret_configuration("3d7(4F)6d_5Pbe") == (["3d7", "(4F)", "6d"], 5, 1, 2, -1)
    assert interpret_configuration("3d6(5D)6d4Ge[9/2]") == (["3d6", "(5D)", "6d"], 4, 4, 0, -1)
    assert interpret_configuration("3d6(3G)4s4p_w5Go[4]") == (["3d6", "(3G)", "4s", "4p"], 5, 4, 1, 4)
    assert interpret_configuration("Eqv st (0S ) 0s  a4P") == ([], 4, 1, 0, 1)
    assert interpret_configuration("3d6    (5D ) 4p  z6Do") == (["3d6", "(5D)", "4p"], 6, 2, 1, 1)
    assert interpret_configuration("3d7b2Fe") == (["3d7"], 2, 3, 0, 2)
    assert interpret_configuration("3d6_3P2e") == (["3d6"], 3, 1, 0, -1)

    # a two-digit principal quantum number is read as such when the orbital has no occupation
    # number, where there is nothing else the digits could belong to
    assert interpret_configuration("3d6(5D)10d_5Pe") == (["3d6", "(5D)", "10d"], 5, 1, 0, -1)

    # ...and also with an occupation number, when the digits cannot belong to a preceding
    # orbital (start of string, or right after a parent term)
    assert interpret_configuration("10d1_2De") == (["10d1"], 2, 2, 0, -1)
    assert interpret_configuration("3d6(5D)10d1_5Pe") == (["3d6", "(5D)", "10d1"], 5, 1, 0, -1)

    # a digit followed by two letters keeps the digit with the letters ('4sp' is treated as an
    # orbital-plus-occupation, matching the historical handling of this malformed Hillier name)
    assert interpret_configuration("4sp(3P)_7Po[2]") == (["4sp", "(3P)"], 7, 1, 1, -1)

    # an orbital with a SINGLE-digit occupation is always read with a single-digit n, because
    # '3d14s2' is genuinely ambiguous and the occupation-1 reading is the common one
    assert interpret_configuration("3d14s2_2De") == (["3d1", "4s2"], 2, 2, 0, -1)

    # ...but trailing digits after the orbital letter are the occupation, so closed d and f
    # shells with TWO-digit occupations are unambiguous and must keep both digits
    assert interpret_configuration("3d104s_3De") == (["3d10", "4s"], 3, 2, 0, -1)
    assert interpret_configuration("3d104s2_1Se") == (["3d10", "4s2"], 1, 0, 0, -1)
    assert interpret_configuration("4d105s1_2Se") == (["4d10", "5s1"], 2, 0, 0, -1)
    assert interpret_configuration("4f145d106s2_1Se") == (["4f14", "5d10", "6s2"], 1, 0, 0, -1)


def test_hydrogenic_phixs():
    """Hydrogenic cross sections match reference values for the n=1 and n=5 shells."""
    ryd_to_ev = rhd.ryd_to_ev

    rhd.read_hyd_phixsdata()

    oneryd_lambda_angstrom = rhd.hc_in_ev_angstrom / ryd_to_ev
    expected_n1 = np.array(
        [
            [1.0, 6.30341644],
            [1.1, 4.88284569],
            [1.21, 3.77314939],
            [1.331, 2.90845266],
            [1.4641, 2.23644386],
            [1.61051, 1.71560775],
            [1.771561, 1.31303106],
            [1.9487171, 1.00268611],
            [2.1435888, 0.76405918],
            [2.35794768, 0.58102658],
        ]
    )

    phixstable_nl = rhd.get_hydrogenic_nl_phixstable(oneryd_lambda_angstrom, 1, 0, 0)
    assert np.allclose(expected_n1, phixstable_nl[:10], rtol=1e-3)

    phixstable_n = rhd.get_hydrogenic_n_phixstable(oneryd_lambda_angstrom, 1)
    assert np.allclose(expected_n1, phixstable_n[:10], rtol=1e-3)

    oneryd_lambda_angstrom = rhd.hc_in_ev_angstrom / (5**2 * ryd_to_ev)
    expected_n5 = np.array(
        [
            [2.50000000e01, 5.91880525e-02],
            [2.75000000e01, 4.48991991e-02],
            [3.02500000e01, 3.40407216e-02],
            [3.32750000e01, 2.57948374e-02],
            [3.66024999e01, 1.95370282e-02],
            [4.02627499e01, 1.47907913e-02],
            [4.42890249e01, 1.11930170e-02],
            [4.87179274e01, 8.46718091e-03],
            [5.35897201e01, 6.40292666e-03],
            [5.89486921e01, 4.84036618e-03],
        ]
    )
    phixstable_nl = rhd.get_hydrogenic_nl_phixstable(oneryd_lambda_angstrom, 5, 0, 4)
    assert np.allclose(expected_n5, phixstable_nl[:10], rtol=1e-3)

    phixstable_n = rhd.get_hydrogenic_n_phixstable(oneryd_lambda_angstrom, 5)
    assert np.allclose(expected_n5, phixstable_n[:10], rtol=1e-3)


def test_hydrogenic_nl_phixs_offset_type8():
    """CMFGEN cross-section type 8 (modified hydrogenic split l).

    Pinned against the SUB_PHOT_GEN type-8 branch in CMFGEN's newsubs/sub_phot_gen.f:

        IF(FREQ_VEC(I) .GE. EDGE+CROSS_A(LMIN+3))THEN
          U=FREQ_VEC(I)/(EDGE+CROSS_A(LMIN+3))
          X=LOG10(U) ... interpolate log10(BF_L_CROSS) linearly in X ...
          SUM=SUM/ZION/ZION
          PHOT(I)=PHOT(I) + SUM/((LEND-LST+1)*(LEND+LST+1))
    """
    from artisatomic.base import h_in_ev_seconds

    rhd.read_hyd_phixsdata()

    ryd_to_ev = rhd.ryd_to_ev

    # real Fe II parameters from FE/II/10sep16/phot_op.dat: n=4, l=1, nu_o=0.88936
    threshold_ev, n, l_start, l_end, nu_o, zion = 7.90, 4, 1, 1, 0.88936, 2
    lambda_angstrom = rhd.hc_in_ev_angstrom / threshold_ev
    e_o_ev = nu_o * 1e15 * h_in_ev_seconds

    phixstable = rhd.get_hydrogenic_nl_phixstable(lambda_angstrom, n, l_start, l_end, nu_o=nu_o, zion=zion)

    energy_ev = phixstable[:, 0] * ryd_to_ev
    below_offset_edge = energy_ev < threshold_ev + e_o_ev

    # zero everywhere below the offset edge, including at the true threshold
    assert below_offset_edge[0]
    assert np.all(phixstable[below_offset_edge, 1] == 0.0)
    # and non-zero immediately above it
    assert np.all(phixstable[~below_offset_edge, 1] > 0.0)

    # independent reimplementation of the CMFGEN branch
    grid = rhd.hyd_phixs_energygrid_ryd[n, l_start]
    u_grid = grid / grid[0]  # not in-place: the module-global table must stay untouched
    sigma_table = np.zeros(len(u_grid))
    for l in range(l_start, l_end + 1):
        sigma_table += (2 * l + 1) * rhd.hyd_phixs[n, l]

    for index, en_ev in enumerate(energy_ev):
        if en_ev < threshold_ev + e_o_ev:
            continue
        u = en_ev / (threshold_ev + e_o_ev)
        expected = 10 ** np.interp(np.log10(u), np.log10(u_grid), np.log10(sigma_table))
        expected /= zion**2 * (l_end - l_start + 1) * (l_end + l_start + 1)
        assert np.isclose(phixstable[index, 1], expected, rtol=1e-10)

    # the energy grid must be untouched by the offset: it still starts at the true threshold
    assert np.isclose(phixstable[0, 0], threshold_ev / ryd_to_ev, rtol=1e-10)

    # nu_o=None (type 2) must be unaffected and must not require zion
    type2 = rhd.get_hydrogenic_nl_phixstable(lambda_angstrom, n, l_start, l_end)
    assert type2[0, 1] > 0.0
    assert np.isclose(type2[0, 0], threshold_ev / ryd_to_ev, rtol=1e-10)


def test_hydrogenic_phixs_effective_charge_scaling():
    """A hydrogenic level of charge Z must have sigma_threshold = sigma_th(H, n=1) / Z**2.

    The H (Z=1) cases in test_hydrogenic_phixs() cannot detect a spurious extra factor of
    Z_eff**2, so check the scaling explicitly for Z > 1.
    """
    ryd_to_ev = rhd.ryd_to_ev

    rhd.read_hyd_phixsdata()

    sigma_hydrogen_1s = rhd.get_hydrogenic_n_phixstable(rhd.hc_in_ev_angstrom / ryd_to_ev, 1)[0][1]

    for atomic_number in (1, 2, 3, 6, 26):
        for n in (1, 2, 5):
            # a hydrogenic level of charge Z and principal quantum number n ionizes at Z**2 / n**2 Ryd
            threshold_ev = atomic_number**2 * ryd_to_ev / n**2
            phixstable = rhd.get_hydrogenic_n_phixstable(rhd.hc_in_ev_angstrom / threshold_ev, n)

            # Kramers: sigma_threshold = 7.91 Mb * n / Z**2 * g_bf, and g_bf at threshold
            # depends only on n, so comparing at the same n leaves a ratio of exactly n / Z**2
            same_n_hydrogen = rhd.get_hydrogenic_n_phixstable(rhd.hc_in_ev_angstrom / (ryd_to_ev / n**2), n)
            assert np.isclose(phixstable[0][1], same_n_hydrogen[0][1] / atomic_number**2, rtol=1e-6)

        # the n=1 threshold cross section must fall exactly as 1 / Z**2
        threshold_ev = atomic_number**2 * ryd_to_ev
        phixstable = rhd.get_hydrogenic_n_phixstable(rhd.hc_in_ev_angstrom / threshold_ev, 1)
        assert np.isclose(phixstable[0][1], sigma_hydrogen_1s / atomic_number**2, rtol=1e-6)


def test_match_hydrogenic_phixs_is_not_double_scaled():
    """match_hydrogenic_phixs() must not rescale the table returned by get_hydrogenic_n_phixstable().

    get_hydrogenic_n_phixstable() already contains the effective-charge scaling, so applying
    a second factor of Z_eff**2 would suppress every cross section (a factor of ~20 for a
    typical E_th = 11 eV, n = 5 valence level).
    """
    import argparse

    rhd.read_hyd_phixsdata()

    ryd_to_ev = rhd.ryd_to_ev

    # a single hydrogenic n=1 level of a Z=2 ion: threshold is 4 Ryd, so sigma_th = 6.307 / 4 Mb
    ionization_energy_ev = 4 * ryd_to_ev
    dflevels = pl.DataFrame(
        {
            "levelid": [0],
            "energyabovegsinpercm": [0.0],
            "g": [2.0],
            "levelname": ["s1s  1S,enpercm=0.0,j=0.5"],
        }
    )
    args = argparse.Namespace(
        nphixspoints=100, phixsnuincrement=0.03, optimaltemperature=6000, nlevels_hydrogenic_for_unknown_phixs=100
    )

    crosssections, targetfractions, thresholds = match_hydrogenic_phixs(
        atomic_number=2,
        energy_levels=dflevels,
        ionization_energy_ev=ionization_energy_ev,
        ion_handler="kurucz",
        args=args,
    )

    assert thresholds[0] == ionization_energy_ev
    assert targetfractions[0] == [(0, 1.0)]  # the upper ion's ground state

    expected_threshold_mb = rhd.get_hydrogenic_n_phixstable(rhd.hc_in_ev_angstrom / ionization_energy_ev, 1)[0][1]
    assert abs(expected_threshold_mb - 6.3067 / 4) < 1e-3  # exact hydrogenic value for He II 1s
    # the downsampled first point is a bin average, so allow a few percent
    assert abs(crosssections[0][0] / expected_threshold_mb - 1) < 0.05

    # levels above the ionization energy must be skipped rather than dividing by a negative threshold
    dflevels_unbound = pl.DataFrame(
        {
            "levelid": [0],
            "energyabovegsinpercm": [2 * ionization_energy_ev / hc_in_ev_cm],
            "g": [2.0],
            "levelname": ["s1s  1S,enpercm=0.0,j=0.5"],
        }
    )
    crosssections, targetfractions, thresholds = match_hydrogenic_phixs(
        atomic_number=2,
        energy_levels=dflevels_unbound,
        ionization_energy_ev=ionization_energy_ev,
        ion_handler="kurucz",
        args=args,
    )
    assert np.isnan(thresholds[0])  # NaN is "no threshold energy", which write_phixs_data() skips
    assert targetfractions[0] == []
    assert np.all(crosssections[0] == 0.0)


def test_nlevels_hydrogenic_for_unknown_phixs_caps_the_level_count():
    """-nlevels_hydrogenic_for_unknown_phixs sets how many of the lowest levels get an estimate."""
    import argparse

    rhd.read_hyd_phixsdata()

    ionization_energy_ev = 4 * rhd.ryd_to_ev
    nlevels = 5
    dflevels = pl.DataFrame(
        {
            "levelid": list(range(nlevels)),
            # ascending, and all well below the ionization energy so none is skipped as unbound
            "energyabovegsinpercm": [i * 1000.0 for i in range(nlevels)],
            "g": [2.0] * nlevels,
            "levelname": ["s1s  1S,enpercm=0.0,j=0.5"] * nlevels,
        }
    )

    for nlevels_option, n_expected in ((0, 0), (2, 2), (nlevels + 10, nlevels)):
        args = argparse.Namespace(
            nphixspoints=100,
            phixsnuincrement=0.03,
            optimaltemperature=6000,
            nlevels_hydrogenic_for_unknown_phixs=nlevels_option,
        )
        _, targetfractions, thresholds = match_hydrogenic_phixs(
            atomic_number=2,
            energy_levels=dflevels,
            ionization_energy_ev=ionization_energy_ev,
            ion_handler="kurucz",
            args=args,
        )
        assert sum(1 for targets in targetfractions if targets) == n_expected
        assert np.count_nonzero(~np.isnan(thresholds)) == n_expected

    # the limit bounds the levels considered, not the tables produced: an unbound level inside it
    # is skipped but still counts, so asking for 3 here yields only the one bound level below them
    unbound_percm = 2 * ionization_energy_ev / hc_in_ev_cm
    dflevels_partly_unbound = dflevels.with_columns(
        energyabovegsinpercm=pl.Series([0.0, unbound_percm, unbound_percm, 1000.0, 2000.0])
    )
    args = argparse.Namespace(
        nphixspoints=100, phixsnuincrement=0.03, optimaltemperature=6000, nlevels_hydrogenic_for_unknown_phixs=3
    )
    _, targetfractions, thresholds = match_hydrogenic_phixs(
        atomic_number=2,
        energy_levels=dflevels_partly_unbound,
        ionization_energy_ev=ionization_energy_ev,
        ion_handler="kurucz",
        args=args,
    )
    assert np.count_nonzero(~np.isnan(thresholds)) == 1


def test_write_phixs_data_with_no_phixs_arrays():
    """A reader that found no photoionization data must not make write_phixs_data() index off the end.

    resolve_photoion_targetfractions() fills a target list for every level whenever the reader supplied none, and
    readhillierdata.get_photoiontargetfractions() always gives at least the ground state. If the
    reader also left the cross-section and threshold arrays empty, the level ids from those target
    lists have nothing behind them.
    """
    import argparse

    args = argparse.Namespace(nphixspoints=100, phixsnuincrement=0.03, optimaltemperature=6000)
    flog = io.StringIO()
    fphixs = io.StringIO()

    write_phixs_data(
        fphixs,
        atomic_number=26,
        ion_stage=1,
        photoionization_crosssections=np.empty((0, args.nphixspoints)),
        photoionization_targetfractions=[[(0, 1.0)] for _ in range(3)],
        photoionization_thresholds_ev=np.empty(0),
        args=args,
        flog=flog,
    )

    assert not fphixs.getvalue()
    assert "Writing 0 phixs tables" in flog.getvalue()


def make_iondata(ion_stage, is_top_ion, targetfractions=None, targetconfigs=None):
    """Build a minimal single-level IonData for the target-fraction resolver tests."""
    from artisatomic.iondata import IonData

    return IonData(
        ion_stage=ion_stage,
        handler="cmfgen",
        is_top_ion=is_top_ion,
        ionization_energy_ev=10.0,
        dfenergylevels=pl.DataFrame(
            {
                "levelid": [0],
                "energyabovegsinpercm": [0.0],
                "g": [9.0],
                "levelname": [f"gs{ion_stage}"],
            }
        ),
        dftransitions=pl.DataFrame(),
        transition_count_of_level_name={},
        upsilondict={},
        hillier_photoion_targetconfigs=targetconfigs,
        photoionization_crosssections=np.empty((0, 100)),
        photoionization_targetfractions=targetfractions if targetfractions is not None else [],
        photoionization_thresholds_ev=np.empty(0),
    )


def test_resolve_photoion_targetfractions():
    """Each non-top ion gets its targets resolved against the next ion up; the top ion gets none."""
    from artisatomic.iondata import resolve_photoion_targetfractions

    # the lower ion names the upper ion's ground state as its only target configuration
    lower = make_iondata(1, is_top_ion=False, targetconfigs=[[("gs2", 1.0)]])
    upper = make_iondata(2, is_top_ion=True)
    resolve_photoion_targetfractions([lower, upper])

    assert lower.photoionization_targetfractions == [[(0, 1.0)]]
    # the top ion has no upper ion to photoionise to, so it is left exactly as it was read
    assert upper.photoionization_targetfractions == []


def test_resolve_photoion_targetfractions_keeps_reader_supplied():
    """An ion whose reader already gave per-level fractions (e.g. the hydrogenic estimate) keeps them."""
    from artisatomic.iondata import resolve_photoion_targetfractions

    # a target list the Hillier resolver would never produce, so an overwrite would be visible
    supplied = [[(7, 1.0)]]
    lower = make_iondata(1, is_top_ion=False, targetfractions=supplied, targetconfigs=[[("gs2", 1.0)]])
    resolve_photoion_targetfractions([lower, make_iondata(2, is_top_ion=True)])

    assert lower.photoionization_targetfractions == supplied


def test_resolve_photoion_targetfractions_rejects_misordered_ions():
    """A list that is not one element's ions in ascending order is rejected rather than resolved.

    Each ion is resolved against the next entry as its upper ion, so a top ion anywhere but last
    means the levels being matched belong to the wrong ion.
    """
    from artisatomic.iondata import resolve_photoion_targetfractions

    with pytest.raises(ValueError, match="ascending ion stage order"):
        resolve_photoion_targetfractions([make_iondata(1, is_top_ion=True), make_iondata(2, is_top_ion=True)])

    # a list whose last ion is not the top ion is missing the upper ion the last entry needs
    with pytest.raises(ValueError, match="ascending ion stage order"):
        resolve_photoion_targetfractions([make_iondata(1, is_top_ion=False), make_iondata(2, is_top_ion=False)])


def test_write_output_files_rejects_unresolved_targetfractions(tmp_path):
    """Writing an ion that still needs resolving must fail loudly, not drop its cross sections.

    write_output_files() no longer resolves target fractions itself, so an ion with cross sections
    but no targets would have every one of its tables silently skipped by write_phixs_data().
    """
    import argparse

    from artisatomic.output import write_output_files

    (tmp_path / "logs").mkdir()
    tmpargs = argparse.Namespace(
        output_folder=str(
            tmp_path,
        ),
        output_folder_logs="logs",
        nophixs=False,
        nphixspoints=100,
    )
    lower = make_iondata(1, is_top_ion=False)
    lower.photoionization_crosssections = np.zeros((1, 100))

    with pytest.raises(ValueError, match="call resolve_photoion_targetfractions"):
        write_output_files(26, [lower, make_iondata(2, is_top_ion=True)], tmpargs)


def test_read_coldata_term_to_j_redistribution():
    """A term-resolved effective collision strength must be shared over the J levels of BOTH terms.

    ARTIS forms the collisional excitation rate coefficient as proportional to upsilon_ij / g_i,
    so the invariant that makes the total term-to-term rate correct is

        sum_i sum_j upsilon_ij == upsilon_term,   upsilon_ij = upsilon_term * g_i/g_L * g_j/g_U

    O III has term-resolved collision data (col_data_oiii_butler_2012.dat) and a J-split level
    list, so it exercises the redistribution; Fe II names its collision transitions with J
    values, so its values must pass through untouched.
    """
    import argparse
    import contextlib
    from collections import defaultdict

    args = argparse.Namespace(electrontemperature=5000)

    def read_ion(atomic_number, ion_stage):
        flog = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()):
            _, dflevels, _, _ = readhillierdata.read_levels_and_transitions(atomic_number, ion_stage, flog)
            upsilondict = readhillierdata.read_coldata(atomic_number, ion_stage, dflevels, flog, args)
        levelids_of_term = defaultdict(list)
        for levelid, levelname in enumerate(dflevels["levelname"]):
            levelids_of_term[levelname.split("[")[0]].append(levelid)
        return dflevels["g"].to_list(), upsilondict, levelids_of_term

    gvalues, upsilondict, levelids_of_term = read_ion(8, 3)

    lower_ids = levelids_of_term["2s2_2p2_3Pe"]  # J = 0, 1, 2 with g = 1, 3, 5
    upper_ids = levelids_of_term["2s_2p3_3Do"]
    assert [gvalues[i] for i in lower_ids] == [1.0, 3.0, 5.0]

    sums_from_lower = [
        sum(upsilondict[i, j] for j in upper_ids if upsilondict.get((i, j), -1.0) > 0.0) for i in lower_ids
    ]

    # the single value in the collision data file for this term pair
    upsilon_term = 5.791
    assert abs(sum(sums_from_lower) - upsilon_term) < 1e-3

    # and it is split over the lower levels in proportion to g_i (1 : 3 : 5 out of g_L = 9)
    for g_lower, total in zip([1.0, 3.0, 5.0], sums_from_lower, strict=True):
        assert abs(total - upsilon_term * g_lower / 9.0) < 1e-3

    # Fe II collision data is already J-resolved, so every value passes through unscaled
    _, upsilondict_fe2, _ = read_ion(26, 2)
    assert sum(1 for v in upsilondict_fe2.values() if v > 0.0) == 10601


def test_add_handler_if_not_set():
    """Adding a handler returns a new list and never overrides an ion that is already present."""
    ion_handlers: list[tuple[int, list[tuple[int, str]]]] = [(26, [(1, "cmfgen"), (2, "cmfgen")])]
    unchanged = [(26, [(1, "cmfgen"), (2, "cmfgen")])]

    # adding an ion for a new element must not modify the input list
    result = add_handler_if_not_set(ion_handlers, 58, 1, "dream")
    assert ion_handlers == unchanged
    assert result == [(26, [(1, "cmfgen"), (2, "cmfgen")]), (58, [(1, "dream")])]

    # add an ion to an existing element
    result = add_handler_if_not_set(ion_handlers, 26, 3, "dream")
    assert ion_handlers == unchanged
    assert result == [(26, [(1, "cmfgen"), (2, "cmfgen"), (3, "dream")])]

    # an already-present ion stage keeps the handler it was given, whatever the new one says
    result = add_handler_if_not_set(ion_handlers, 26, 2, "dream")
    assert result == unchanged

    # ion stages can be given as tuples or lists (e.g. straight from json.load())
    ion_handlers_json = t.cast("list[tuple[int, list[tuple[int, str]]]]", [(26, [[1, "cmfgen"]])])
    result = add_handler_if_not_set(ion_handlers_json, 26, 1, "dream")
    assert result == [(26, [[1, "cmfgen"]])]


def test_parse_ion_handlers():
    """The JSON form becomes tuples, and an ion that names no handler is rejected."""
    from artisatomic import parse_ion_handlers

    # json.load() gives nested lists; every ion must come back as an (ion_stage, handler) tuple
    assert parse_ion_handlers([[26, [[1, "cmfgen"], [2, "cmfgen"]]]]) == [(26, [(1, "cmfgen"), (2, "cmfgen")])]

    # a bare ion stage is a leftover from when the handler was optional. It must be named as such
    # here, rather than failing later where neither the element nor the file would be mentioned.
    with pytest.raises(TypeError, match=r"Z=26 ion stage 2 .* names no handler"):
        parse_ion_handlers([[26, [[1, "cmfgen"], 2]]])


def test_split_element_ionstage_str():
    """'FeII' splits into element and ion stage, including the symbols made only of Roman numeral letters."""
    from artisatomic import split_element_ionstage_str

    assert split_element_ionstage_str("FeII") == (26, 2)
    assert split_element_ionstage_str("DyIII") == (66, 3)
    assert split_element_ionstage_str("SiI") == (14, 1)
    assert split_element_ionstage_str("HI") == (1, 1)

    # rstrip("IVX") would leave nothing behind for the elements whose symbols are made of
    # those letters, so these are the cases that used to raise ValueError
    assert split_element_ionstage_str("VI") == (23, 1)  # vanadium I, not "V" as a numeral
    assert split_element_ionstage_str("VIII") == (23, 3)  # vanadium III
    assert split_element_ionstage_str("IV") == (53, 5)  # iodine V
    assert split_element_ionstage_str("II") == (53, 1)  # iodine I
    assert split_element_ionstage_str("XeIV") == (54, 4)

    with pytest.raises(ValueError, match="Could not split"):
        split_element_ionstage_str("NotAnIon")


def test_hillier_extend_ion_list():
    """The CMFGEN ion list honours the maximum ion stage and the hydrogen exclusion."""
    result = readhillierdata.extend_ion_list([], maxionstage=1, include_hydrogen=False)
    assert (2, [(1, "cmfgen")]) in result
    assert (26, [(1, "cmfgen")]) in result
    assert all(atomic_number != 1 for atomic_number, _ in result)
    assert all(entry == (1, "cmfgen") for _, listions in result for entry in listions)


def test_reduce_phixs_tables_worker():
    """Downsampling a cross-section table preserves the recombination rate it was weighted for."""
    nphixspoints = 100
    phixsnuincrement = 0.03
    temperature = 6000.0
    sigma_0 = 4.0

    # dense input table with a hydrogenic-like sigma_0 * (E_threshold / E)**3 cross section
    energyryd = np.linspace(1.0, 1.0 + phixsnuincrement * (nphixspoints + 1) * 2, 5000)
    tablein = np.column_stack([energyryd, sigma_0 * (energyryd[0] / energyryd) ** 3])

    reduced = reduce_phixs_tables_worker(temperature, nphixspoints, phixsnuincrement, tablein)
    assert len(reduced) == nphixspoints
    assert abs(reduced[0] / sigma_0 - 1) < 0.05  # first point close to the threshold cross section
    assert np.all(np.diff(reduced) < 0)  # monotonically decreasing

    # a constant cross section must be preserved exactly
    tablein_const = np.column_stack([energyryd, np.full_like(energyryd, 2.5)])
    reduced_const = reduce_phixs_tables_worker(temperature, nphixspoints, phixsnuincrement, tablein_const)
    assert np.allclose(reduced_const, 2.5, rtol=1e-6)

    # the downsampling must preserve the recombination-rate integral
    # sigma(nu) * nu**2 * exp(-h*nu / (k_B * T)) at the optimisation temperature
    ryd_to_hz = 3289841960250880.5
    h_over_kb_in_k_sec = 4.799243073366221e-11

    def recomb_integral(en_ryd, sigmas):
        nu = np.asarray(en_ryd) * ryd_to_hz
        return np.trapezoid(sigmas * nu**2 * np.exp(-h_over_kb_in_k_sec * nu / temperature), nu)

    # reconstruct the reduced table as piecewise-constant over the output grid intervals
    xgrid = np.linspace(1.0, 1.0 + phixsnuincrement * (nphixspoints + 1), num=nphixspoints + 1, endpoint=False)
    interval_edges = [xgrid[0], *(0.5 * (xgrid[i] + xgrid[i + 1]) for i in range(nphixspoints))]
    dense_en: list[float] = []
    dense_sigma: list[float] = []
    for i in range(nphixspoints):
        segment = np.linspace(interval_edges[i], interval_edges[i + 1], 200)
        dense_en.extend(segment)
        dense_sigma.extend([reduced[i]] * len(segment))

    integral_reduced = recomb_integral(dense_en, np.array(dense_sigma))
    selection = (energyryd >= interval_edges[0]) & (energyryd <= interval_edges[-1])
    integral_input = recomb_integral(energyryd[selection], tablein[selection, 1])
    assert abs(integral_reduced / integral_input - 1) < 0.01


def test_read_adf04():
    """An adf04 file yields levels and effective collision strengths keyed by zero-based level ids."""
    flog = io.StringIO()
    ionization_energy_ev, energylevels, upsilondict = readqubdata.read_adf04(
        (PYDIR / ".." / "atomic-data-qub" / "co_tyndall_test_sample" / "adf04_v1").resolve(), 27, 3, flog
    )
    assert abs(ionization_energy_ev - 40.964007) < 1e-5
    assert len(energylevels) == 262
    assert len(upsilondict) == 235
    level1 = energylevels[0]
    assert level1 is not None
    assert level1.levelname == "3s23p63d7(4F)_4Fe[9/2]_id=1"
    assert level1.energyabovegsinpercm == 0.0
    assert level1.g == 10.0
    assert level1.parity == 0


def test_write_adata_level_comment():
    """The level comment is the level's name, with no padding.

    artistools reads the comment as `line.split(maxsplit=4)[4].strip("'")`, which strips quotes but
    not whitespace, so any padding written here ends up inside the level name it reports. The
    Hillier display replacements are applied here rather than in the reader, where the name is the
    key that transitions are matched on.
    """
    dfhillier = leveltuples_to_pldataframe(
        pl.DataFrame(
            {
                "levelname": ["someion_gs"],
                "g": [9.0],
                "energyabovegsinpercm": [0.0],
                "lambdaangstrom": [911.0],
                "hillierlevelid": [1],
                "parity": [0],
            }
        )
    )
    buf = io.StringIO()
    write_adata(buf, 26, 2, dfhillier, 10.0, {}, io.StringIO())
    hillier_line = buf.getvalue().splitlines()[1]
    assert hillier_line.endswith(" someion_gs")
    assert hillier_line.split(maxsplit=4)[4] == "someion_gs"

    # a level name containing spaces is written unchanged, and reads back as everything after
    # the fourth field
    spacedlevelname = "3Pe index 1 '2s22p2'"
    dfspaced = leveltuples_to_pldataframe(
        pl.DataFrame({"levelname": [spacedlevelname], "energyabovegsinpercm": [0.0], "g": [9.0]})
    )
    buf = io.StringIO()
    write_adata(buf, 8, 1, dfspaced, 13.6, {}, io.StringIO())
    spaced_line = buf.getvalue().splitlines()[1]
    assert spaced_line.endswith(" " + spacedlevelname)
    assert spaced_line.split(maxsplit=4)[4] == spacedlevelname


def test_get_level_valence_n():
    """Each reader's level names yield the valence electron's principal quantum number."""
    # each handler has its own level-name format and parser
    assert readkuruczdata.get_level_valence_n("s5p  3P,enpercm=14276.381,j=0.0") == 5
    assert readtanakajpltdata.get_level_valence_n("2,even,{  4d- 3  4d+ 1  5s+ 1 }") == 5
    assert readfloers25data.get_level_valence_n("4f10") == 4
    assert readfloers25data.get_level_valence_n("4f9.6s") == 6
    assert readfloers25data.get_level_valence_n("5s2.5p5") == 5
    assert readfacdata.get_level_valence_n("4f9 6s1") == 6
    assert readfacdata.get_level_valence_n("4f10") == 4

    # the floers25 and fac level names carry a uniquifying suffix, which must be ignored
    assert readfloers25data.get_level_valence_n("4f10 J=8 index=0") == 4
    assert readfloers25data.get_level_valence_n("4f9.6s J=15/2 index=137") == 6
    assert readfloers25data.get_level_valence_n("5s2.5p5 J=3/2 index=2") == 5
    assert readfacdata.get_level_valence_n("4f9 6s1 Ilev=42") == 6
    assert readfacdata.get_level_valence_n("4f10 Ilev=0") == 4
    assert readqubdata.get_level_valence_n("3d7_4Fe[9/2]_id=1") == 3
    assert readqubdata.get_level_valence_n("5s2_1Se[0/2]_id=1") == 5
