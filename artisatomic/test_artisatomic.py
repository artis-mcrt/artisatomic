#!/usr/bin/env python3
import io
import typing as t

import numpy as np
import polars as pl

from artisatomic import add_handler_if_not_set
from artisatomic import get_default_handler
from artisatomic import get_term_as_tuple
from artisatomic import interpret_configuration
from artisatomic import interpret_parent_term
from artisatomic import PYDIR
from artisatomic import readfacdata
from artisatomic import readfloers25data
from artisatomic import readhillierdata
from artisatomic import readkuruczdata
from artisatomic import readnahardata
from artisatomic import readqubdata
from artisatomic import readtanakajpltdata
from artisatomic import reduce_configuration
from artisatomic import reduce_phixs_tables_worker
from artisatomic import score_config_match


def test_reduce_configuration():
    assert reduce_configuration("3d64s  (6D ) 8p  j5Fo") == "3d64s8p_5Fo"
    assert reduce_configuration("3d6_3P2e") == "3d6_3Pe"


def test_interpret_term():
    assert get_term_as_tuple("3d5(6S)4s(7S)4d6De") == (6, 2, 0)
    assert get_term_as_tuple("3d6_3P2e") == (3, 1, 0)


def test_interpret_parent_term():
    assert interpret_parent_term("(3P2)") == (3, 1, 2)
    assert interpret_parent_term("(b2D)") == (2, 2, -1)


def test_interpret_configuration():
    assert interpret_configuration("3d7(4F)6d_5Pbe") == (["3d7", "(4F)", "6d"], 5, 1, 2, -1)
    assert interpret_configuration("3d6(5D)6d4Ge[9/2]") == (["3d6", "(5D)", "6d"], 4, 4, 0, -1)
    assert interpret_configuration("3d6(3G)4s4p_w5Go[4]") == (["3d6", "(3G)", "4s", "4p"], 5, 4, 1, 4)
    assert interpret_configuration("Eqv st (0S ) 0s  a4P") == ([], 4, 1, 0, 1)
    assert interpret_configuration("3d6    (5D ) 4p  z6Do") == (["3d6", "(5D)", "4p"], 6, 2, 1, 1)
    assert interpret_configuration("3d7b2Fe") == (["3d7"], 2, 3, 0, 2)
    assert interpret_configuration("3d6_3P2e") == (["3d6"], 3, 1, 0, -1)


def test_score_config_match():
    assert score_config_match("3d64s  (4P ) 4p  w5Do", "3d6(3P)4s4p_w5Do[4]") == 100
    match1 = score_config_match("3d64s  (6D ) 5g  i5F ", "3d6(5D)4s5g_5Fe[4]")
    assert match1 >= 49
    assert score_config_match("3d64s  (6D ) 5g  (1S) i5F ", "3d6(5D)4s5g_5Fe[4]") == match1
    assert score_config_match("3d6    (5D ) 6s  e6D ", "3d6(5D)6se6De[9/2]") > score_config_match(
        "3d6    (5D ) 6s  e6D ", "3d6(5D)5s6De[9/2]"
    )

    assert score_config_match("Eqv st (0S ) 0s  a4P", "3d5_4Pe[4]") == 5
    assert score_config_match("3d6    (5D ) 0s  b2F ", "3d7b2Fe") >= 12
    assert score_config_match("3d5    (2D ) 4p  v3Po", "3d5(b2D)4p_3Po") == 98


def test_hydrogenic_phixs():
    ryd_to_ev = 13.605693122994232
    import artisatomic.readhillierdata as rhd

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
    import artisatomic.readhillierdata as rhd

    rhd.read_hyd_phixsdata()

    h_in_ev_seconds = 4.135667696923859e-15
    ryd_to_ev = 13.605693122994232

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
    u_grid = np.array(rhd.hyd_phixs_energygrid_ryd[(n, l_start)])
    u_grid /= u_grid[0]
    sigma_table = np.zeros(len(u_grid))
    for l in range(l_start, l_end + 1):
        sigma_table += (2 * l + 1) * np.array(rhd.hyd_phixs[(n, l)])

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
    ryd_to_ev = 13.605693122994232
    import artisatomic.readhillierdata as rhd

    rhd.read_hyd_phixsdata()

    sigma_hydrogen_1s = rhd.get_hydrogenic_n_phixstable(rhd.hc_in_ev_angstrom / ryd_to_ev, 1)[0][1]

    for atomic_number in (1, 2, 3, 6, 26):
        for n in (1, 2, 5):
            # a hydrogenic level of charge Z and principal quantum number n ionizes at Z**2 / n**2 Ryd
            threshold_ev = atomic_number**2 * ryd_to_ev / n**2
            phixstable = rhd.get_hydrogenic_n_phixstable(rhd.hc_in_ev_angstrom / threshold_ev, n)

            # Kramers: sigma_threshold = 7.91 Mb * n / Z**2 * g_bf, and the gaunt factor at
            # threshold depends only on n, so the ratio to the n=1 value is exactly n / Z**2
            # once the n-dependence of g_bf is divided out by comparing at the same n.
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

    import artisatomic
    import artisatomic.readhillierdata as rhd

    rhd.read_hyd_phixsdata()

    ryd_to_ev = 13.605693122994232
    hc_in_ev_cm = artisatomic.hc_in_ev_cm

    # a single hydrogenic n=1 level of a Z=2 ion: threshold is 4 Ryd, so sigma_th = 6.307 / 4 Mb
    ionization_energy_ev = 4 * ryd_to_ev
    dflevels = pl.DataFrame(
        {
            "levelid": [0, 1],
            "energyabovegsinpercm": [None, 0.0],
            "g": [None, 2.0],
            "levelname": [None, "s1s  1S,enpercm=0.0,j=0.5"],
        }
    )
    args = argparse.Namespace(nphixspoints=100, phixsnuincrement=0.03, optimaltemperature=6000)

    crosssections, targetfractions, thresholds = artisatomic.match_hydrogenic_phixs(
        atomic_number=2,
        energy_levels=dflevels,
        ionization_energy_ev=ionization_energy_ev,
        ion_handler="kurucz",
        args=args,
    )

    assert thresholds[1] == ionization_energy_ev
    assert targetfractions[1] == [(1, 1.0)]

    expected_threshold_mb = rhd.get_hydrogenic_n_phixstable(rhd.hc_in_ev_angstrom / ionization_energy_ev, 1)[0][1]
    assert abs(expected_threshold_mb - 6.3067 / 4) < 1e-3  # exact hydrogenic value for He II 1s
    # the downsampled first point is a bin average, so allow a few percent
    assert abs(crosssections[1][0] / expected_threshold_mb - 1) < 0.05

    # levels above the ionization energy must be skipped rather than dividing by a negative threshold
    dflevels_unbound = pl.DataFrame(
        {
            "levelid": [0, 1],
            "energyabovegsinpercm": [None, 2 * ionization_energy_ev / hc_in_ev_cm],
            "g": [None, 2.0],
            "levelname": [None, "s1s  1S,enpercm=0.0,j=0.5"],
        }
    )
    crosssections, targetfractions, thresholds = artisatomic.match_hydrogenic_phixs(
        atomic_number=2,
        energy_levels=dflevels_unbound,
        ionization_energy_ev=ionization_energy_ev,
        ion_handler="kurucz",
        args=args,
    )
    assert thresholds[1] == 0.0
    assert targetfractions[1] == []
    assert np.all(crosssections[1] == 0.0)


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
            _, energy_levels, _, _, _ = readhillierdata.read_levels_and_transitions(atomic_number, ion_stage, flog)
            upsilondict = readhillierdata.read_coldata(atomic_number, ion_stage, energy_levels, flog, args)
        levelids_of_term = defaultdict(list)
        for levelid, level in enumerate(energy_levels[1:], 1):
            levelids_of_term[level.levelname.split("[")[0]].append(levelid)
        return energy_levels, upsilondict, levelids_of_term

    energy_levels, upsilondict, levelids_of_term = read_ion(8, 3)

    lower_ids = levelids_of_term["2s2_2p2_3Pe"]  # J = 0, 1, 2 with g = 1, 3, 5
    upper_ids = levelids_of_term["2s_2p3_3Do"]
    assert [energy_levels[i].g for i in lower_ids] == [1.0, 3.0, 5.0]

    sums_from_lower = [
        sum(upsilondict[(i, j)] for j in upper_ids if upsilondict.get((i, j), -1.0) > 0.0) for i in lower_ids
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
    ion_handlers: list[tuple[int, list[int | tuple[int, str]]]] = [(26, [1, 2])]

    # adding an ion for a new element must not modify the input list
    result = add_handler_if_not_set(ion_handlers, 58, 1, "dream")
    assert ion_handlers == [(26, [1, 2])]
    assert result == [(26, [1, 2]), (58, [(1, "dream")])]

    # add an ion to an existing element
    result = add_handler_if_not_set(ion_handlers, 26, 3, "dream")
    assert ion_handlers == [(26, [1, 2])]
    assert result == [(26, [1, 2, (3, "dream")])]

    # an already-present ion stage is not replaced or duplicated
    result = add_handler_if_not_set(ion_handlers, 26, 2, "dream")
    assert result == [(26, [1, 2])]

    # ion stages with handlers can be given as tuples or lists (e.g. loaded from JSON)
    ion_handlers_json = t.cast("list[tuple[int, list[int | tuple[int, str]]]]", [(26, [[1, "cmfgen"]])])
    result = add_handler_if_not_set(ion_handlers_json, 26, 1, "dream")
    assert result == [(26, [[1, "cmfgen"]])]


def test_get_default_handler():
    assert get_default_handler(2, 3) == "boyle"
    assert get_default_handler(26, 1) == "cmfgen"
    assert get_default_handler(56, 2) == "cmfgen"
    assert get_default_handler(38, 1) == "qub_data"
    # QUB calculations take precedence over DREAM for W, Pt, and Au ion stages 1-3
    assert get_default_handler(74, 1) == "qub_data"
    assert get_default_handler(78, 3) == "qub_data"
    assert get_default_handler(79, 2) == "qub_data"
    assert get_default_handler(74, 4) == "dream"
    assert get_default_handler(60, 2) == "dream"
    assert get_default_handler(45, 1) == "kurucz"


def test_hillier_extend_ion_list():
    result = readhillierdata.extend_ion_list([], maxionstage=1, include_hydrogen=False)
    assert (2, [(1, "cmfgen")]) in result
    assert (26, [(1, "cmfgen")]) in result
    assert all(atomic_number != 1 for atomic_number, _ in result)
    assert all(entry == (1, "cmfgen") for _, listions in result for entry in listions)


def test_reduce_phixs_tables_worker():
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
    flog = io.StringIO()
    ionization_energy_ev, energylevels, upsilondict = readqubdata.read_adf04(
        (PYDIR / ".." / "atomic-data-qub" / "co_tyndall_test_sample" / "adf04_v1").resolve(), 27, 3, flog
    )
    assert abs(ionization_energy_ev - 40.964007) < 1e-5
    assert len(energylevels) - 1 == 262
    assert len(upsilondict) == 235
    level1 = energylevels[1]
    assert level1 is not None
    assert level1.levelname == "3s23p63d7(4F)_4Fe[9/2]_id=1"
    assert level1.energyabovegsinpercm == 0.0
    assert level1.g == 10.0
    assert level1.parity == 0


def test_read_nahar_energy_level_file_missing():
    # a missing file should be logged and returned as empty data, not crash
    flog = io.StringIO()
    (
        nahar_energy_levels,
        nahar_core_states,
        nahar_level_index_of_state,
        nahar_configurations,
        nahar_ionization_potential_rydberg,
    ) = readnahardata.read_nahar_energy_level_file("does/not/exist.en.ls.txt", 26, 2, flog)
    assert nahar_energy_levels == [None]
    assert nahar_core_states == []
    assert nahar_level_index_of_state == {}
    assert nahar_configurations == {}
    assert nahar_ionization_potential_rydberg == -1.0
    assert "does not exist" in flog.getvalue()


def test_nahar_get_photoiontargetfractions():
    dflower = pl.DataFrame(
        {"levelid": [0, 1], "energyabovegsinpercm": [None, 0.0], "g": [None, 9.0], "levelname": [None, "gs"]}
    )
    dfupper = pl.DataFrame(
        {"levelid": [0, 1], "energyabovegsinpercm": [None, 0.0], "g": [None, 10.0], "levelname": [None, "gs2"]}
    )
    nahar_core_states = [None, readnahardata.NaharCoreState(1, "3d6", "5De", 0.0)]
    targetlist = readnahardata.get_photoiontargetfractions(dflower, dfupper, nahar_core_states, {}, io.StringIO())
    assert targetlist[1] == [(1, 1.0)]


def test_get_level_valence_n():
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
