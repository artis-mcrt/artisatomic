#!/usr/bin/env python3
"""Tests for the artisatomic readers, parsers and output writers."""

import io
import operator
import typing as t

import numpy as np
import polars as pl
import pytest

from artisatomic import add_handler_if_not_set
from artisatomic import add_level_ids_forbidden
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

    # The only L character can belong to a parenthesised parent term. Reporting '(4D)' would
    # describe the parent rather than this level, so the term is unreadable -- but the trailing
    # 'o' still gives the parity.
    assert readhillierdata.get_term_as_tuple("3d5(4D)4po[3]") == (-1, -1, 1)
    assert readhillierdata.get_term_as_tuple("3d4(3P2)4po[1/2]") == (-1, -1, 1)

    # an L character at index 0 leaves no room for the multiplicity, and config[-1] would wrap
    # round to the end of the name and read some unrelated character as it
    assert readhillierdata.get_term_as_tuple("S2") == (-1, -1, -1)
    assert readhillierdata.get_term_as_tuple("P2") == (-1, -1, -1)


def test_get_level_parity():
    """A CMFGEN level's parity comes from its 'e'/'o' suffix, and merged levels have none at all."""
    get_level_parity = readhillierdata.get_level_parity

    # the 'e'/'o' suffix, which is what nearly every name carries
    assert get_level_parity("1s2_1Se") == 0
    assert get_level_parity("2p_2Po") == 1
    assert get_level_parity("3d5(6S)4s(7S)4d6De") == 0

    # intermediate-coupling names, where the only term letter belongs to the parent term in
    # parentheses. Reading the term instead of the suffix leaves these with no parity at all,
    # and then any two of them wrongly compare equal.
    assert get_level_parity("3d5(4D)4po[3]") == 1
    assert get_level_parity("3d4(3P2)4po[1/2]") == 1
    assert get_level_parity("3d6(3P2)4pbo[5/2]") == 1

    # no suffix: sum l over the orbitals instead
    assert get_level_parity("5s2.5p5") == 1  # 0*2 + 1*5
    assert get_level_parity("3s23p63d7(4F)") == 0  # 0*2 + 1*6 + 2*7, the parent term skipped

    # ...and where there is no suffix and no readable orbital either, there is no parity
    assert get_level_parity("Eqv st (0S ) 0s  a4P") < 0

    # Levels that merge sub-levels of both parities have no parity to read, and must not be given
    # one: '1___' and '13___' hold every l of that n (g = 2n^2), '8SNG'/'8TRP' are He I's merged
    # singlets and triplets, and 'w'/'z' are merge markers for the high-l orbitals of a shell.
    for merged in ("1___", "2___", "13___", "8SNG", "8TRP", "2s2_29w_2W", "10z_2Z", "2s2_2p3(4So)5z_5Z"):
        assert get_level_parity(merged) < 0


def test_has_merged_orbital():
    """Merge markers are orbital letters standing for several l at once, so l >= n gives them away."""
    from artisatomic import has_merged_orbital

    assert has_merged_orbital("2s2_13w_2W")  # 'w' would be l=19, but n=13
    assert has_merged_orbital("10z_2Z")  # 'z' would be l=22, but n=10
    assert has_merged_orbital("2s2_2p3(4So)5z_5Z")

    assert not has_merged_orbital("3d6(5D)10d_5Pe")  # 10d is a real orbital, l=2 < n=10
    assert not has_merged_orbital("2p_2Po")
    assert not has_merged_orbital("3d7(4F)6d_5Pbe")

    # A token with no principal quantum number cannot be judged: n would default to 0 and then
    # l >= n holds for every orbital letter, which would call the whole configuration merged and
    # throw away the parity these names state outright with their 'o'.
    for hasnon in ("3d8(2H)sp_2Go", "3d2(2D)sp_4Fo", "3d7(a2D)4sep_4Fo", "SEJ_s6p_7Fo", "3s2_3p_nd_a3Po"):
        assert not has_merged_orbital(hasnon)
        assert readhillierdata.get_level_parity(hasnon) == 1

    # ...while a real merge marker still wins over an 'e'/'o' suffix, because the level really
    # does span both parities. Al XI names one: '10z_2Zo', g=168.
    assert has_merged_orbital("10z_2Zo")
    assert readhillierdata.get_level_parity("10z_2Zo") < 0


def test_get_parity_from_multi_orbital_token():
    """A digit-letter-letter run is one token holding two orbitals that share a principal number."""
    from artisatomic import get_config_parity

    # '4sp(3P)_7Po' splits to the token '4sp', i.e. 4s and 4p: l = 0 + 1 is odd, matching the 'o'.
    # Reading only the first letter and calling the rest an occupation raises on int('p').
    assert get_config_parity("4sp(3P)_7Po") == 1
    assert readhillierdata.get_level_parity("4sp(3P)_7Po[2]") == 1


def test_get_config_parity():
    """Parity is the sum of l over the occupied orbitals, skipping parent terms and merge markers."""
    from artisatomic import get_config_parity

    # sum of l over the occupied orbitals, mod 2
    assert get_config_parity("3d64s2") == 0  # 2 * 6 = 12
    assert get_config_parity("5s2.5p5") == 1  # 0 * 2 + 1 * 5 = 5

    # parent terms in parentheses are not occupied orbitals and must be skipped, not parsed
    assert get_config_parity("3s23p63d7(4F)") == 0  # 0*2 + 1*6 + 2*7 = 20
    assert get_config_parity("3d6(5D)4s_6De") == 0  # 2 * 6 = 12

    # closed shells with two-digit occupations: a truncated '4f1' reading would give the
    # wrong (odd) parity here, since 3*14 is even but 3*1 is odd
    assert get_config_parity("4f145d96s2") == 0  # 3*14 + 2*9 + 0 = 60

    # CMFGEN packs a shell's high-l levels into one level whose orbital letter is a merge marker,
    # not a real l ('5z' would be l=22, '13w' l=19). It spans several l of both parities, so only
    # the real orbitals decide the parity.
    assert get_config_parity("2s2_2p3(4So)5z_5Z") == 1  # 2s2 + 2p3 = 3, the 5z contributes none
    assert get_config_parity("2s2_13w_2W") == 0  # 2s2 = 0, the 13w contributes none

    # A bare configuration read as a name-with-a-term is mistaken for a term, so no orbitals come
    # back. The sum is then empty, and 0 is a real parity that would be the wrong answer: right
    # for '3d7' (2 * 7 = 14) only by coincidence, and wrong for '5p3' (1 * 3 = 3, odd). None says
    # so instead. Callers holding a bare configuration pass hasterm=False, below.
    assert interpret_configuration("3d7")[0] == []
    assert get_config_parity("3d7") is None
    assert get_config_parity("5p3") is None


def test_parity_of_a_bare_configuration():
    """A name with no term is all configuration, and adf04 writes its orbitals in upper case."""
    from artisatomic import get_config_parity

    # the whole string is orbitals, so nothing is lost off the end and odd really reads as odd
    assert get_config_parity("3d7", hasterm=False) == 0  # 2 * 7 = 14
    assert get_config_parity("5s2", hasterm=False) == 0  # 0 * 2
    assert get_config_parity("5p3", hasterm=False) == 1  # 1 * 3
    assert get_config_parity("2p", hasterm=False) == 1  # a lone orbital with no occupation

    # ADAS adf04 configurations: upper case, space separated, with the level's own index in
    # brackets at the end. Stripping a term took the last orbital with it, so '4P1' was lost and
    # every level of FeIII.adf04 came out even; 3s2 3p6 3d5 4p1 is 0 + 6 + 10 + 1 = 17, odd.
    assert get_config_parity("3S2 3P6 3D5 4P1   (1)", hasterm=False) == 1
    assert get_config_parity("3S2 3P6 3D6   (5)", hasterm=False) == 0  # 0 + 6 + 12 = 18
    assert interpret_configuration("3S2 3P6 3D5 4P1   (1)", hasterm=False)[0] == [
        "3S2",
        "3P6",
        "3D5",
        "4P1",
        "(1)",
    ]

    # ...but upper case is only an orbital where there is no term to confuse it with. With a term
    # the case still matters: '8SNG' is He I's merged singlets, not an 8s orbital, and the parent
    # term left over from '3d4(3H)s44p_x3Io' is not a merge marker.
    assert get_config_parity("8SNG") is None
    assert readhillierdata.get_level_parity("8SNG") < 0
    assert readhillierdata.get_level_parity("3d4(3H)s44p_x3Io[6]") == 1


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
        assert sum(bool(targets) for targets in targetfractions) == n_expected
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


def test_read_phixs_tables_multiple_photoionisation_files(monkeypatch):
    """A level with a cross-section table in more than one phot file keeps the largest, rescaled.

    Every CMFGEN ion with several entries in ions_data[...].photfilenames has one file per final
    state of the upper ion, and a level is normally present in all of them: O I's phot_nosm_A and
    phot_nosm_B share all 107 of their configuration names. Treating the second table as an error
    made those ions (C I, C III, N I, N III, O I, O IV, F II, F III, P IV) unreadable, and none of
    them was in the tests/ matrix, which was Fe/Co/Ni only.

    Only one table can be written per level, so the one with the largest threshold cross section
    wins. write_phixs_data() writes it as the level's TOTAL and splits it over the targets, so it
    must first be divided by the kept target's own fraction, which is what this pins: reading each
    phot file alone gives that target's raw table, and the combined read must reproduce the winner
    divided by its fraction.
    """
    import argparse
    import contextlib

    ionfiles = readhillierdata.ions_data[8, 1]
    # checked before the expensive reads below, so a change here fails as itself
    assert len(ionfiles.photfilenames) == 2, f"O I is expected to have two phot files, got {ionfiles.photfilenames}"

    rhd.read_hyd_phixsdata()
    args = argparse.Namespace(nphixspoints=100, phixsnuincrement=0.03, optimaltemperature=6000)

    def read_phixs(photfilenames):
        """Read O I's cross sections using only the named phot files."""
        monkeypatch.setitem(readhillierdata.ions_data, (8, 1), ionfiles._replace(photfilenames=photfilenames))
        flog = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()):
            _, dflevels, _, _ = rhd.read_levels_and_transitions(8, 1, flog)
            crosssections, targetconfigs, _thresholds = rhd.read_phixs_tables(8, 1, dflevels, args, flog)
        return crosssections, targetconfigs, flog.getvalue()

    file_a, file_b = ionfiles.photfilenames
    crosssections_a, targets_a, _ = read_phixs([file_a])
    crosssections_b, targets_b, _ = read_phixs([file_b])
    crosssections, targetconfigs, log = read_phixs([file_a, file_b])

    # the duplicates are reported rather than raised
    assert "has a cross section table in more than one photoionisation file" in log

    # every level that was read has a table, not just the ones from the first file
    assert all(np.any(crosssections[levelid]) for levelid, targets in enumerate(targetconfigs) if targets)

    # a level in both files ends up with both targets recorded
    levelids_in_both = [
        levelid
        for levelid, targets in enumerate(targetconfigs)
        if targets and len(targets) > 1 and np.any(crosssections_a[levelid]) and np.any(crosssections_b[levelid])
    ]
    assert levelids_in_both, "expected O I levels with a cross section table in both phot files"

    for levelid in levelids_in_both:
        # each single-file read has one target holding 100% of the level, so its table is unrescaled
        targetlist_a, targetlist_b, targetlist = targets_a[levelid], targets_b[levelid], targetconfigs[levelid]
        assert targetlist_a is not None
        assert targetlist_b is not None
        assert targetlist is not None

        threshold_a = crosssections_a[levelid][np.nonzero(crosssections_a[levelid])][0]
        threshold_b = crosssections_b[levelid][np.nonzero(crosssections_b[levelid])][0]
        kept_crosssections, kept_target = (
            (crosssections_a[levelid], targetlist_a[0][0])
            if threshold_a >= threshold_b
            else (crosssections_b[levelid], targetlist_b[0][0])
        )
        kept_fraction = dict(targetlist)[kept_target]
        # a table left at one target's share would be low by exactly this factor
        assert 0.0 < kept_fraction < 1.0
        assert np.allclose(crosssections[levelid], kept_crosssections / kept_fraction, rtol=1e-10)


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


def test_add_level_ids_forbidden_parity():
    """Equal parity means forbidden, but a null parity is absent and matches nothing, itself included."""
    dflevels = pl.DataFrame(
        {
            "levelid": [0, 1, 2, 3, 4],
            "parity": [0, 1, None, None, 0],  # two real parities, two absent ones, then a repeat of 0
        },
        schema={"levelid": pl.Int64, "parity": pl.Int64},
    )
    dftransitions = pl.DataFrame(
        {
            "lowerlevel": [0, 0, 0, 2, 0],
            "upperlevel": [1, 2, 3, 3, 4],
            "A": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    forbidden = add_level_ids_forbidden(dflevels, dftransitions)["forbidden"].to_list()

    # even -> odd permitted; a real parity never matches an absent one; two absent ones do not
    # match each other either, which is the whole reason absence is spelled null; two real
    # even ones do match
    assert forbidden == [False, False, False, False, True]


def test_get_level_j():
    """J is read from the brackets a J-resolved CMFGEN level name ends with."""
    get_level_j = readhillierdata.get_level_j

    assert get_level_j("3d6_a5De[4]") == 4.0
    assert get_level_j("3d5(4D)4po[9/2]") == 4.5
    assert get_level_j("3d4(3P2)4po[1/2]") == 0.5

    # a name with a brace and nothing after it gives J there; in pair coupling the brace holds
    # K and the trailing bracket is J. The last group is the one read, and both come out right.
    assert get_level_j("2s2_2p(2P<1/2>)4f_2{5/2}e") == 2.5
    assert get_level_j("2p5(2P*<1/2>)3d_2{3/2}o[1]") == 1.0

    # Not every trailing bracket is a J: Si X and S X number their levels there instead. g tells
    # them apart, because J is only J where g == 2J + 1. A 3P term has no J = 3.
    assert get_level_j("2p3p3Pe[3]", g=5.0) is None
    assert get_level_j("2s2_2p3_4So[2]", g=4.0) is None
    assert get_level_j("3d6_a5De[4]", g=9.0) == 4.0

    # A term-resolved level has no J of its own -- its g counts every J of the term -- and nor
    # does a merged level. Neither may be given one.
    for noj in ("1___", "8SNG", "10z_2Zo", "2s2_2p3(4So)5z_5Z", "3d6(5D)4s_6De"):
        assert get_level_j(noj) is None


def test_add_level_ids_forbidden_delta_j():
    """E1 needs |dJ| <= 1 and forbids J=0 -> J=0, on a transition that carries no f."""
    # opposite parities throughout, so the Laporte rule can never fire and only dJ decides
    dflevels = pl.DataFrame(
        {
            "levelid": [0, 1, 2, 3, 4],
            "parity": [0, 1, 0, 1, 0],
            "j": [1.0, 2.0, 3.0, 0.0, 0.0],
        },
        schema={"levelid": pl.Int64, "parity": pl.Int64, "j": pl.Float64},
    )
    # A = 0 marks the upsilon-only pairs, which no source called an electric dipole line
    dftransitions = pl.DataFrame(
        {
            "lowerlevel": [0, 0, 4, 0],
            "upperlevel": [1, 2, 3, 3],
            "A": [0.0, 0.0, 0.0, 0.0],
        }
    )
    result = add_level_ids_forbidden(dflevels, dftransitions)
    # the joins do not keep the row order, so read the answers back by transition
    forbidden = {(lo, up): f for lo, up, f in result[["lowerlevel", "upperlevel", "forbidden"]].iter_rows()}

    assert forbidden[0, 1] is False  # J 1 -> 2, dJ = 1
    assert forbidden[0, 2] is True  # J 1 -> 3, dJ = 2
    assert forbidden[4, 3] is True  # J 0 -> 0, forbidden even though dJ = 0
    assert forbidden[0, 3] is False  # J 1 -> 0, dJ = 1


def test_add_level_ids_forbidden_delta_j_yields_to_an_oscillator_strength():
    """A source that gives the transition an f says it is E1, and that beats the J labels.

    Some data sets disagree with themselves. CMFGEN's provisional F III set splits a term by a
    nominal 0.8 cm-1 and then shares the term's f over all the J pairs, so it lists dJ = 2 lines
    with f as large as 0.116. To call those forbidden would give a strong line the forbidden
    collision approximation.
    """
    dflevels = pl.DataFrame(
        {"levelid": [0, 1], "parity": [1, 0], "j": [0.5, 2.5]},
        schema={"levelid": pl.Int64, "parity": pl.Int64, "j": pl.Float64},
    )
    dftransitions = pl.DataFrame({"lowerlevel": [0], "upperlevel": [1], "A": [3.4e9], "f": [0.116]})

    result = add_level_ids_forbidden(dflevels, dftransitions)

    # dJ = 2, so the rule is broken and reported, but the f keeps the transition permitted
    assert result["breaksdeltaj"].to_list() == [True]
    assert result["forbidden"].to_list() == [False]

    # with no f and no A, the same pair is forbidden
    nof = dftransitions.with_columns(A=0.0).drop("f")
    assert add_level_ids_forbidden(dflevels, nof)["forbidden"].to_list() == [True]

    # A weak f is not evidence of an E1 line, so the J labels decide it. C IV lists
    # 2p_2Po[1/2] -> 3d_2De[5/2] at f = 2.7e-10, and its 108 cm-1 fine structure means its J
    # labels are sound, so that line really is forbidden.
    weakf = dftransitions.with_columns(A=4.1, f=2.7e-10)
    assert add_level_ids_forbidden(dflevels, weakf)["forbidden"].to_list() == [True]


def test_log_deltaj_contradictions_judges_f_and_a_separately():
    """Only a transition strong enough to be E1 contradicts its own J labels.

    f has no units and an E1 line carries 1e-3 to 1. A is a rate in s-1 over many decades, where
    a forbidden line still reaches ~1e2, so the same cut would report every forbidden line a
    reader supplies A for as a contradiction.
    """
    from artisatomic.output import log_deltaj_contradictions

    def warnings_for(dftransitions: pl.DataFrame) -> str:
        flog = io.StringIO()
        log_deltaj_contradictions(flog, dftransitions, "Test II")
        return flog.getvalue()

    breaksrule = {"lowerlevel": [0], "upperlevel": [1], "breaksdeltaj": [True]}

    # f: a strong line contradicts the labels, a forbidden line's own small f does not
    assert "WARNING" in warnings_for(pl.DataFrame({**breaksrule, "A": [3.4e9], "f": [0.116]}))
    assert not warnings_for(pl.DataFrame({**breaksrule, "A": [1.0e-2], "f": [1.9e-9]}))
    # ...and the quiet case is the one the rule now marks forbidden, so the two agree
    assert not warnings_for(pl.DataFrame({**breaksrule, "A": [4.1], "f": [2.7e-10]}))

    # A, where a forbidden line reaches 14 s-1 in QUB's Co III and must stay quiet
    assert "WARNING" in warnings_for(pl.DataFrame({**breaksrule, "A": [1.9e8]}))
    assert not warnings_for(pl.DataFrame({**breaksrule, "A": [14.0]}))

    # a transition that keeps the rule is never reported, however strong it is
    assert not warnings_for(pl.DataFrame({**breaksrule, "breaksdeltaj": [False], "A": [1.9e8]}))


def test_add_level_ids_forbidden_delta_j_needs_both_levels():
    """A level with no J turns off the dJ rule for its transitions. It does not undo the parities."""
    dflevels = pl.DataFrame(
        {"levelid": [0, 1, 2], "parity": [0, 1, 1], "j": [1.0, None, 5.0]},
        schema={"levelid": pl.Int64, "parity": pl.Int64, "j": pl.Float64},
    )
    dftransitions = pl.DataFrame({"lowerlevel": [0, 1], "upperlevel": [1, 2], "A": [0.0, 0.0]})

    # the first pair has no J to compare and two different parities, so nothing can call it
    # forbidden. The second pair shares a parity, which the missing J must not undo.
    assert add_level_ids_forbidden(dflevels, dftransitions)["forbidden"].to_list() == [False, True]

    # a frame with no j column at all behaves as it did before this code read J
    nojcol = dflevels.drop("j")
    assert add_level_ids_forbidden(nojcol, dftransitions)["forbidden"].to_list() == [False, True]


def test_add_level_ids_forbidden_treats_negative_parity_as_a_real_one():
    """Absence is null and only null, so a negative number is an ordinary parity that can match.

    Three readers used to spell "no parity" as the negated level id, which made level 0 come out
    as a real even parity while every other level was unmatchable. Nothing may depend on negative
    numbers meaning absent any more, so two levels sharing one are forbidden like any other pair.
    """
    dflevels = pl.DataFrame({"levelid": [0, 1], "parity": [-7, -7]}, schema={"levelid": pl.Int64, "parity": pl.Int64})
    dftransitions = pl.DataFrame({"lowerlevel": [0], "upperlevel": [1], "A": [1.0]})

    assert add_level_ids_forbidden(dflevels, dftransitions)["forbidden"].to_list() == [True]


@pytest.mark.parametrize(
    ("parity", "dtype"),
    [
        ([None, None], pl.Int64),  # the canonical spelling of an absent parity
        ([float("nan"), float("nan")], pl.Float64),  # NaN compares equal to itself, so it must cast away
        (["1", "1"], pl.String),  # pandas infers text from a blank column, e.g. FAC's P
    ],
)
def test_add_level_ids_forbidden_unreadable_parity(parity, dtype):
    """A parity that is not a whole number means the level has none, not that it matches itself."""
    dflevels = pl.DataFrame({"levelid": [0, 1], "parity": parity}, schema={"levelid": pl.Int64, "parity": dtype})
    dftransitions = pl.DataFrame({"lowerlevel": [0], "upperlevel": [1], "A": [1.0]})

    forbidden = add_level_ids_forbidden(dflevels, dftransitions)["forbidden"].to_list()

    # a readable string parity is a real parity and still counts; the rest are unknown
    assert forbidden == [dtype == pl.String]
    assert forbidden[0] is not None  # would raise in write_transition_data


def test_readhillierdata_hydrogen_lyman_alpha_is_permitted():
    """H I is merged n-levels throughout, so nothing in it may come out forbidden.

    Every H I level is named '<n>___' and holds every l of that n, giving it no definite parity.
    Treating those as a matching parity made all 435 H I transitions forbidden, Lyman alpha
    included -- the strongest permitted line there is, listed in hi_osc.dat with f = 0.4162.
    """
    _, dflevels, dftransitions, _ = readhillierdata.read_levels_and_transitions(1, 1, io.StringIO())

    # no level has a parity that could match another's
    assert dflevels["parity"].is_null().all()

    dflevels = dflevels.with_row_index("levelid").with_columns(pl.col("levelid").cast(pl.Int64))
    dftransitions = add_level_ids_forbidden(dflevels, dftransitions)
    assert not any(dftransitions["forbidden"].to_list())

    lymanalpha = dftransitions.filter((pl.col("lowerlevel") == 0) & (pl.col("upperlevel") == 1))
    assert lymanalpha.height == 1
    assert lymanalpha["A"].item() == pytest.approx(4.696e8)
    assert not lymanalpha["forbidden"].item()


def test_readboyledata_levels_have_no_parity(monkeypatch):
    """The AOIFE data set supplies no parities, so no transition may come out forbidden.

    add_level_ids_forbidden() marks a transition forbidden when its two levels share a parity, so
    giving every level the same parity made every transition of the ion forbidden — and helium has
    plenty of permitted ones. A null parity cannot match another, which is how readlisbondata and
    readkuruczdata say the same thing.

    The aoife.hdf5 in the repository is a placeholder (the real file is gitignored, fetched per
    its README), so this builds the three tables the reader wants in memory.
    """
    import h5py  # pyright: ignore[reportMissingTypeStubs]

    from artisatomic import readboyledata

    # compound dtypes, as a real HDF5 table has: the integer columns must stay integers, or the
    # level names format differently in the two readers below
    levels_dtype = [("atomic_number", "i8"), ("ion_number", "i8"), ("level_number", "i8")]
    levels_dtype += [("energy", "f8"), ("g", "f8"), ("metastable", "i8")]
    lines_dtype = [("line_id", "i8"), ("wavelength", "f8"), ("atomic_number", "i8"), ("ion_number", "i8")]
    lines_dtype += [("f_ul", "f8"), ("f_lu", "f8"), ("level_number_lower", "i8"), ("level_number_upper", "i8")]
    lines_dtype += [("nu", "f8"), ("B_lu", "f8"), ("B_ul", "f8"), ("A_ul", "f8")]

    # He I: three levels, and two lines between them
    levels_data = np.array(
        [
            (2, 0, 0, 0.0, 1.0, 0),
            (2, 0, 1, 159856.0, 3.0, 1),
            (2, 0, 2, 166278.0, 1.0, 0),
            (2, 1, 0, 0.0, 2.0, 0),  # He II, must be filtered out
        ],
        dtype=levels_dtype,
    )
    lines_data = np.array(
        [
            (0, 584.0, 2, 0, 0.1, 0.3, 0, 2, 0.0, 0.0, 0.0, 1.8e9),
            (1, 10830.0, 2, 0, 0.2, 0.6, 1, 2, 0.0, 0.0, 0.0, 1.0e7),
        ],
        dtype=lines_dtype,
    )

    with h5py.File("aoife-test", "w", driver="core", backing_store=False) as fakefile:
        fakefile.create_dataset("/levels_data", data=levels_data)
        fakefile.create_dataset("/lines_data", data=lines_data)
        monkeypatch.setattr(readboyledata, "get_aoife_dataset", lambda: fakefile)

        energy_levels = readboyledata.read_levels_data(2, 1)
        transitions, transition_count_of_level_name = readboyledata.read_lines_data(2, 1)

    # the other ion's level is filtered out
    assert len(energy_levels) == 3

    # no level has a parity, so no pair of them can compare equal
    assert all(level.parity is None for level in energy_levels)

    dflevels = leveltuples_to_pldataframe(
        pl.DataFrame(
            {
                "levelname": [level.levelname for level in energy_levels],
                "energyabovegsinpercm": [level.energyabovegsinpercm for level in energy_levels],
                "g": [level.g for level in energy_levels],
                "parity": [level.parity for level in energy_levels],
            }
        )
    )
    dftransitions = pl.DataFrame(
        {
            "lowerlevel": [t.lowerlevel for t in transitions],
            "upperlevel": [t.upperlevel for t in transitions],
            "A": [t.A for t in transitions],
        }
    )
    assert not any(add_level_ids_forbidden(dflevels, dftransitions)["forbidden"].to_list())

    # the two readers must agree on the level names, or the adata.txt transition counts land on
    # levels that do not exist: one formatted the number through int() and the other did not
    assert set(transition_count_of_level_name) <= {level.levelname for level in energy_levels}
    assert transition_count_of_level_name[energy_levels[2].levelname] == 2


def test_readlisbondata_maps_file_indices_to_energy_sorted_ids():
    """Lisbon lines name their levels by position in the levels file, which is re-sorted by energy.

    Indexing the sorted list with the file's own position attaches every transition to the wrong
    pair of levels whenever the source CSV is not already in energy order, which is what the map
    returned by read_levels_data() (as in readfacdata) is for. The map is keyed by row position
    rather than index label, matching the levels.iloc[...] lookup LisbonReader itself does.
    """
    import pandas as pd

    from artisatomic import readlisbondata

    # deliberately not in energy order: file index 0 is the HIGHEST level, 2 the ground state
    dflevels = pd.DataFrame(
        {"energy": [5000.0, 1000.0, 0.0], "j": [2.0, 1.0, 0.0], "label": ["top", "mid", "gs"]}, index=[0, 1, 2]
    )

    energy_levels, levelid_of_fileindex = readlisbondata.read_levels_data(dflevels)

    # levels come back in ascending energy, so the file's order is exactly reversed
    assert [level.energyabovegsinpercm for level in energy_levels] == [0.0, 1000.0, 5000.0]
    assert levelid_of_fileindex == {2: 0, 1: 1, 0: 2}
    # a null parity never matches another, so the Laporte rule cannot fire...
    assert all(level.parity is None for level in energy_levels)
    # ...and J is what the delta J rule has to go on, so it must reach the level tuple
    assert [level.j for level in energy_levels] == [0.0, 1.0, 2.0]

    # one line from the file's level 2 (the ground state) to its level 0 (the top level)
    dflines = pd.DataFrame(
        {"A": [1.5e8]},
        index=pd.MultiIndex.from_tuples([(2, 0)], names=["level_index_lower", "level_index_upper"]),
    )
    transitions, transition_count_of_level_name = readlisbondata.read_lines_data(
        energy_levels, dflines, levelid_of_fileindex
    )

    # ...which is level id 0 -> 2 after the sort, written with the lower id first
    assert len(transitions) == 1
    assert (transitions[0].lowerlevel, transitions[0].upperlevel) == (0, 2)
    assert transitions[0].A == 1.5e8
    assert transition_count_of_level_name == {energy_levels[0].levelname: 1, energy_levels[2].levelname: 1}

    # a line naming a level the table does not have means the two files disagree about the
    # numbering. Skipping it would drop every transition and write a silently empty ion.
    dflines_unknown = pd.DataFrame(
        {"A": [1.0]},
        index=pd.MultiIndex.from_tuples([(2, 99)], names=["level_index_lower", "level_index_upper"]),
    )
    with pytest.raises(ValueError, match="names level index 99"):
        readlisbondata.read_lines_data(energy_levels, dflines_unknown, levelid_of_fileindex)


def test_add_handler_if_not_set():
    """Adding a handler returns a new list and never overrides an ion that is already present."""
    ion_handlers: list[tuple[int, list[tuple[int, str]]]] = [(26, [(1, "cmfgen"), (2, "cmfgen")])]
    unchanged = [(26, [(1, "cmfgen"), (2, "cmfgen")])]

    # add an ion for a new element
    result = add_handler_if_not_set(ion_handlers, 58, 1, "dream")
    assert result == [(26, [(1, "cmfgen"), (2, "cmfgen")]), (58, [(1, "dream")])]

    # add an ion to an existing element
    result = add_handler_if_not_set(ion_handlers, 26, 3, "dream")
    assert result == [(26, [(1, "cmfgen"), (2, "cmfgen"), (3, "dream")])]

    # an already-present ion stage keeps the handler it was given, whatever the new one says
    result = add_handler_if_not_set(ion_handlers, 26, 2, "dream")
    assert result == unchanged

    # the calls return a new list each time, so none of them touched the one they were given
    assert ion_handlers == unchanged

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


def test_parent_elevel_zero_normalisation_is_anchored():
    r"""Only a parent level that IS 0.0 may be renamed to 0, not one that merely contains it.

    polars' str.replace() takes a pattern and matches anywhere in the value, so a bare "0.0" also
    rewrote "10.05" to "105" and "100.0" to "100", moving those levels into the wrong group_by
    bucket in download_gammaspec_betaminus_alpha. literal=True does not help: "0.0" really is a
    substring of "10.05". Anchoring with ^...$ is what confines it to the whole value.
    """
    parent_elevels = ["0.0", "0", "10.05", "100.0", "1234.5", "0.05"]
    normalised = (
        pl.DataFrame({"parent_elevel": parent_elevels})
        .with_columns(pl.col("parent_elevel").str.replace(r"^0\.0$", "0"))["parent_elevel"]
        .to_list()
    )

    # the ground state, however it is spelled, collapses to one group; nothing else moves
    assert normalised == ["0", "0", "10.05", "100.0", "1234.5", "0.05"]
    assert all(float(before) == float(after) for before, after in zip(parent_elevels, normalised, strict=True))


def test_parallel_map_rejects_iterables_of_different_lengths():
    """A short iterable is refused, whichever path the call would otherwise have taken."""
    from artisatomic import parallel_map

    # Executor.map() and thread_map() stop at the shortest iterable while the serial shortcut's
    # zip(strict=True) raises, so the check has to happen before the path is chosen. 4 items takes
    # the shortcut and 40 the pool, and neither may quietly do less work than it was asked for.
    for nitems in (4, 40):
        with pytest.raises(ValueError, match=r"different lengths"):
            parallel_map(operator.sub, range(nitems), range(nitems - 1))


def test_parallel_map_matches_serial_results_on_both_sides_of_the_cutoff():
    """Both paths apply fn to one item of each iterable, in the order they were given."""
    from artisatomic import parallel_map

    # 4 items takes the serial shortcut, 40 goes to the pool. Subtraction does not commute, so
    # the results also pin which iterable reaches which parameter.
    for nitems in (4, 40):
        minuends = list(range(nitems))
        subtrahends = list(range(100, 100 + nitems))
        expected = [minuend - subtrahend for minuend, subtrahend in zip(minuends, subtrahends, strict=True)]

        assert parallel_map(operator.sub, minuends, subtrahends) == expected

    # a generator is consumed once and has no length, so it must survive being materialised
    assert parallel_map(operator.sub, (i for i in range(4)), [1] * 4) == [-1, 0, 1, 2]


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


def test_parse_gfall_collapses_label_whitespace():
    r"""Kurucz labels are fixed-width padded, so their whitespace runs must be collapsed.

    Expr.replace() swaps whole values equal to a literal, so `.replace(r"\s+", " ")` on a label
    column matched nothing and every run survived into the level names written to adata.txt.
    Only `.str.replace_all()` treats the argument as a pattern.
    """
    gfall = readkuruczdata.parse_gfall(str(readkuruczdata.find_gfall(38, 0))).collect()

    labels = pl.concat([gfall["label_lower"], gfall["label_upper"]]).unique().to_list()
    assert labels, "expected some labels for Sr I"
    # no run of two or more spaces survives, and none is left padded at either end
    assert not [label for label in labels if "  " in label]
    assert all(label == label.strip() for label in labels)
    # the collapse must join the parts rather than delete the separator ('s4d  1D' -> 's4d 1D')
    assert any(" " in label for label in labels)


def test_get_level_valence_n():
    """Each reader's level names yield the valence electron's principal quantum number."""
    # each handler has its own level-name format and parser
    assert readkuruczdata.get_level_valence_n("s5p  3P,enpercm=14276.381,j=0.0") == 5
    assert readtanakajpltdata.get_level_valence_n("2,even,{  4d- 3  4d+ 1  5s+ 1 }") == 5

    # the 2024 Ge-sequence JPLT files append an LS-coupled term string after the configuration
    assert (
        readtanakajpltdata.get_level_valence_n(
            "1,even,3s2_3p6_3d10_4s2_4p2                  3s(2).3p(6).3d(10).4s(2).4p(2)_3P"
        )
        == 4
    )
    # parent terms may follow a shell ("4p(3)4S") or stand as their own segment ("3P2_3P.7p")
    assert readtanakajpltdata.get_level_valence_n("6,odd,3s2_3p6_3d10_4s4p3   3s(2).3p(6).3d(10).4s.4p(3)4S_5S") == 4
    assert (
        readtanakajpltdata.get_level_valence_n("60,odd,3d(10)1S0.4s(2).4p(6).4d(10)1S0_1S.5s(2).5p(2)3P2_3P.7p_4D") == 7
    )
    assert readtanakajpltdata.get_level_valence_n("1,even,5d(10).6s(2).6p(6).7s(2)_1S") == 7
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


def test_readkuruczdata_drops_repeated_lines_but_not_merged_ones(monkeypatch):
    """Gfall repeats some lines, and separately lists distinct lines that share a level pair.

    A repeat has the same labels at both ends, and is one line given twice, once at its observed
    wavelength and once at the Ritz one. ARTIS adds the A values of two rows that share a level
    pair (input.cc), so keeping both would double the line.

    Rows that share a level pair with DIFFERENT labels are separate transitions whose levels the
    (energy, J) key merged into one. Sr I has 785 of those, two of them strong, and dropping them
    would delete real lines. They are left for ARTIS to combine.
    """
    from artisatomic import readkuruczdata

    # pin the committed sample, so the answer does not depend on which corpus is unpacked
    monkeypatch.setattr(readkuruczdata, "kuruczdatapath", PYDIR / ".." / "atomic-data-kurucz" / "test_sample")

    gfall = readkuruczdata.parse_gfall(str(readkuruczdata.find_gfall(39, 1))).collect()
    levelkey = ["energyabovegsinpercm_lower", "j_lower", "energyabovegsinpercm_upper", "j_upper"]
    # Y II holds one repeat: s5p z3P -> d5d g3D at both 241.7267 and 241.7308 nm
    assert gfall.height - gfall.unique(levelkey).height == 1
    assert gfall.height - gfall.unique([*levelkey, "label_lower", "label_upper"]).height == 1

    _, dflevels, transitions, _ = readkuruczdata.read_levels_and_transitions(39, 2, io.StringIO())

    # the repeat is gone, and no level pair is left with two rows for ARTIS to add up
    assert transitions.height == gfall.height - 1
    assert transitions.group_by(["lowerlevel", "upperlevel"]).len().filter(pl.col("len") > 1).is_empty()

    # the surviving row keeps one line's A, not the sum of the two
    levelid_of_name = dict(dflevels.select("levelname", "levelid").iter_rows())
    lower = levelid_of_name["s5p z3P,enpercm=23776.241,j=1.0"]
    upper = levelid_of_name["d5d g3D,enpercm=65132.0,j=1.0"]
    kept = transitions.filter((pl.col("lowerlevel") == lower) & (pl.col("upperlevel") == upper))
    assert kept.height == 1
    assert kept["A"].item() == pytest.approx(3.805e8, rel=1e-3)


def test_write_phixs_data_keeps_a_table_with_no_threshold():
    """A cross section is real data; a threshold ARTIS never reads is not a reason to drop it."""
    import argparse

    from artisatomic import write_phixs_data

    args = argparse.Namespace(optimaltemperature=3000, nphixspoints=2, phixsnuincrement=0.1)
    crosssections = np.array([[1.0, 0.5], [2.0, 1.0]])
    targetfractions = [[(0, 1.0)], [(0, 1.0)]]
    thresholds = np.array([13.6, np.nan])  # the second level's threshold is unknown

    out = io.StringIO()
    write_phixs_data(out, 8, 1, crosssections, targetfractions, thresholds, args, io.StringIO())
    written = out.getvalue()

    # one header line per level: both get a table, where level 2 used to be discarded outright
    headers = [line for line in written.splitlines() if line.split()[0] == "8"]
    assert len(headers) == 2
    assert headers[0].split()[-2:] == ["1", "1.360000E+01"]
    # the unknown threshold is written as zero, not as a NaN the reader would choke on
    assert headers[1].split()[-2:] == ["2", "0.000000E+00"]
    assert "nan" not in written.lower()

    # ...and level 2's cross sections are all there
    assert written.splitlines()[4:] == ["  2.00000000E+00", "  1.00000000E+00"]
