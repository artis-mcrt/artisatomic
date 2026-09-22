#!/usr/bin/env python3
"""Tests for the artisatomic readers, parsers and output writers."""

import argparse
import contextlib
import functools
import importlib
import io
import json
import operator
import pickle  # ruff: ignore[suspicious-pickle-import]  # the test writes a pandas HDFStore
import typing as t
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from xopen import xopen

from artisatomic import readadasdata
from artisatomic import readfacdata
from artisatomic import readfloers25data
from artisatomic import readhillierdata
from artisatomic import readhillierdata as rhd
from artisatomic import readkuruczdata
from artisatomic import readmonsdata
from artisatomic import readtanakajpltdata
from artisatomic.base import add_handlers_if_not_set
from artisatomic.base import drop_transitions_of_levels
from artisatomic.base import gf_to_a_coefficient
from artisatomic.base import h_in_ev_seconds
from artisatomic.base import hc_in_ev_angstrom
from artisatomic.base import hc_in_ev_cm
from artisatomic.base import leveltuples_to_pldataframe
from artisatomic.base import output_xgrid
from artisatomic.base import PhixsData
from artisatomic.base import PYDIR
from artisatomic.base import rewrite_file_as_utf8
from artisatomic.base import ryd_to_ev
from artisatomic.base import scan_file_lines
from artisatomic.base import transition_count_of_level
from artisatomic.base import xopen_check_extension
from artisatomic.cli import build_parser
from artisatomic.levelnames import convert_eissner_to_standard
from artisatomic.levelnames import expand_standard_config
from artisatomic.levelnames import get_config_parity
from artisatomic.levelnames import has_merged_orbital
from artisatomic.levelnames import interpret_configuration
from artisatomic.levelnames import is_eissner_config
from artisatomic.output import add_level_ids_forbidden
from artisatomic.output import write_adata
from artisatomic.output import write_phixs_data
from artisatomic.output import write_transition_data
from artisatomic.phixs import combine_phixs_routes
from artisatomic.phixs import match_hydrogenic_phixs
from artisatomic.phixs import PHIXS_TARGET_FRACTION_CUT
from artisatomic.phixs import reduce_phixs_tables_worker


def phixs_args(**overrides: t.Any) -> argparse.Namespace:
    """Build the command-line options with their defaults, so the tests run the values the command runs."""
    args = build_parser().parse_args([])
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def test_interpret_term():
    """get_term_as_tuple() reads the LS term from a level name, and reports unknown for an unreadable name."""
    assert readhillierdata.get_term_as_tuple("3d5(6S)4s(7S)4d6De") == (6, 2, 0)
    assert readhillierdata.get_term_as_tuple("3d6_3P2e") == (3, 1, 0)

    # a name with no L character must report "unknown", not raise UnboundLocalError
    for unreadable in ("e2x", "o12", "3d5", "12"):
        assert readhillierdata.get_term_as_tuple(unreadable) == (-1, -1, -1)

    # The only L character can belong to a parenthesised parent term. A report of '(4D)' would
    # describe the parent and not this level, so the term is unreadable. The trailing 'o' still
    # gives the parity.
    assert readhillierdata.get_term_as_tuple("3d5(4D)4po[3]") == (-1, -1, 1)
    assert readhillierdata.get_term_as_tuple("3d4(3P2)4po[1/2]") == (-1, -1, 1)

    # an L character at index 0 leaves no room for the multiplicity. config[-1] would then wrap
    # round to the end of the name and read some unrelated character as the multiplicity
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
    # parentheses. A read of the term instead of the suffix leaves these with no parity at all,
    # and then any two of them wrongly compare equal.
    assert get_level_parity("3d5(4D)4po[3]") == 1
    assert get_level_parity("3d4(3P2)4po[1/2]") == 1
    assert get_level_parity("3d6(3P2)4pbo[5/2]") == 1

    # no suffix: sum l over the orbitals instead
    assert get_level_parity("5s2.5p5") == 1  # 0*2 + 1*5
    assert get_level_parity("3s23p63d7(4F)") == 0  # 0*2 + 1*6 + 2*7, the parent term skipped

    # ...and where there is no suffix and no readable orbital either, there is no parity
    assert get_level_parity("Eqv st (0S ) 0s  a4P") < 0

    # Levels that merge sub-levels of both parities have no parity to read, and must not get one.
    # '1___' and '13___' hold every l of that n (g = 2n^2). '8SNG' and '8TRP' are He I's merged
    # singlets and triplets. 'w' and 'z' are merge markers for the high-l orbitals of a shell.
    for merged in ("1___", "2___", "13___", "8SNG", "8TRP", "2s2_29w_2W", "10z_2Z", "2s2_2p3(4So)5z_5Z"):
        assert get_level_parity(merged) < 0


def test_has_merged_orbital():
    """Merge markers are orbital letters that stand for several l at once: w and z, or a letter with l >= n."""
    assert has_merged_orbital("2s2_13w_2W")  # w is a merge marker at any n
    assert has_merged_orbital("10z_2Z")  # z too
    assert has_merged_orbital("2s2_2p3(4So)5z_5Z")

    assert not has_merged_orbital("3d6(5D)10d_5Pe")  # 10d is a real orbital, l=2 < n=10
    assert not has_merged_orbital("2p_2Po")
    assert not has_merged_orbital("3d7(4F)6d_5Pbe")

    # has_merged_orbital() cannot judge a token with no principal quantum number. n would default
    # to 0, and then l >= n holds for every orbital letter. That would call the whole configuration
    # merged and discard the parity that these names state with their 'o'.
    for hasnon in ("3d8(2H)sp_2Go", "3d2(2D)sp_4Fo", "3d7(a2D)4sep_4Fo", "SEJ_s6p_7Fo", "3s2_3p_nd_a3Po"):
        assert not has_merged_orbital(hasnon)
        assert readhillierdata.get_level_parity(hasnon) == 1

    # ...while a real merge marker still wins over an 'e'/'o' suffix, because the level really
    # does span both parities. Al XI names one: '10z_2Zo', g=168.
    assert has_merged_orbital("10z_2Zo")
    assert readhillierdata.get_level_parity("10z_2Zo") < 0


def test_get_parity_from_multi_orbital_token():
    """A digit-letter-letter run is one token that holds two orbitals that share a principal number."""
    # '4sp(3P)_7Po' splits to the token '4sp', i.e. 4s and 4p: l = 0 + 1 is odd, which matches the 'o'.
    # A read of only the first letter, with the rest as an occupation, raises on int('p').
    assert get_config_parity("4sp(3P)_7Po") == 1
    assert readhillierdata.get_level_parity("4sp(3P)_7Po[2]") == 1


def test_get_config_parity():
    """Parity is the sum of l over the occupied orbitals, and the sum skips parent terms and merge markers."""
    # sum of l over the occupied orbitals, mod 2
    assert get_config_parity("3d64s2") == 0  # 2 * 6 = 12
    assert get_config_parity("5s2.5p5") == 1  # 0 * 2 + 1 * 5 = 5

    # parent terms in parentheses are not occupied orbitals, so the parser must skip them, not parse them
    assert get_config_parity("3s23p63d7(4F)") == 0  # 0*2 + 1*6 + 2*7 = 20
    assert get_config_parity("3d6(5D)4s_6De") == 0  # 2 * 6 = 12

    # closed shells with two-digit occupations: a truncated read of '4f1' would give the
    # wrong (odd) parity here. 3*14 is even, but 3*1 is odd
    assert get_config_parity("4f145d96s2") == 0  # 3*14 + 2*9 + 0 = 60

    # CMFGEN packs a shell's high-l levels into one level whose orbital letter is a merge marker
    # (w or z), not a real l. It spans several l of both parities, so only the real orbitals
    # decide the parity.
    assert get_config_parity("2s2_2p3(4So)5z_5Z") == 1  # 2s2 + 2p3 = 3, the 5z contributes none
    assert get_config_parity("2s2_13w_2W") == 0  # 2s2 = 0, the 13w contributes none

    # When the parser reads a bare configuration as a name with a term, it takes the configuration
    # for the term, and no orbitals come back. The sum is then empty, and 0 is a real parity that
    # would be the wrong answer. It is right for '3d7' (2 * 7 = 14) only by coincidence, and wrong
    # for '5p3' (1 * 3 = 3, odd). None says so instead. Callers with a bare configuration pass
    # hasterm=False, below.
    assert interpret_configuration("3d7")[0] == []
    assert get_config_parity("3d7") is None
    assert get_config_parity("5p3") is None


def test_parity_of_a_bare_configuration():
    """A name with no term is all configuration, and adf04 writes its orbitals in upper case."""
    # the whole string is orbitals, so the parser loses nothing off the end and odd reads as odd
    assert get_config_parity("3d7", hasterm=False) == 0  # 2 * 7 = 14
    assert get_config_parity("5s2", hasterm=False) == 0  # 0 * 2
    assert get_config_parity("5p3", hasterm=False) == 1  # 1 * 3
    assert get_config_parity("2p", hasterm=False) == 1  # a lone orbital with no occupation

    # ADAS adf04 configurations: upper case, space separated, with the level's own index in
    # brackets at the end. The strip of a term also removed the last orbital, '4P1', so every level
    # of FeIII.adf04 came out even. 3s2 3p6 3d5 4p1 is 0 + 6 + 10 + 1 = 17, odd.
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
    # the case still matters. '8SNG' is He I's merged singlets, not an 8s orbital. The parent term
    # left over from '3d4(3H)s44p_x3Io' is not a merge marker.
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

    # the parser reads a two-digit principal quantum number as such when the orbital has no
    # occupation number, because then nothing else can own the digits
    assert interpret_configuration("3d6(5D)10d_5Pe") == (["3d6", "(5D)", "10d"], 5, 1, 0, -1)

    # ...and also with an occupation number, when the digits cannot belong to a preceding
    # orbital (start of string, or right after a parent term)
    assert interpret_configuration("10d1_2De") == (["10d1"], 2, 2, 0, -1)
    assert interpret_configuration("3d6(5D)10d1_5Pe") == (["3d6", "(5D)", "10d1"], 5, 1, 0, -1)

    # a digit followed by two letters keeps the digit with the letters. The parser treats '4sp' as
    # an orbital plus an occupation, which is how the code handled this malformed Hillier name before
    assert interpret_configuration("4sp(3P)_7Po[2]") == (["4sp", "(3P)"], 7, 1, 1, -1)

    # the parser always reads an orbital with a SINGLE-digit occupation with a single-digit n.
    # '3d14s2' is really ambiguous, and the occupation-1 read is the common one
    assert interpret_configuration("3d14s2_2De") == (["3d1", "4s2"], 2, 2, 0, -1)

    # ...but trailing digits after the orbital letter are the occupation. Closed d and f shells
    # with TWO-digit occupations are unambiguous and must keep both digits
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
    """CMFGEN cross section type 8 (modified hydrogenic split l).

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

    # the offset must not change the energy grid: it still starts at the true threshold
    assert np.isclose(phixstable[0, 0], threshold_ev / ryd_to_ev, rtol=1e-10)

    # nu_o=None (type 2) must not see the offset and must not require zion
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
            # a hydrogenic level of charge Z and principal quantum number n ionises at Z**2 / n**2 Ryd
            threshold_ev = atomic_number**2 * ryd_to_ev / n**2
            phixstable = rhd.get_hydrogenic_n_phixstable(rhd.hc_in_ev_angstrom / threshold_ev, n)

            # Kramers (1923, Phil. Mag., 46, 836-871, doi:10.1080/14786442308565244):
            # sigma_threshold = 7.91 Mb * n / Z**2 * g_bf, and g_bf at threshold
            # depends only on n. A comparison at the same n therefore leaves a ratio of exactly n / Z**2
            same_n_hydrogen = rhd.get_hydrogenic_n_phixstable(rhd.hc_in_ev_angstrom / (ryd_to_ev / n**2), n)
            assert np.isclose(phixstable[0][1], same_n_hydrogen[0][1] / atomic_number**2, rtol=1e-6)

        # the n=1 threshold cross section must fall exactly as 1 / Z**2
        threshold_ev = atomic_number**2 * ryd_to_ev
        phixstable = rhd.get_hydrogenic_n_phixstable(rhd.hc_in_ev_angstrom / threshold_ev, 1)
        assert np.isclose(phixstable[0][1], sigma_hydrogen_1s / atomic_number**2, rtol=1e-6)


def test_match_hydrogenic_phixs_is_not_double_scaled():
    """match_hydrogenic_phixs() must not rescale the table returned by get_hydrogenic_n_phixstable().

    get_hydrogenic_n_phixstable() already contains the effective-charge scaling. A second
    factor of Z_eff**2 would suppress every cross section (a factor of ~20 for a typical
    E_th = 11 eV, n = 5 valence level).
    """
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
    args = phixs_args(nlevels_hydrogenic_for_unknown_phixs=100)

    crosssections, targetfractions, thresholds = match_hydrogenic_phixs(
        atomic_number=2,
        energy_levels=dflevels,
        ionization_energy_ev=ionization_energy_ev,
        ion_handler="kurucz",
        get_level_valence_n=readkuruczdata.get_level_valence_n,
        args=args,
        flog=io.StringIO(),
    )

    assert thresholds[0] == ionization_energy_ev
    assert targetfractions[0] == [(0, 1.0)]  # the upper ion's ground state

    expected_threshold_mb = rhd.get_hydrogenic_n_phixstable(rhd.hc_in_ev_angstrom / ionization_energy_ev, 1)[0][1]
    assert abs(expected_threshold_mb - 6.3067 / 4) < 1e-3  # exact hydrogenic value for He II 1s
    # the downsampled first point is a bin average, so allow a few percent
    assert abs(crosssections[0][0] / expected_threshold_mb - 1) < 0.05

    # match_hydrogenic_phixs() must skip a level above the ionisation energy, not divide by a negative threshold
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
        get_level_valence_n=readkuruczdata.get_level_valence_n,
        args=args,
        flog=io.StringIO(),
    )
    # NaN means "no threshold energy". The empty target list below makes write_phixs_data() skip the level.
    assert np.isnan(thresholds[0])
    assert targetfractions[0] == []
    assert np.all(crosssections[0] == 0.0)


def test_nlevels_hydrogenic_for_unknown_phixs_caps_the_level_count():
    """-nlevels_hydrogenic_for_unknown_phixs sets how many of the lowest levels get an estimate."""
    rhd.read_hyd_phixsdata()

    ionization_energy_ev = 4 * rhd.ryd_to_ev
    nlevels = 5
    dflevels = pl.DataFrame(
        {
            "levelid": list(range(nlevels)),
            # ascending, and all well below the ionisation energy, so the estimate skips none as unbound
            "energyabovegsinpercm": [i * 1000.0 for i in range(nlevels)],
            "g": [2.0] * nlevels,
            "levelname": ["s1s  1S,enpercm=0.0,j=0.5"] * nlevels,
        }
    )

    for nlevels_option, n_expected in ((0, 0), (2, 2), (nlevels + 10, nlevels)):
        args = phixs_args(nlevels_hydrogenic_for_unknown_phixs=nlevels_option)
        _, targetfractions, thresholds = match_hydrogenic_phixs(
            atomic_number=2,
            energy_levels=dflevels,
            ionization_energy_ev=ionization_energy_ev,
            ion_handler="kurucz",
            get_level_valence_n=readkuruczdata.get_level_valence_n,
            args=args,
            flog=io.StringIO(),
        )
        assert sum(bool(targets) for targets in targetfractions) == n_expected
        assert np.count_nonzero(~np.isnan(thresholds)) == n_expected

    # the lowest levels are the lowest by energy, not the first rows. With two unbound levels in
    # rows 1 and 2, a request for 3 gives the three bound levels in rows 0, 3 and 4
    unbound_percm = 2 * ionization_energy_ev / hc_in_ev_cm
    dflevels_partly_unbound = dflevels.with_columns(
        energyabovegsinpercm=pl.Series([0.0, unbound_percm, unbound_percm, 1000.0, 2000.0])
    )
    args = phixs_args(nlevels_hydrogenic_for_unknown_phixs=3)
    _, targetfractions, thresholds = match_hydrogenic_phixs(
        atomic_number=2,
        energy_levels=dflevels_partly_unbound,
        ionization_energy_ev=ionization_energy_ev,
        ion_handler="kurucz",
        get_level_valence_n=readkuruczdata.get_level_valence_n,
        args=args,
        flog=io.StringIO(),
    )
    assert [bool(targets) for targets in targetfractions] == [True, False, False, True, True]

    # the limit bounds the levels considered, not the tables produced. The estimate skips an unbound
    # level among the lowest by energy, but that level still counts. A request for 4 here therefore
    # gives the same three
    args = phixs_args(nlevels_hydrogenic_for_unknown_phixs=4)
    _, targetfractions, thresholds = match_hydrogenic_phixs(
        atomic_number=2,
        energy_levels=dflevels_partly_unbound,
        ionization_energy_ev=ionization_energy_ev,
        ion_handler="kurucz",
        get_level_valence_n=readkuruczdata.get_level_valence_n,
        args=args,
        flog=io.StringIO(),
    )
    assert np.count_nonzero(~np.isnan(thresholds)) == 3


def test_match_hydrogenic_phixs_takes_the_lowest_levels_by_energy():
    """A reader that keeps its file's order still gets the estimate for its lowest levels by energy."""
    rhd.read_hyd_phixsdata()
    ionization_energy_ev = 4 * rhd.ryd_to_ev
    # row 0 is the highest level and row 1 the ground state
    dflevels = pl.DataFrame(
        {
            "levelid": [0, 1, 2],
            "energyabovegsinpercm": [5000.0, 0.0, 2500.0],
            "g": [2.0, 2.0, 2.0],
            "levelname": ["s1s  1S,enpercm=0.0,j=0.5"] * 3,
        }
    )
    _, targetfractions, thresholds = match_hydrogenic_phixs(
        atomic_number=2,
        energy_levels=dflevels,
        ionization_energy_ev=ionization_energy_ev,
        ion_handler="kurucz",
        get_level_valence_n=readkuruczdata.get_level_valence_n,
        args=phixs_args(nlevels_hydrogenic_for_unknown_phixs=2),
        flog=io.StringIO(),
    )
    assert [bool(targets) for targets in targetfractions] == [False, True, True]
    # each threshold belongs to its own level id, not to its position in the sorted order
    assert thresholds[1] == ionization_energy_ev
    assert thresholds[2] == pytest.approx(ionization_energy_ev - hc_in_ev_cm * 2500.0)
    assert np.isnan(thresholds[0])


def test_write_phixs_data_with_no_phixs_arrays():
    """A reader that found no photoionisation data must not make write_phixs_data() index off the end.

    resolve_photoion_targetfractions() fills a target list for every level whenever the reader supplied none, and
    readhillierdata.get_photoiontargetfractions() gives at least the ground state to every level
    that has a target configuration list. If the
    reader also left the cross section and threshold arrays empty, the level ids from those target
    lists have nothing behind them.
    """
    args = phixs_args()
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
    assert "artisatomic writes 0 cross section tables to phixsdata_v2.txt." in flog.getvalue()


def make_iondata(ion_stage, is_top_ion, targetfractions=None, targetconfigs=None):
    """Build a minimal single-level IonData, with the source line that each comment block needs."""
    from artisatomic.base import COMMENT_TABLES
    from artisatomic.iondata import IonData

    return IonData(
        comments={table: [f"source: the source of the {table} test data"] for table in COMMENT_TABLES},
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
        upsilondict={},
        photoion_targetconfigs=targetconfigs,
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
    # the top ion has no upper ion to photoionise to, so the resolver leaves it as the reader gave it
    assert upper.photoionization_targetfractions == []


def test_resolve_photoion_targetfractions_keeps_reader_supplied():
    """An ion whose reader already gave per-level fractions (e.g. the hydrogenic estimate) keeps them."""
    from artisatomic.iondata import resolve_photoion_targetfractions

    # a target list the Hillier resolver would never produce, so an overwrite would be visible
    supplied = [[(7, 1.0)]]
    lower = make_iondata(1, is_top_ion=False, targetfractions=supplied, targetconfigs=[[("gs2", 1.0)]])
    resolve_photoion_targetfractions([lower, make_iondata(2, is_top_ion=True)])

    assert lower.photoionization_targetfractions == supplied


def test_resolve_photoion_targetfractions_rejects_a_half_given_log_pair():
    """A caller must give both atomic_number and logpath, or neither, so a log is not lost silently."""
    from artisatomic.iondata import resolve_photoion_targetfractions

    lower = make_iondata(1, is_top_ion=False, targetconfigs=[[("gs2", 1.0)]])
    upper = make_iondata(2, is_top_ion=True)

    with pytest.raises(ValueError, match="both atomic_number and logpath"):
        resolve_photoion_targetfractions([lower, upper], 8)


def test_resolve_photoion_targetfractions_rejects_misordered_ions():
    """The resolver rejects a list that is not one element's ions in ascending order.

    The resolver matches each ion against the next entry as its upper ion. A top ion anywhere but
    last therefore means that the matched levels belong to the wrong ion.
    """
    from artisatomic.iondata import resolve_photoion_targetfractions

    with pytest.raises(ValueError, match="ascending ion stage order"):
        resolve_photoion_targetfractions([make_iondata(1, is_top_ion=True), make_iondata(2, is_top_ion=True)])

    # a list whose last ion is not the top ion lacks the upper ion that the last entry needs
    with pytest.raises(ValueError, match="ascending ion stage order"):
        resolve_photoion_targetfractions([make_iondata(1, is_top_ion=False), make_iondata(2, is_top_ion=False)])


def test_write_output_files_rejects_unresolved_targetfractions(tmp_path):
    """The writer must reject an ion that still needs the resolver, not drop its cross sections.

    write_output_files() no longer resolves target fractions itself. For an ion with cross sections
    but no targets, write_phixs_data() would silently skip every one of its tables.
    """
    from artisatomic.output import write_output_files

    tmpargs = phixs_args(output_folder=str(tmp_path))
    lower = make_iondata(1, is_top_ion=False)
    lower.photoionization_crosssections = np.zeros((1, 100))

    with pytest.raises(ValueError, match="call resolve_photoion_targetfractions"):
        write_output_files(26, [lower, make_iondata(2, is_top_ion=True)], tmpargs)


def read_phixs_tables_of_one_level(monkeypatch, tables: list[np.ndarray], args) -> tuple[PhixsData, str]:
    """Run read_phixs_tables() on one level with one synthetic raw table for each route.

    The fake read_file() stores the tables, so no file is read. The ion gets one phot file for
    each table. The result is the PhixsData record and the log text.
    """
    ionfiles = readhillierdata.ions_data[8, 1]
    photfilenames = tuple(f"phot_data_{chr(ord('A') + filenum)}" for filenum in range(len(tables)))
    monkeypatch.setitem(readhillierdata.ions_data, (8, 1), ionfiles._replace(photfilenames=photfilenames))

    def read_file(reader, filenum, _filename, _photfilename):
        reader.phixstables[filenum] = {"ground": tables[filenum]}
        reader.phixstargets[filenum] = f"target{filenum}"

    monkeypatch.setattr(rhd.PhotFileReader, "read_file", read_file)
    monkeypatch.setattr(rhd, "read_hyd_phixsdata", lambda: None)
    levels = pl.DataFrame({"levelname": ["ground"], "lambdaangstrom": [hc_in_ev_angstrom / ryd_to_ev]})
    flog = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()):
        result = rhd.read_phixs_tables(8, 1, levels, args, flog)
    return result, flog.getvalue()


@pytest.mark.parametrize("threshold_ratio", [0.5, 2.0])
@pytest.mark.parametrize("amplitudes", [(1.0, 1.0), (1.0, 3.0), (3.0, 1.0)])
@pytest.mark.parametrize("offset", [False, True])
def test_photoion_target_fractions_preserve_normalised_amplitudes(monkeypatch, threshold_ratio, amplitudes, offset):
    """Targets with the same normalised shape keep their amplitude ratio at different thresholds.

    With offset, both routes are zero below twice their first energy. The weights of the output
    bins depend on the absolute frequency, so the ratio holds to one percent and not exactly.
    """
    args = phixs_args()
    u = np.linspace(1.0, 6.0, 501)
    shape = u**-3
    if offset:
        shape[u < 2.0] = 0.0
    tables = [
        np.column_stack((u * threshold, amplitude * shape))
        for threshold, amplitude in zip((1.0, threshold_ratio), amplitudes, strict=True)
    ]

    result, _log = read_phixs_tables_of_one_level(monkeypatch, tables, args)
    assert result.targetconfigs is not None
    targets = result.targetconfigs[0]
    assert targets is not None
    assert [name for name, _ in targets] == ["target0", "target1"]
    fractions = [fraction for _, fraction in targets]
    assert fractions == pytest.approx(np.array(amplitudes) / sum(amplitudes), rel=1e-2)

    # the fraction of a target is the sum of the reduced table of its route over the total.
    # The shared table is the sum of the reduced tables of the two routes.
    xgrid = output_xgrid(args.nphixspoints, args.phixsnuincrement)
    reduced = [reduce_phixs_tables_worker(args.optimaltemperature, xgrid, table) for table in tables]
    sums = [table.sum() for table in reduced]
    assert fractions == pytest.approx([tablesum / sum(sums) for tablesum in sums])
    np.testing.assert_allclose(result.crosssections[0], reduced[0] + reduced[1])


def test_photoion_target_fractions_offset_route(monkeypatch):
    """A route with an offset edge gets the fraction of its sum, and the shared table stays open.

    Both routes fall as u^-3. Route 1 is zero below u = 2. The output grid runs from u = 1 to
    u = 4, so the integrals are 15/32 and 3/32. The shared table is the sum of the two reduced
    tables, so it is open at u = 1. Before this rule, a comparison at the open edge of route 1
    read route 0 in its tail, and the table of route 1, zero below u = 2, went to both targets.
    """
    args = phixs_args()
    assert output_xgrid(args.nphixspoints, args.phixsnuincrement)[-1] == pytest.approx(4.0)
    u = np.linspace(1.0, 6.0, 5001)
    shape = u**-3
    tables = [np.column_stack((u, shape)), np.column_stack((2.0 * u, np.where(u < 2.0, 0.0, shape)))]

    result, log = read_phixs_tables_of_one_level(monkeypatch, tables, args)
    assert result.targetconfigs is not None
    targets = result.targetconfigs[0]
    assert targets is not None
    assert [name for name, _ in targets] == ["target0", "target1"]
    assert [fraction for _, fraction in targets] == pytest.approx([15 / 18, 3 / 18], rel=5e-2)
    assert result.crosssections[0][0] > 0.0
    assert "in 2 photoionisation files" in log
    assert "target0:" in log
    assert "target1:" in log


def test_photoion_target_fraction_at_repeated_open_edge(monkeypatch):
    """A route whose open edge repeats an energy after a zero row keeps its share of the total.

    C II 2s2_6s_2Se has such a tabulated table. A factor from one point at the edge is fragile
    there. A product that rounds to just below the edge reads the zero row through np.interp().
    """
    args = phixs_args()
    edge = 0.19199342756764298
    table = np.array([[0.14034607278336475, 0.0], [edge, 0.0], [edge, 1.372], [1.0, 0.1]])
    u = np.linspace(1.0, 6.0, 501)
    tables = [table, np.column_stack((u, u**-3))]
    result, log = read_phixs_tables_of_one_level(monkeypatch, tables, args)
    assert result.targetconfigs is not None
    targets = result.targetconfigs[0]
    assert targets is not None
    xgrid = output_xgrid(args.nphixspoints, args.phixsnuincrement)
    sums = [reduce_phixs_tables_worker(args.optimaltemperature, xgrid, table).sum() for table in tables]
    assert sums[0] > 0.0
    assert [fraction for _, fraction in targets] == pytest.approx([tablesum / sum(sums) for tablesum in sums])
    assert "so it will have no phixs" not in log


def test_photoion_target_below_cut_drops_with_its_route(monkeypatch):
    """A target below 2% of the total leaves the target list, and its route leaves the table."""
    args = phixs_args()
    u = np.linspace(1.0, 6.0, 501)
    tables = [np.column_stack((u, u**-3)), np.column_stack((u, 0.005 * u**-3))]
    result, log = read_phixs_tables_of_one_level(monkeypatch, tables, args)
    assert result.targetconfigs is not None
    assert result.targetconfigs[0] == [("target0", 1.0)]
    xgrid = output_xgrid(args.nphixspoints, args.phixsnuincrement)
    np.testing.assert_allclose(
        result.crosssections[0], reduce_phixs_tables_worker(args.optimaltemperature, xgrid, tables[0])
    )
    assert "The target target1 has less than 2% of the total, so the output drops it." in log


def test_read_phixs_tables_multiple_photoionisation_files():
    """The O I target fractions come from the sums of the reduced tables of the two routes.

    The mirror below reduces the raw tables itself and calls no helper of the reader. A change
    to the reader cannot pass unseen. The test checks the fractions and the shared table of every
    level with two routes, and it pins the fractions of two levels as literals. Both files go
    through one reader, so the second route keeps its excitation energy.
    """
    from artisatomic.phixs import reduce_phixs_tables

    ionfiles = readhillierdata.ions_data[8, 1]
    # checked before the expensive reads below, so a change here fails as itself
    assert len(ionfiles.photfilenames) == 2, f"O I is expected to have two phot files, got {ionfiles.photfilenames}"

    rhd.read_hyd_phixsdata()
    args = phixs_args()

    flog = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()):
        _, dflevels, _ = rhd.read_levels_and_transitions(8, 1, flog)
        phixs = rhd.read_phixs_tables(8, 1, dflevels, args, flog)
    crosssections, targetconfigs = phixs.crosssections, phixs.targetconfigs
    assert targetconfigs is not None
    log = flog.getvalue()

    # the reader reports the duplicates and does not raise
    assert "has a cross section table in 2 photoionisation files" in log

    # every level that the reader read has a table, not only the ones from the first file
    assert all(np.any(crosssections[levelid]) for levelid, targets in enumerate(targetconfigs) if targets)

    # the routes of every level, read the way read_phixs_tables() reads them
    levelnames = dflevels["levelname"].to_list()
    firstindex_of_name: dict[str, int] = {}
    firstindex_of_namenoj: dict[str, int] = {}
    for levelindex, levelname in enumerate(levelnames):
        firstindex_of_name.setdefault(levelname, levelindex)
        firstindex_of_namenoj.setdefault(levelname.split("[")[0], levelindex)
    reader = rhd.PhotFileReader(
        8, 1, 2, dflevels["lambdaangstrom"].to_list(), firstindex_of_name, firstindex_of_namenoj, io.StringIO()
    )
    reduced_of_filenum = []
    for filenum, photfilename in enumerate(ionfiles.photfilenames):
        photpath = Path(rhd.hillier_ion_folder(8, 1), ionfiles.folder, photfilename)
        with contextlib.redirect_stdout(io.StringIO()):
            reader.read_file(filenum, photpath, photfilename)
        reduced_of_filenum.append(
            reduce_phixs_tables(
                reader.phixstables[filenum], args.optimaltemperature, args.nphixspoints, args.phixsnuincrement
            )
        )

    def expected_routes(matchname: str) -> list[tuple[str, float, np.ndarray]]:
        """Give the target, the factor and the reduced table of each open route of a level.

        The factor is the sum of the reduced table. A route with no reduced table or with an
        all-zero reduced table drops out here, as it does in the reader.
        """
        factors = []
        for filenum, reduced in enumerate(reduced_of_filenum):
            reducedtable = reduced.get(matchname)
            if reducedtable is None or not np.any(reducedtable):
                continue
            factors.append((reader.phixstargets[filenum], float(reducedtable.sum()), reducedtable))
        return factors

    matchname_of_levelid = [name if reader.j_splitting_on else name.split("[")[0] for name in levelnames]
    levelids_in_both = [
        levelid
        for levelid, matchname in enumerate(matchname_of_levelid)
        if all(np.any(reduced.get(matchname, np.zeros(1))) for reduced in reduced_of_filenum)
    ]
    assert levelids_in_both, "expected O I levels with a cross section table in both phot files"

    for levelid in levelids_in_both:
        factors = expected_routes(matchname_of_levelid[levelid])
        assert len(factors) > 1, f"level {levelid}: expected a route in each of the two files"
        # the targets below the 2% cut drop out, and the rest normalise to one. The strongest
        # route always stays.
        factor_sum_nofilter = sum(value for _, value, _ in factors)
        largest = max(value for _, value, _ in factors)
        keptroutes = [
            (target, value, reducedtable)
            for target, value, reducedtable in factors
            if value == largest or value / factor_sum_nofilter > PHIXS_TARGET_FRACTION_CUT
        ]
        keptfactors = [(target, value) for target, value, _ in keptroutes]
        factor_sum = sum(value for _, value in keptfactors)
        expected_fractions = [(target, value / factor_sum) for target, value in keptfactors]

        targetlist = targetconfigs[levelid]
        assert targetlist is not None
        assert [target for target, _ in targetlist] == [target for target, _ in expected_fractions]
        assert [fraction for _, fraction in targetlist] == pytest.approx(
            [fraction for _, fraction in expected_fractions]
        )

        # the shared table is the sum of the reduced tables of the kept routes, after the 2% cut
        assert np.allclose(crosssections[levelid], sum(reducedtable for _, _, reducedtable in keptroutes), rtol=1e-10)

    # The fractions of two O I levels, as literals. The mirror above and the reader share the
    # raw tables. A literal is therefore the only check that a change to a shared helper cannot
    # move.
    # The two files give the 1Do level one cross section as a function of the ratio, scaled to
    # two edges. Its fractions differ from one half only through the bin weights. The 3Do level
    # has two different tables, so its fractions pin the rule.
    targetlist_1do = targetconfigs[levelnames.index("2s2_2p3(2Do)3s_1Do[2]")]
    assert targetlist_1do is not None
    assert [target for target, _ in targetlist_1do] == ["2s2_2p3_4So/2p^3_4So", "2s2_2p3_2Do"]
    assert [fraction for _, fraction in targetlist_1do] == pytest.approx([0.5, 0.5], abs=5e-4)
    targetlist_3do = targetconfigs[levelnames.index("2s2_2p3(2Do)3s_3Do[3]")]
    assert targetlist_3do is not None
    assert [target for target, _ in targetlist_3do] == ["2s2_2p3_4So/2p^3_4So", "2s2_2p3_2Do"]
    assert [fraction for _, fraction in targetlist_3do] == pytest.approx([0.9675, 0.0325], abs=5e-5)


def test_read_coldata_term_to_j_redistribution():
    """The reader must share a term-resolved effective collision strength over the J levels of BOTH terms.

    ARTIS forms the collisional excitation rate coefficient as proportional to upsilon_ij / g_i,
    so the invariant that makes the total term-to-term rate correct is

        sum_i sum_j upsilon_ij == upsilon_term,   upsilon_ij = upsilon_term * g_i/g_L * g_j/g_U

    O III has term-resolved collision data (col_data in OXY/III/19apr23) and a J-split level
    list, so it exercises the redistribution. Fe II names its collision transitions with J
    values, so its values must pass through unchanged.
    """
    import argparse
    import contextlib
    from collections import defaultdict

    args = argparse.Namespace(electrontemperature=5000)

    def read_ion(atomic_number, ion_stage):
        flog = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()):
            _, dflevels, _ = readhillierdata.read_levels_and_transitions(atomic_number, ion_stage, flog)
            upsilondict = readhillierdata.read_coldata(atomic_number, ion_stage, dflevels, args, flog)
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

    # and the reader splits it over the lower levels in proportion to g_i (1 : 3 : 5 out of g_L = 9)
    for g_lower, total in zip([1.0, 3.0, 5.0], sums_from_lower, strict=True):
        assert abs(total - upsilon_term * g_lower / 9.0) < 1e-3

    # Fe II collision data is already J-resolved, so every value passes through unscaled. The
    # first row of col_data, a6De[9/2] -> a6De[7/2], gives 3.230 in the T = 0.5e4 K column
    _, upsilondict_fe2, _ = read_ion(26, 2)
    assert sum(1 for v in upsilondict_fe2.values() if v > 0.0) == 10601
    assert upsilondict_fe2[0, 1] == pytest.approx(3.23)


def test_add_level_ids_forbidden_rejects_an_unknown_level_id():
    """A transition whose level id names no level must fail, not vanish in the inner join."""
    from artisatomic.output import add_level_ids_forbidden

    dflevels = pl.DataFrame({"levelid": [0, 1, 2], "levelname": ["a", "b", "c"], "parity": [0, 1, 0]})
    dftransitions = pl.DataFrame({"lowerlevel": [0, 0], "upperlevel": [1, 5], "A": [1.0, 1.0]})

    with pytest.raises(ValueError, match="1 transitions name a level id that is not one of the 3 levels"):
        add_level_ids_forbidden(dflevels, dftransitions)


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

    # even -> odd is permitted. A real parity never matches an absent one. Two absent ones do not
    # match each other either, which is the reason that null spells absence. Two real even ones
    # do match
    assert forbidden == [False, False, False, False, True]


def test_get_level_j():
    """get_level_j() reads J from the brackets at the end of a J-resolved CMFGEN level name."""
    get_level_j = readhillierdata.get_level_j

    assert get_level_j("3d6_a5De[4]") == 4.0
    assert get_level_j("3d5(4D)4po[9/2]") == 4.5
    assert get_level_j("3d4(3P2)4po[1/2]") == 0.5

    # a name with a brace and nothing after it gives J there; in pair coupling the brace holds
    # K and the trailing bracket is J. The last group is the one read, and both come out right.
    assert get_level_j("2s2_2p(2P<1/2>)4f_2{5/2}e") == 2.5
    assert get_level_j("2p5(2P*<1/2>)3d_2{3/2}o[1]") == 1.0

    # Not every trailing bracket is a J. Si X and S X number their levels there instead. g separates
    # the two cases, because J is only J where g == 2J + 1. A 3P term has no J = 3.
    assert get_level_j("2p3p3Pe[3]", g=5.0) is None
    assert get_level_j("2s2_2p3_4So[2]", g=4.0) is None
    assert get_level_j("3d6_a5De[4]", g=9.0) == 4.0

    # A term-resolved level has no J of its own, because its g counts every J of the term. A merged
    # level has no J either. The parser must not give either of them one.
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
    nominal 0.8 cm-1 and then shares the term's f over all the J pairs. It therefore lists dJ = 2
    lines with f as large as 0.116. To call those forbidden would give a strong line the forbidden
    collision approximation.
    """
    dflevels = pl.DataFrame(
        {"levelid": [0, 1], "parity": [1, 0], "j": [0.5, 2.5]},
        schema={"levelid": pl.Int64, "parity": pl.Int64, "j": pl.Float64},
    )
    dftransitions = pl.DataFrame({"lowerlevel": [0], "upperlevel": [1], "A": [3.4e9], "f": [0.116]})

    result = add_level_ids_forbidden(dflevels, dftransitions)

    # dJ = 2 breaks the rule, which the column records, but the f keeps the transition permitted
    assert result["breaksdeltaj"].to_list() == [True]
    assert result["forbidden"].to_list() == [False]

    # with no f and no A, the same pair comes out forbidden
    nof = dftransitions.with_columns(A=0.0).drop("f")
    assert add_level_ids_forbidden(dflevels, nof)["forbidden"].to_list() == [True]

    # A weak f is not evidence of an E1 line, so the J labels decide it. C IV lists
    # 2p_2Po[1/2] -> 3d_2De[5/2] at f = 2.7e-10. Its 108 cm-1 fine structure means that its J
    # labels are sound, so that line really is forbidden.
    weakf = dftransitions.with_columns(A=4.1, f=2.7e-10)
    assert add_level_ids_forbidden(dflevels, weakf)["forbidden"].to_list() == [True]


def test_log_deltaj_contradictions_judges_f_and_a_separately():
    """Only a transition strong enough to be E1 contradicts its own J labels.

    f has no units and an E1 line carries 1e-3 to 1. A is a rate in s-1 over many decades, and a
    forbidden line still reaches ~1e2. The same cut would therefore report every forbidden line
    with a reader-supplied A as a contradiction.
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

    # A, where a forbidden line reaches 14 s-1 in the QUB Co III data and must stay quiet
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

    Three readers used to spell "no parity" as the negated level id. That made level 0 come out as
    a real even parity, while no other level could match. Nothing may depend on negative numbers as
    a mark of absence any more. Two levels that share one are forbidden, like any other pair.
    """
    dflevels = pl.DataFrame({"levelid": [0, 1], "parity": [-7, -7]}, schema={"levelid": pl.Int64, "parity": pl.Int64})
    dftransitions = pl.DataFrame({"lowerlevel": [0], "upperlevel": [1], "A": [1.0]})

    assert add_level_ids_forbidden(dflevels, dftransitions)["forbidden"].to_list() == [True]


@pytest.mark.parametrize(
    ("parity", "dtype"),
    [
        ([None, None], pl.Int64),  # the canonical spelling of an absent parity
        ([float("nan"), float("nan")], pl.Float64),  # NaN compares equal to itself, so it must cast away
        (["1", "1"], pl.String),  # a reader that gives text where a whole number belongs
    ],
)
def test_add_level_ids_forbidden_unreadable_parity(parity, dtype):
    """A parity that is not a whole number means the level has none, not that it matches itself."""
    dflevels = pl.DataFrame({"levelid": [0, 1], "parity": parity}, schema={"levelid": pl.Int64, "parity": dtype})
    dftransitions = pl.DataFrame({"lowerlevel": [0], "upperlevel": [1], "A": [1.0]})

    forbidden = add_level_ids_forbidden(dflevels, dftransitions)["forbidden"].to_list()

    # a readable string parity is a real parity and still counts; the rest are unknown
    assert forbidden == [dtype == pl.String]


def test_readhillierdata_hydrogen_lyman_alpha_is_permitted():
    """H I is merged n-levels throughout, so nothing in it may come out forbidden.

    Every H I level has the name '<n>___' and holds every l of that n, so it has no definite parity.
    When the code treated those as a matching parity, all 435 H I transitions came out forbidden,
    Lyman alpha included. That is the strongest permitted line there is, and hi_osc.dat lists it
    with f = 0.4162.
    """
    _, dflevels, dftransitions = readhillierdata.read_levels_and_transitions(1, 1, io.StringIO())

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

    add_level_ids_forbidden() marks a transition forbidden when its two levels share a parity. The
    same parity on every level therefore made every transition of the ion forbidden, and helium
    has many permitted ones. A null parity cannot match another, which is how readlisbondata and
    readkuruczdata say the same thing.

    The repository does not hold aoife.hdf5. The .gitignore of atomic-data-helium-boyle excludes
    every .hdf5 file, and its README.txt gives the download link. This test therefore builds the
    three tables that the reader wants in memory.
    """
    import h5py

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
            (2, 1, 0, 0.0, 2.0, 0),  # He II, which the reader must filter out
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
        transitions = readboyledata.read_lines_data(2, 1)

    # the reader filters out the other ion's level
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

    # the file's level numbers are the level ids, so the writer's count lands on the right level
    assert transition_count_of_level(dftransitions, len(energy_levels))[2] == 2


def write_pandas_hdfstore(path, columns, indexlevels):
    """Write a pandas HDFStore of the "fixed" format, as DataFrame.to_hdf() writes one.

    The test writes the file itself, because the DREAM line list is not part of this repository
    and pandas is not a dependency. columns maps each column name to its values, and indexlevels
    maps each index level name to its values.
    """
    import h5py

    with h5py.File(path, "w") as h5file:
        group = h5file.create_group("atomic_data")
        group.attrs["pandas_type"] = b"frame"
        group.attrs["axis1_nlevels"] = len(indexlevels)
        group.attrs["nblocks"] = 2

        def write_pickle(name, obj):
            # pytables holds a pickled object as a variable-length array of bytes
            dataset = group.create_dataset(name, (1,), dtype=h5py.vlen_dtype(np.uint8))
            dataset[0] = np.frombuffer(pickle.dumps(obj), dtype=np.uint8)
            return dataset

        for level, (name, values) in enumerate(indexlevels.items()):
            uniquevalues = list(dict.fromkeys(values))
            dataset = write_pickle(f"axis1_level{level}", np.array(uniquevalues, dtype=object))
            dataset.attrs["name"] = name.encode()
            group.create_dataset(
                f"axis1_label{level}", data=np.array([uniquevalues.index(v) for v in values], dtype=np.int8)
            )

        # block 0 holds the float columns, block 1 the columns of Python objects
        names = list(columns)
        nrows = len(next(iter(columns.values())))
        floatnames = [name for name in names if all(isinstance(v, float) for v in columns[name])]
        objectnames = [name for name in names if name not in floatnames]
        group.create_dataset("block0_items", data=np.array([n.encode() for n in floatnames]))
        group.create_dataset(
            "block0_values", data=np.array([[columns[n][row] for n in floatnames] for row in range(nrows)])
        )
        group.create_dataset("block1_items", data=np.array([n.encode() for n in objectnames]))
        write_pickle(
            "block1_values",
            np.array([[columns[n][row] for n in objectnames] for row in range(nrows)], dtype=object),
        )
        group.create_dataset("axis0", data=np.array([n.encode() for n in names]))


def test_read_pandas_hdfstore_rebuilds_the_frame(tmp_path):
    """The DREAM reader rebuilds a pandas HDFStore with h5py, and needs no pandas to do it.

    The index levels come first, then the columns in the order of the frame. The reader matches
    the values to the columns by name, so the two blocks may hold them in any order.
    """
    from artisatomic.readdreamdata import read_pandas_hdfstore

    path = tmp_path / "store.h5"
    write_pandas_hdfstore(
        path,
        columns={
            "Wavelength": [3175.982, 3215.81],
            "Lower_Level": [0, 1053],
            "Lower_Type": ["(e)", "(o)"],
            "Lower_J": [1.5, 2.5],
            # one column of two types keeps the Object type, as CF does in the DREAM line list
            "CF": [0.047, "n"],
        },
        indexlevels={"Z": [57, 57], "C": [0, 1]},
    )

    df = read_pandas_hdfstore(path)
    assert df.columns == ["Z", "C", "Wavelength", "Lower_Level", "Lower_Type", "Lower_J", "CF"]
    # a column of Python integers becomes Int64, so the polars expressions can read it
    assert df.schema["Z"] == pl.Int64
    assert df.schema["Lower_Level"] == pl.Int64
    assert df.schema["CF"] == pl.Object
    assert df.select("Z", "C", "Wavelength", "Lower_Level", "Lower_Type", "Lower_J").rows() == [
        (57, 0, 3175.982, 0, "(e)", 1.5),
        (57, 1, 3215.81, 1053, "(o)", 2.5),
    ]
    assert df["CF"].to_list() == [0.047, "n"]


def test_read_pandas_hdfstore_rejects_another_format(tmp_path):
    """A file that is not a frame of the "fixed" format names the format that the reader needs."""
    import h5py

    from artisatomic.readdreamdata import read_pandas_hdfstore

    path = tmp_path / "table.h5"
    with h5py.File(path, "w") as h5file:
        h5file.create_group("atomic_data").attrs["pandas_type"] = b"frame_table"
    with pytest.raises(ValueError, match="format='fixed'"):
        read_pandas_hdfstore(path)

    path = tmp_path / "twoframes.h5"
    with h5py.File(path, "w") as h5file:
        h5file.create_group("first")
        h5file.create_group("second")
    with pytest.raises(ValueError, match="expects exactly one frame"):
        read_pandas_hdfstore(path)


def test_readlisbondata_reads_the_levels_and_lines_csv(tmp_path):
    """The reader reads the two CSV files of one ion, past the eight lines of provenance.

    The reader derives J from g, keeps the file index of every level, and keeps the wavelength
    in Angstrom for the gf-to-A constant.
    """
    from artisatomic import readlisbondata

    (tmp_path / "levels.csv").write_text(
        lisbon_provenance("Number Levels", 2) + ",Energy[cm^-1],g,RelConfig\n0,0.0,1,4f(2)0\n1,1000.0,5,4f(2)4\n"
    )
    (tmp_path / "lines.csv").write_text(
        lisbon_provenance("Number Transitions", 1) + ",Lower,Upper,gf,Wavelength[Ang]\n0,0,1,0.25,10000.0\n"
    )

    dflevels = readlisbondata.read_levels_csv(tmp_path / "levels.csv")
    dflines = readlisbondata.read_lines_csv(tmp_path / "lines.csv")

    assert dflevels.select("fileindex", "energy", "j", "label").rows() == [
        (0, 0.0, 0.0, "4f(2)0"),
        (1, 1000.0, 2.0, "4f(2)4"),
    ]
    assert dflines.select("level_index_lower", "level_index_upper", "gf", "wavelength").rows() == [
        (0, 1, 0.25, 10000.0)
    ]

    # the levels are already in energy order here, so every level keeps its file index
    energy_levels, levelid_of_fileindex = readlisbondata.read_levels_data(dflevels)
    assert levelid_of_fileindex == {0: 0, 1: 1}
    assert [level.levelname for level in energy_levels] == ["4f(2)0, j=0.0, index=0", "4f(2)4, j=2.0, index=1"]
    assert [level.g for level in energy_levels] == [1.0, 5.0]

    transitions = readlisbondata.read_lines_data(energy_levels, dflines, levelid_of_fileindex)
    assert [(tr.lowerlevel, tr.upperlevel) for tr in transitions] == [(0, 1)]
    assert pytest.approx(0.25 / (gf_to_a_coefficient * 5.0 * 10000.0**2)) == transitions[0].A


def test_readlisbondata_maps_file_indices_to_energy_sorted_ids():
    """Lisbon lines name their levels by position in the levels file, which the reader re-sorts by energy.

    An index into the sorted list with the file's own position attaches every transition to the
    wrong pair of levels. That happens whenever the source CSV is not already in energy order. The
    map that read_levels_data() returns (as in readfacdata) prevents it. Row position keys the
    map, which matches the way the transitions file names its levels.
    """
    from artisatomic import readlisbondata

    # deliberately not in energy order: file index 0 is the HIGHEST level, 2 the ground state
    dflevels = pl.DataFrame(
        {
            "fileindex": [0, 1, 2],
            "energy": [5000.0, 1000.0, 0.0],
            "j": [2.0, 1.0, 0.0],
            "label": ["top", "mid", "gs"],
        }
    )

    energy_levels, levelid_of_fileindex = readlisbondata.read_levels_data(dflevels)

    # levels come back in ascending energy, so the file's order is exactly reversed
    assert [level.energyabovegsinpercm for level in energy_levels] == [0.0, 1000.0, 5000.0]
    assert levelid_of_fileindex == {2: 0, 1: 1, 0: 2}
    # a null parity never matches another, so the Laporte rule cannot fire...
    assert all(level.parity is None for level in energy_levels)
    # ...and J is what the delta J rule has to go on, so it must reach the level tuple
    assert [level.j for level in energy_levels] == [0.0, 1.0, 2.0]

    # one line from the file's level 2 (the ground state) to its level 0 (the top level). The second
    # row is the same line with the file's labels in the reverse order
    dflines = pl.DataFrame(
        {
            "level_index_lower": [2, 0],
            "level_index_upper": [0, 2],
            "gf": [1.0, 1.0],
            "wavelength": [2000.0, 2000.0],
        }
    )
    transitions = readlisbondata.read_lines_data(energy_levels, dflines, levelid_of_fileindex)

    # ...which is level id 0 -> 2 after the sort, written with the lower id first, both times
    assert len(transitions) == 2
    assert [(transition.lowerlevel, transition.upperlevel) for transition in transitions] == [(0, 2), (0, 2)]
    # A uses the g of the level that ended up as the upper one (J=2, g=5), whichever the file
    # called "Upper"
    expected_a = 1.0 / (gf_to_a_coefficient * 5.0 * 2000.0**2)
    assert all(pytest.approx(expected_a) == transition.A for transition in transitions)
    dftransitions = pl.DataFrame(
        {"lowerlevel": [t.lowerlevel for t in transitions], "upperlevel": [t.upperlevel for t in transitions]}
    )
    assert transition_count_of_level(dftransitions, len(energy_levels)) == [2, 0, 2]
    # the names carry the file index, so two levels with one label and J stay apart
    assert [level.levelname for level in energy_levels] == [
        "gs, j=0.0, index=2",
        "mid, j=1.0, index=1",
        "top, j=2.0, index=0",
    ]

    # a line that names a level the table does not have means that the two files disagree about
    # the numbering. To skip it would drop every transition and write a silently empty ion.
    dflines_unknown = pl.DataFrame(
        {"level_index_lower": [2], "level_index_upper": [99], "gf": [1.0], "wavelength": [2000.0]}
    )
    with pytest.raises(ValueError, match="names file index 99"):
        readlisbondata.read_lines_data(energy_levels, dflines_unknown, levelid_of_fileindex)


def lisbon_provenance(countkey: str, rowcount: int) -> str:
    """Return the eight provenance lines of a Lisbon CSV, with the row count that its header gives.

    The real files carry "Number Levels: N" in a levels file and "Number Transitions: N" in a
    transitions file, on the seventh of the eight lines. The reader checks the count. The lines
    above it differ between the two file types, and the reader reads none of them.
    """
    islevels = countkey == "Number Levels"
    lines = [
        f"                {'Levels' if islevels else 'Transitons'} NdIII                 ",
        "*" * 71,
        "Last updated: 2021-06-13 ",
        *([] if islevels else ["Multipole:-1  #negative values for electic transtions, positive for magnetic"]),
        "Z: 60",
        "Ionization Stage: 2",
        *(["Ground State Energy[eV]: -261525.855"] if islevels else []),
        f"{countkey}: {rowcount}",
        "*" * 71,
    ]
    return "\n".join(lines) + "\n"


def write_lisbon_fixture(tmp_path, energies_percm):
    """Write the levels and transitions CSV of Nd III, with the given level energies in cm^-1.

    Every level gets g = 3, and every pair of adjacent levels gets one line. The eight rows of
    provenance text match the real files.
    """
    iondir = tmp_path / "Nd" / "NdIII"
    iondir.mkdir(parents=True)
    levelrows = "".join(f"{i},4f4,4f-3(9){i},3,0.0,{energy}\n" for i, energy in enumerate(energies_percm))
    (iondir / "NdIII_Levels.csv").write_text(
        lisbon_provenance("Number Levels", len(energies_percm))
        + ",Config,RelConfig,g,Energy[eV],Energy[cm^-1]\n"
        + levelrows
    )

    linerows = "".join(f"{i + 1},{i},0.1,800.0,12500.0,0.5,1.0\n" for i in range(len(energies_percm) - 1))
    (iondir / "NdIII_Transitions.csv").write_text(
        lisbon_provenance("Number Transitions", len(energies_percm) - 1)
        + "Upper,Lower,DeltaE[eV],DeltaE[cm^-1],Wavelength[Ang],gf,TR_rate[1/s]\n"
        + linerows
    )


def test_readlisbondata_drops_the_levels_above_the_ionisation_energy(tmp_path, monkeypatch):
    """The reader drops a level above the NIST ionisation energy, as the FAC reader does.

    A level above the ionisation energy is not a bound level of the ion. The level ids stay
    contiguous after the drop, and every line that names a dropped level goes with it.
    """
    from artisatomic import readlisbondata

    # Nd III ionises at 22.09 eV, which is 178168 cm^-1. The last two levels are above it
    write_lisbon_fixture(tmp_path, [0.0, 1000.0, 2000.0, 200000.0, 300000.0])
    monkeypatch.setenv("ARTISATOMIC_LISBON_PATH", str(tmp_path))

    flog = io.StringIO()
    ionization_energy_in_ev, energy_levels, transitions = readlisbondata.read_levels_and_transitions(60, 3, flog)

    assert ionization_energy_in_ev == 22.09
    assert [level.energyabovegsinpercm for level in energy_levels] == [0.0, 1000.0, 2000.0]
    # the level ids are contiguous, and each name still carries the level's own file index
    assert [level.levelname for level in energy_levels] == [
        "4f-3(9)0, j=1.0, index=0",
        "4f-3(9)1, j=1.0, index=1",
        "4f-3(9)2, j=1.0, index=2",
    ]
    # the four lines are 0-1, 1-2, 2-3 and 3-4. The last two name a dropped level
    assert [(transition.lowerlevel, transition.upperlevel) for transition in transitions] == [(0, 1), (1, 2)]
    assert "The reader dropped 2 levels that are above the ionisation energy." in flog.getvalue()
    assert "skipped 2 transitions" in flog.getvalue()


def test_readlisbondata_keeps_a_level_at_the_ionisation_energy(tmp_path, monkeypatch):
    """The drop takes a level ABOVE the ionisation energy, so a level at that energy stays.

    The FAC reader uses the same strict comparison. A level at the ionisation energy is the
    series limit, and it still has an energy that adata.txt can hold.
    """
    from artisatomic import readlisbondata
    from artisatomic.base import hc_in_ev_cm

    # Nd III ionises at 22.09 eV. The second level sits exactly there, the third just above it
    at_ionization = 22.09 / hc_in_ev_cm
    write_lisbon_fixture(tmp_path, [0.0, at_ionization, at_ionization * 1.001])
    monkeypatch.setenv("ARTISATOMIC_LISBON_PATH", str(tmp_path))

    _, energy_levels, _ = readlisbondata.read_levels_and_transitions(60, 3, io.StringIO())

    assert [level.energyabovegsinpercm for level in energy_levels] == [0.0, at_ionization]


def test_readlisbondata_stops_on_an_ion_whose_levels_are_all_above_the_ionisation_energy(tmp_path, monkeypatch):
    """An ion with no bound level stops the run, as an ion with no level in the file does.

    The guard on the file itself runs before the drop, so it cannot see this case. Such an ion
    would go to the output with no level and no line.
    """
    from artisatomic import readlisbondata

    write_lisbon_fixture(tmp_path, [200000.0, 300000.0])
    monkeypatch.setenv("ARTISATOMIC_LISBON_PATH", str(tmp_path))

    with pytest.raises(ValueError, match="Every one of the 2 levels"):
        readlisbondata.read_levels_and_transitions(60, 3, io.StringIO())


def test_readlisbondata_stops_on_a_level_with_no_energy(tmp_path, monkeypatch):
    """A level with a blank energy stops the run, and the message names the levels file.

    The drop of the levels above the ionisation energy keeps such a level, because a null
    comparison is null. The check runs before the drop, so it sees every level of the file.
    """
    from artisatomic import readlisbondata

    write_lisbon_fixture(tmp_path, [0.0, "", 2000.0])
    monkeypatch.setenv("ARTISATOMIC_LISBON_PATH", str(tmp_path))

    with pytest.raises(ValueError, match="level with no energy"):
        readlisbondata.read_levels_and_transitions(60, 3, io.StringIO())


@pytest.mark.parametrize("extension", [".gz", ".zst", ".xz"])
def test_readlisbondata_reads_a_compressed_csv(tmp_path, monkeypatch, extension):
    """The reader finds a compressed levels or transitions CSV, as every other reader does.

    The setup scripts of the other data sets compress their files. polars reads the gzip and the
    zstd form itself. It cannot read the xz form, which the reader decompresses into memory.
    """
    from artisatomic import readlisbondata

    write_lisbon_fixture(tmp_path, [0.0, 1000.0, 2000.0])
    iondir = tmp_path / "Nd" / "NdIII"
    for csvfile in (iondir / "NdIII_Levels.csv", iondir / "NdIII_Transitions.csv"):
        with xopen(f"{csvfile}{extension}", mode="wt", encoding="utf-8") as fout:
            fout.write(csvfile.read_text(encoding="utf-8"))
        csvfile.unlink()
    monkeypatch.setenv("ARTISATOMIC_LISBON_PATH", str(tmp_path))

    _, energy_levels, transitions = readlisbondata.read_levels_and_transitions(60, 3, io.StringIO())

    assert [level.energyabovegsinpercm for level in energy_levels] == [0.0, 1000.0, 2000.0]
    assert [(transition.lowerlevel, transition.upperlevel) for transition in transitions] == [(0, 1), (1, 2)]


def test_readlisbondata_stops_on_a_transition_with_a_blank_level_index(tmp_path, monkeypatch):
    """A line with a blank level index stops the run, and the message names the column.

    is_in() gives null for a null index. The filter that drops the transitions of a dropped level
    would drop such a line and count it with the transitions above the ionisation energy.
    """
    from artisatomic import readlisbondata

    write_lisbon_fixture(tmp_path, [0.0, 1000.0, 2000.0])
    linesfile = tmp_path / "Nd" / "NdIII" / "NdIII_Transitions.csv"
    linesfile.write_text(linesfile.read_text(encoding="utf-8").replace("1,0,0.1", "1,,0.1"), encoding="utf-8")
    monkeypatch.setenv("ARTISATOMIC_LISBON_PATH", str(tmp_path))

    with pytest.raises(ValueError, match="level_index_lower"):
        readlisbondata.read_levels_and_transitions(60, 3, io.StringIO())


def test_readlisbondata_stops_on_a_file_that_holds_fewer_rows_than_its_header_declares(tmp_path, monkeypatch):
    """The Lisbon header declares the row count, as the FAC header does.

    A copy of a shared-drive file can stop between two rows. The short file then parses without
    an error, and the ion goes to the output with a part of its transitions.
    """
    from artisatomic import readlisbondata

    write_lisbon_fixture(tmp_path, [0.0, 1000.0, 2000.0])
    linesfile = tmp_path / "Nd" / "NdIII" / "NdIII_Transitions.csv"
    # drop the last row, which leaves one row under the declared count of two
    linesfile.write_text(linesfile.read_text(encoding="utf-8").rsplit("\n", maxsplit=2)[0] + "\n", encoding="utf-8")
    monkeypatch.setenv("ARTISATOMIC_LISBON_PATH", str(tmp_path))

    with pytest.raises(ValueError, match="declares Number Transitions = 2 but holds 1 rows"):
        readlisbondata.read_levels_and_transitions(60, 3, io.StringIO())

    # a header with no count at all means that the file stopped inside its own header
    linesfile.write_text(
        linesfile.read_text(encoding="utf-8").replace("Number Transitions:", "Number Somethings:"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="no Number Transitions line"):
        readlisbondata.read_levels_and_transitions(60, 3, io.StringIO())


def test_readlisbondata_stops_on_an_ion_that_the_data_set_does_not_hold():
    """The reader knows the file layout of Nd and U, ion stages 2 and 3, and of no other ion.

    An assert would vanish under python -O, and a wrong element would then give a file-not-found
    error that names a path instead of the ion.
    """
    from artisatomic import readlisbondata

    with pytest.raises(ValueError, match="Nd and U only"):
        readlisbondata.read_levels_and_transitions(26, 2, io.StringIO())

    with pytest.raises(ValueError, match="ion stages 2 and 3 only"):
        readlisbondata.read_levels_and_transitions(60, 4, io.StringIO())


def test_add_handlers_if_not_set():
    """add_handlers_if_not_set() returns a new list and never overrides an ion that is already present."""
    ion_handlers: list[tuple[int, list[tuple[int, str]]]] = [(26, [(1, "cmfgen"), (2, "cmfgen")])]
    unchanged = [(26, [(1, "cmfgen"), (2, "cmfgen")])]

    result = add_handlers_if_not_set(ion_handlers, [(58, 1)], "dream")
    assert result == [(26, [(1, "cmfgen"), (2, "cmfgen")]), (58, [(1, "dream")])]

    result = add_handlers_if_not_set(ion_handlers, [(26, 3)], "dream")
    assert result == [(26, [(1, "cmfgen"), (2, "cmfgen"), (3, "dream")])]

    # an ion stage that is already present keeps its handler, whatever the new one says
    result = add_handlers_if_not_set(ion_handlers, [(26, 2)], "dream")
    assert result == unchanged

    # each call returns a new list, so none of them changed the input list
    assert ion_handlers == unchanged

    # the caller can give the ion stages as tuples or as lists (e.g. directly from json.load())
    ion_handlers_json = t.cast("list[tuple[int, list[tuple[int, str]]]]", [(26, [[1, "cmfgen"]])])
    result = add_handlers_if_not_set(ion_handlers_json, [(26, 1)], "dream")
    assert result == [(26, [[1, "cmfgen"]])]


def test_parse_ion_handlers():
    """The JSON form becomes tuples, and the parser rejects an ion that names no handler."""
    from artisatomic.ionhandlers import parse_ion_handlers

    # json.load() gives nested lists; every ion must come back as an (ion_stage, handler) tuple
    assert parse_ion_handlers([[26, [[1, "cmfgen"], [2, "cmfgen"]]]]) == [(26, [(1, "cmfgen"), (2, "cmfgen")])]

    # a bare ion stage is a leftover from when the handler was optional. The parser must name it as
    # such here. A later failure would mention neither the element nor the file.
    with pytest.raises(TypeError, match=r"Z=26 ion stage 2 .* names no handler"):
        parse_ion_handlers([[26, [[1, "cmfgen"], 2]]])


def test_add_handlers_if_not_set_applies_the_limits():
    """A limit keeps an ion out of the list, and every element in the list keeps one ion or more."""
    limits = {"minionstage": 2, "maxionstage": 3, "maxatomicnumber": 30}

    assert add_handlers_if_not_set([], [(26, 2)], "cmfgen", **limits) == [(26, [(2, "cmfgen")])]
    assert add_handlers_if_not_set([], [(26, 1)], "cmfgen", **limits) == []
    assert add_handlers_if_not_set([], [(26, 4)], "cmfgen", **limits) == []
    assert add_handlers_if_not_set([], [(38, 2)], "cmfgen", **limits) == []

    # A limit of None includes every ion, which is what a direct call to a reader gets. A lower
    # limit of None excludes no ion stage, not even 0.
    assert add_handlers_if_not_set([], [(38, 9)], "cmfgen") == [(38, [(9, "cmfgen")])]
    assert add_handlers_if_not_set([], [(38, 0)], "cmfgen") == [(38, [(0, "cmfgen")])]

    # a rejected ion returns a new sorted list, as an accepted ion does
    unsorted = [(38, [(2, "cmfgen")]), (26, [(1, "cmfgen")])]
    assert add_handlers_if_not_set(unsorted, [(26, 9)], "cmfgen", **limits) == [
        (26, [(1, "cmfgen")]),
        (38, [(2, "cmfgen")]),
    ]


# every module that offers extend_ion_list()
extend_ion_list_modules = [
    "groundstatesonlynist",
    "readdreamdata",
    "readfacdata",
    "readfloers25data",
    "readhillierdata",
    "readmonsdata",
    "readadasdata",
    "readtanakajpltdata",
]


def keep_ions_within_limits(
    ion_handlers: list[tuple[int, list[tuple[int, str]]]],
    minionstage: int,
    maxionstage: int,
    maxatomicnumber: int,
) -> list[tuple[int, list[tuple[int, str]]]]:
    """Drop every ion outside the limits, and every element that then holds no ion."""
    kept = [
        (atomic_number, [ion for ion in listions if minionstage <= ion[0] <= maxionstage])
        for atomic_number, listions in ion_handlers
        if atomic_number <= maxatomicnumber
    ]
    return [(atomic_number, listions) for atomic_number, listions in kept if listions]


@pytest.mark.parametrize("modulename", extend_ion_list_modules)
def test_extend_ion_list_forwards_the_limits(modulename):
    """Each reader gives the three limits to add_handlers_if_not_set(), which keeps its own ions out.

    get_ion_handlers() gives the limits to each reader by keyword. A reader that drops one of them
    changes which ions a run writes. No checksum set finds that, because each set reads an ion
    handlers file instead. The limits below come from the ions of the reader itself, so each one
    excludes a part of them whatever the data set holds.
    """
    module = importlib.import_module(f"artisatomic.{modulename}")
    try:
        allions = module.extend_ion_list([])
    except FileNotFoundError as exc:
        pytest.skip(f"the data set of {modulename} is not available here: {exc}")

    if not allions:
        if module is readhillierdata:
            msg = "the CMFGEN corpus is missing"
            raise AssertionError(msg)
        pytest.skip(f"{modulename} found no ion here")

    ion_stages = sorted({ion_stage for _, listions in allions for ion_stage, _ in listions})
    atomic_numbers = sorted(atomic_number for atomic_number, _ in allions)
    # each limit excludes at least one ion where the reader offers more than one value
    limitsets = [
        (ion_stages[-1], ion_stages[-1], atomic_numbers[-1]),
        (ion_stages[0], ion_stages[0], atomic_numbers[-1]),
        (ion_stages[0], ion_stages[-1], atomic_numbers[0]),
    ]

    for minionstage, maxionstage, maxatomicnumber in limitsets:
        result = module.extend_ion_list(
            [], minionstage=minionstage, maxionstage=maxionstage, maxatomicnumber=maxatomicnumber
        )
        assert result == keep_ions_within_limits(allions, minionstage, maxionstage, maxatomicnumber)
        assert result, "a limit that comes from the ions of the reader must keep one ion or more"


def test_ion_limits_with_an_input_file_stop_the_run(tmp_path, monkeypatch):
    """An ion handlers file selects the ions, so a limit with that file is an error and not a filter."""
    from artisatomic.cli import main
    from artisatomic.ionhandlers import get_ion_handlers

    monkeypatch.chdir(tmp_path)
    # a contiguous selection, so a guard that stops working fails here and not in a later check
    ions = [[ion_stage, "cmfgen"] for ion_stage in range(1, 7)]
    (tmp_path / "artisatomicionhandlers.json").write_text(json.dumps([[8, ions]]), encoding="utf-8")

    # the file keeps ion stage 6, which the -maxionstage of the call excludes
    assert get_ion_handlers(1, 5, 8) == [(8, [(ion_stage, "cmfgen") for ion_stage in range(1, 7)])]

    monkeypatch.setattr("sys.argv", ["makeartisatomicfiles", "-maxionstage", "3", "-maxatomicnumber", "30"])
    with pytest.raises(ValueError, match=r"remove -maxionstage, -maxatomicnumber\."):
        main()

    # a limit that holds the value of its default is still a limit that the command line gave
    monkeypatch.setattr("sys.argv", ["makeartisatomicfiles", "-maxionstage", "5"])
    with pytest.raises(ValueError, match=r"remove -maxionstage\."):
        main()


def test_parent_elevel_zero_normalisation_is_anchored():
    """Only a parent level that IS 0.0 may become 0, not one that only contains it.

    An earlier str.replace("0.0", "0") matched anywhere in the value. So it also rewrote "10.05"
    to "105" and "100.0" to "100". Those levels then went into the wrong group_by bucket in
    download_gammaspec_betaminus_alpha. The test calls the script's own expression.
    """
    from artisatomic.download_gammaspec_betaminus_alpha import normalise_parent_elevel

    parent_elevels = ["0.0", "0", "10.05", "100.0", "1234.5", "0.05", "0.00", "", "x"]
    normalised = (
        pl.DataFrame({"parent_elevel": parent_elevels})
        .with_columns(normalise_parent_elevel())["parent_elevel"]
        .to_list()
    )

    # the ground state, in every spelling, collapses to one group; nothing else moves
    assert normalised == ["0", "0", "10.05", "100.0", "1234.5", "0.05", "0", "", "x"]


def test_parallel_map_rejects_iterables_of_different_lengths():
    """parallel_map() refuses a short iterable, whichever path the call would otherwise have taken."""
    from artisatomic.base import parallel_map

    # Executor.map() and thread_map() stop at the shortest iterable, while the serial shortcut's
    # zip(strict=True) raises. The check therefore has to happen before the choice of the path. 4 items
    # take the shortcut and 40 the pool, and neither may silently do less work than the caller asked for.
    for nitems in (4, 40):
        with pytest.raises(ValueError, match=r"different lengths"):
            parallel_map(operator.sub, range(nitems), range(nitems - 1))


def test_parallel_map_matches_serial_results_on_both_sides_of_the_cutoff():
    """Both paths apply fn to one item of each iterable, in the input order."""
    from artisatomic.base import parallel_map

    # 4 items takes the serial shortcut, 40 goes to the pool. Subtraction does not commute, so
    # the results also show which iterable reaches which parameter.
    for nitems in (4, 40):
        minuends = list(range(nitems))
        subtrahends = list(range(100, 100 + nitems))
        expected = [minuend - subtrahend for minuend, subtrahend in zip(minuends, subtrahends, strict=True)]

        assert parallel_map(operator.sub, minuends, subtrahends) == expected

    # a generator yields only once and has no length, so it must survive the conversion to a list
    assert parallel_map(operator.sub, (i for i in range(4)), [1] * 4) == [-1, 0, 1, 2]


def test_split_element_ionstage_str():
    """'FeII' splits into element and ion stage, including the symbols made only of Roman numeral letters."""
    from artisatomic.base import split_element_ionstage_str

    assert split_element_ionstage_str("FeII") == (26, 2)
    assert split_element_ionstage_str("DyIII") == (66, 3)
    assert split_element_ionstage_str("SiI") == (14, 1)
    assert split_element_ionstage_str("HI") == (1, 1)

    # rstrip("IVX") would leave nothing for the elements whose symbols consist of those letters,
    # so these are the cases that used to raise ValueError
    assert split_element_ionstage_str("VI") == (23, 1)  # vanadium I, not "V" as a numeral
    assert split_element_ionstage_str("VIII") == (23, 3)  # vanadium III
    assert split_element_ionstage_str("IV") == (53, 5)  # iodine V
    assert split_element_ionstage_str("II") == (53, 1)  # iodine I
    assert split_element_ionstage_str("XeIV") == (54, 4)

    with pytest.raises(ValueError, match="Could not split"):
        split_element_ionstage_str("NotAnIon")


def test_hillier_ions_data_matches_the_cmfgen_corpus():
    """ions_data holds one entry for every CMFGEN ion, with the phot file names of that ion.

    The ion stages come from ranges, so a wrong bound drops an ion without a syntax error.
    extend_ion_list() then never offers that ion, and the checksum tests do not find the
    mistake. The two CMFGEN sets read 31 of the 199 ions. This test checks the full list.
    """
    expected_ion_stages = {
        1: [1, 2],  # H
        2: [1, 2],  # He
        6: [1, 2, 3, 4, 5, 6],  # C
        7: [1, 2, 3, 4, 5, 6, 7],  # N
        8: [1, 2, 3, 4, 5, 6, 7, 8],  # O
        9: [2, 3],  # F
        10: [1, 2, 3, 4, 5, 6, 7, 8],  # Ne
        11: [1, 2, 3, 4, 5, 6, 7, 8, 9],  # Na
        12: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Mg
        13: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # Al
        14: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # Si
        15: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # P
        16: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # S
        17: [4, 5, 6, 7],  # Cl
        18: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Ar
        19: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # K
        20: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # Ca
        21: [1, 2, 3],  # Sc
        22: [2, 3],  # Ti
        24: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],  # Cr
        25: [2, 3, 4, 5, 6, 7],  # Mn
        26: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],  # Fe
        27: [1, 2, 3, 4, 5, 6, 7, 8, 9],  # Co
        28: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],  # Ni
    }
    ion_stages = {
        atomic_number: sorted(
            stage for (found_atomic_number, stage) in readhillierdata.ions_data if found_atomic_number == atomic_number
        )
        for atomic_number in sorted({atomic_number for atomic_number, _ in readhillierdata.ions_data})
    }
    assert ion_stages == expected_ion_stages

    # every ion whose phot files are not the single default file. H II is the bare top ion, so
    # it has no files at all.
    expected_photfilenames = {
        (1, 1): ("hiphot.dat",),
        (1, 2): (),
        (2, 1): ("heiphot_a7.dat",),
        (2, 2): ("he2phot.dat",),
        (6, 2): ("phot_data_A", "phot_data_B"),
        (6, 3): ("phot_data_A", "phot_data_B"),
        (7, 1): ("phot_data_A", "phot_data_B", "phot_data_C", "phot_data_D"),
        (7, 2): ("phot_data_A", "phot_data_B"),
        (7, 3): ("phot_data_A", "phot_data_B"),
        (7, 4): ("phot_data_A", "phot_data_B"),
        (8, 1): ("phot_data_A", "phot_data_B"),
        (8, 4): ("phot_data_A", "phot_data_B"),
        (9, 2): ("phot_data_a", "phot_data_b", "phot_data_c"),
        (9, 3): ("phot_data_a", "phot_data_b", "phot_data_c", "phot_data_d"),
        (14, 2): ("phot_data_A", "phot_data_B"),
        (15, 4): ("phot_data_A", "phot_data_B"),
    }
    assert {
        ion: ionfiles.photfilenames
        for ion, ionfiles in readhillierdata.ions_data.items()
        if ionfiles.photfilenames != ("phot_data_A",)
    } == expected_photfilenames


def test_hillier_extend_ion_list():
    """The CMFGEN ion list names the cmfgen handler, and it excludes hydrogen by default."""
    result = readhillierdata.extend_ion_list([])
    assert all(atomic_number != 1 for atomic_number, _ in result)
    assert all(handler == "cmfgen" for _, listions in result for _, handler in listions)
    assert {(2, 1), (26, 1)} <= {(atomic_number, ion_stage) for atomic_number, l in result for ion_stage, _ in l}

    assert any(atomic_number == 1 for atomic_number, _ in readhillierdata.extend_ion_list([], include_hydrogen=True))


def test_combine_phixs_routes():
    """The level's table is the sum of the kept routes, and the fractions are the shares of the sums."""
    strong = np.array([4.0, 2.0, 1.0])
    weak = np.array([1.0, 1.0, 1.0])
    faint = np.array([0.05, 0.0, 0.0])
    closed = np.zeros(3)
    combined = combine_phixs_routes([("a", strong), ("b", weak), ("c", faint), ("d", closed)], fractioncut=0.01)
    assert combined.factors == [("a", 7.0), ("b", 3.0), ("c", 0.05)]
    assert [target for target, _ in combined.fractions] == ["a", "b"]
    assert [fraction for _, fraction in combined.fractions] == pytest.approx([0.7, 0.3])
    assert combined.dropped == [("c", 0.05)]
    np.testing.assert_array_equal(combined.table, strong + weak)
    # the input tables stay as they were
    np.testing.assert_array_equal(strong, [4.0, 2.0, 1.0])

    # a higher cut drops the weak route, and the kept route takes the whole fraction
    combined = combine_phixs_routes([("a", strong), ("b", weak)], fractioncut=0.5)
    assert combined.fractions == [("a", 1.0)]
    np.testing.assert_array_equal(combined.table, strong)

    # the strongest routes stay at every cut, so the table is never empty
    combined = combine_phixs_routes([("a", weak), ("b", weak)], fractioncut=0.5)
    assert combined.fractions == [("a", 0.5), ("b", 0.5)]
    assert combined.dropped == []

    # no open route gives a zero table and no target
    combined = combine_phixs_routes([("a", closed)])
    assert combined.fractions == []
    np.testing.assert_array_equal(combined.table, closed)
    with pytest.raises(ValueError, match="at least one route"):
        combine_phixs_routes([])


def test_reduce_phixs_tables_worker():
    """The downsample of a cross section table keeps the recombination rate at the optimisation temperature."""
    nphixspoints = 100
    phixsnuincrement = 0.03
    temperature = 6000.0
    sigma_0 = 4.0
    xgrid = output_xgrid(nphixspoints, phixsnuincrement)

    # dense input table with a hydrogenic-like sigma_0 * (E_threshold / E)**3 cross section
    energyryd = np.linspace(1.0, 1.0 + phixsnuincrement * (nphixspoints + 1) * 2, 5000)
    tablein = np.column_stack([energyryd, sigma_0 * (energyryd[0] / energyryd) ** 3])

    reduced = reduce_phixs_tables_worker(temperature, xgrid, tablein)
    assert len(reduced) == nphixspoints
    assert abs(reduced[0] / sigma_0 - 1) < 0.05
    assert np.all(np.diff(reduced) < 0)

    # a constant cross section must stay exact
    tablein_const = np.column_stack([energyryd, np.full_like(energyryd, 2.5)])
    reduced_const = reduce_phixs_tables_worker(temperature, xgrid, tablein_const)
    assert np.allclose(reduced_const, 2.5, rtol=1e-6)

    # the downsample must keep the recombination rate integral
    # sigma(nu) * nu**2 * exp(-h*nu / (k_B * T)) at the optimisation temperature
    ryd_to_hz = 3289841960250880.5
    h_over_kb_in_k_sec = 4.799243073366221e-11

    def recomb_integral(en_ryd, sigmas):
        nu = np.asarray(en_ryd) * ryd_to_hz
        return np.trapezoid(sigmas * nu**2 * np.exp(-h_over_kb_in_k_sec * nu / temperature), nu)

    # reconstruct the reduced table as piecewise-constant over the output grid intervals
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


def test_reduce_phixs_tables_worker_weight_does_not_underflow():
    """A high threshold at a low temperature must still give the weighted average of the bin.

    The weight nu**2 exp(-h nu / k T) underflows to zero for h nu / k T above 745. The worker
    subtracts the lowest frequency of the bin from nu, which keeps the weight at 1.0 or below.
    Without the subtraction each such bin gets the unweighted mean of its samples.
    """
    nphixspoints = 100
    phixsnuincrement = 0.03
    temperature = 500.0
    threshold_ryd = 30.7 / ryd_to_ev
    binindex = 50

    # the edges of one output bin, as the worker computes them
    xgrid = output_xgrid(nphixspoints, phixsnuincrement)
    enlow = 0.5 * (xgrid[binindex - 1] + xgrid[binindex]) * threshold_ryd
    enhigh = 0.5 * (xgrid[binindex] + xgrid[binindex + 1]) * threshold_ryd

    # a cross section that steps up by a factor of 100 at the middle of that bin
    energyryd = np.linspace(threshold_ryd, 5.0 * threshold_ryd, 20001)
    sigma_low = 1.0
    sigma_high = 100.0
    tablein = np.column_stack([energyryd, np.where(energyryd < 0.5 * (enlow + enhigh), sigma_low, sigma_high)])

    reduced = reduce_phixs_tables_worker(temperature, xgrid, tablein)

    # the weight falls by exp(-10.7) over the first half of the bin, so the low edge dominates
    assert abs(reduced[binindex] - sigma_low) < 0.05
    # the unweighted mean of the bin is about 50, which is what the underflow gave before
    assert reduced[binindex] < 0.1 * (sigma_low + sigma_high)


def test_reduce_phixs_tables_worker_rejects_unsorted_table():
    """A table that decreases in energy must raise, because the worker interpolates in energy."""
    energyryd = np.array([1.0, 1.2, 1.1, 1.3])
    tablein = np.column_stack([energyryd, np.full_like(energyryd, 2.0)])
    with pytest.raises(ValueError, match="decreases"):
        reduce_phixs_tables_worker(6000.0, output_xgrid(100, 0.03), tablein)


def test_reduce_phixs_tables_names_the_key_of_a_bad_table():
    """A negative cross section must raise, and the message must name the key of the table.

    reduce_phixs_tables() holds the keys, and the worker gets one table. The worker must
    therefore receive the key of its own table.
    """
    from artisatomic.phixs import reduce_phixs_tables

    energyryd = np.linspace(1.0, 20.0, 500)
    tablein = np.column_stack([energyryd, np.full_like(energyryd, -1.0)])
    with pytest.raises(ValueError, match=r"bin integral is not positive.*'Fe I 3d7 a4F'"):
        reduce_phixs_tables({"Fe I 3d7 a4F": tablein}, 6000.0, 100, 0.03)

    # a key alone does not say which ion or which file the table came from, so a caller can
    # give a label as well
    with pytest.raises(ValueError, match=r"Z=26 Fe I phot_data.*'Fe I 3d7 a4F'"):
        reduce_phixs_tables({"Fe I 3d7 a4F": tablein}, 6000.0, 100, 0.03, label="Z=26 Fe I phot_data")

    # parallel_map() runs a batch of more than 32 tables in the pool, so the key must
    # reach the worker there too, and the error must come back with its message
    goodtable = np.column_stack([energyryd, np.full_like(energyryd, 1.0)])
    tables = {f"level {i}": goodtable for i in range(40)}
    tables["bad level"] = tablein
    with pytest.raises(ValueError, match=r"bin integral is not positive.*'bad level'"):
        reduce_phixs_tables(tables, 6000.0, 100, 0.03)


def adf04_sample_path() -> Path:
    """Return the path of the committed ADAS adf04 sample."""
    return (PYDIR / ".." / "atomic-data-adas" / "co_tyndall_test_sample" / "adf04_v1").resolve()


def test_read_adf04():
    """An adf04 file yields levels and effective collision strengths keyed by zero-based level ids."""
    flog = io.StringIO()
    ionization_energy_ev, energylevels, upsilondict, _ = readadasdata.read_adf04(
        adf04_sample_path(), flog, 5010.0, 27, 3
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


def test_is_adf04_terminator():
    """The terminator test reads the first field, so padding and negative values do not confuse it."""
    for line in ("   -1\n", "  -1\n", "-1\n", "\t-1\n", "  -1  -1\n"):
        assert readadasdata.is_adf04_terminator(line)

    for line in ("  -1.0E+00 no data for this pair\n", "  -10  3 1.0\n", "\n", "   1   2 1.0+00\n"):
        assert not readadasdata.is_adf04_terminator(line)


def read_adf04_sample_lines() -> list[str]:
    """Return the lines of the committed adf04 sample, which stops inside the collision block."""
    with xopen_check_extension(adf04_sample_path()) as fsample:
        return fsample.readlines()


def test_read_adf04_stops_at_the_collision_terminator(tmp_path):
    """The reader stops at the "-1" row, and skips the rows of a different process."""
    lines = read_adf04_sample_lines()
    # an ADAS process-code row sits inside the collision block; the trailer follows the terminator
    processrow = "R  1  +1" + " 3.24-13" * 22 + "\n"
    trailer = ["  -1\n", "  -1  -1\n", "C-----\n", "C PRODUCER: test\n", "   1   2 9.99+99\n"]
    filepath = tmp_path / "27_3.adf04"
    filepath.write_text("".join([*lines, processrow, *trailer]))

    flog = io.StringIO()
    _, energylevels, upsilondict, _ = readadasdata.read_adf04(filepath, flog, 5010.0, 27, 3)
    assert len(energylevels) == 262
    assert len(upsilondict) == 235
    assert "The reader skipped 1 collision rows that are not an electron impact excitation." in flog.getvalue()
    assert "The levels, the transitions and the collision strengths come from " in flog.getvalue()
    assert "level pairs" not in flog.getvalue()


def test_read_adf04_keeps_the_rows_after_a_negative_value(tmp_path):
    """A row that starts with a negative number is not the terminator, so the block continues."""
    lines = read_adf04_sample_lines()
    middle = 264 + (len(lines) - 264) // 2
    filepath = tmp_path / "27_3.adf04"
    filepath.write_text("".join([*lines[:middle], "  -1.0E+00 no data for this pair\n", *lines[middle:]]))

    flog = io.StringIO()
    _, _, upsilondict, _ = readadasdata.read_adf04(filepath, flog, 5010.0, 27, 3)
    assert len(upsilondict) == 235


def test_rename_old_data_directory(tmp_path, capsys):
    """The reader renames atomic-data-qub, or it merges its files into the new directory."""
    old, new = tmp_path / "atomic-data-qub", tmp_path / "atomic-data-adas"
    readadasdata.rename_old_data_directory(old, new)  # no old directory: nothing to do
    assert not new.exists()

    old.mkdir()
    (old / "26_3.adf04").write_text("a", encoding="utf-8")
    readadasdata.rename_old_data_directory(old, new)
    assert not old.exists()
    assert (new / "26_3.adf04").read_text(encoding="utf-8") == "a"

    # After an update of the repository, the new directory holds the tracked files. The sample
    # directory is then in the two directories, and the Finder writes .DS_Store into each.
    (old / "co_tyndall_test_sample").mkdir(parents=True)
    (old / "co_tyndall_test_sample" / "untracked.gz").write_text("untracked", encoding="utf-8")
    (old / "co_tyndall").mkdir()
    (old / "26_3.adf04").write_text("old", encoding="utf-8")
    (new / "co_tyndall_test_sample").mkdir()
    (new / "co_tyndall_test_sample" / "adf04_v1.gz").write_text("tracked", encoding="utf-8")
    for directory in (old, new):
        (directory / ".DS_Store").write_text("", encoding="utf-8")
    capsys.readouterr()
    readadasdata.rename_old_data_directory(old, new)
    assert (new / "co_tyndall").is_dir()
    assert (new / "co_tyndall_test_sample" / "untracked.gz").read_text(encoding="utf-8") == "untracked"
    assert (new / "co_tyndall_test_sample" / "adf04_v1.gz").read_text(encoding="utf-8") == "tracked"
    # the function does not replace a file of the new directory, and it names the file that stays
    assert (new / "26_3.adf04").read_text(encoding="utf-8") == "a"
    assert [path.name for path in old.rglob("*")] == ["26_3.adf04"]
    output = capsys.readouterr().out
    assert "keeps these files" in output
    assert "26_3.adf04" in output


def test_rename_old_data_directory_does_not_move_the_data_of_a_symbolic_link(tmp_path, capsys):
    """A link to a shared directory keeps its data, and a failure of the rename does not stop the run."""
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "26_3.adf04").write_text("a", encoding="utf-8")
    old, new = tmp_path / "atomic-data-qub", tmp_path / "atomic-data-adas"
    old.symlink_to(shared, target_is_directory=True)

    new.mkdir()
    readadasdata.rename_old_data_directory(old, new)
    assert old.is_symlink()
    assert (shared / "26_3.adf04").exists()
    assert not list(new.iterdir())
    assert "is a symbolic link" in capsys.readouterr().out

    new.rmdir()
    readadasdata.rename_old_data_directory(old, new)
    assert new.is_symlink()
    assert not old.is_symlink()
    assert (shared / "26_3.adf04").exists()

    # a file has the new name: the old directory keeps its files, and no exception comes out
    old.mkdir()
    (old / "27_3.adf04").write_text("b", encoding="utf-8")
    new.unlink()
    new.write_text("", encoding="utf-8")
    readadasdata.rename_old_data_directory(old, new)
    assert (old / "27_3.adf04").exists()


def test_extend_ion_list_finds_a_compressed_adf04():
    """Ion discovery must find an adf04 file that is compressed."""
    assert (38, [(1, "adas")]) in readadasdata.extend_ion_list({})
    assert (20, [(3, "adas")]) in readadasdata.extend_ion_list({})


def test_convert_eissner_to_standard():
    """The converter gives the standard notation of the documented example and of a Ca III level."""
    assert convert_eissner_to_standard("521522563524565") == "1s22s22p63s23p6"
    assert convert_eissner_to_standard("522563524555516") == "2s22p63s23p53d1"
    # the order of the AUTOSTRUCTURE files: 9=4d, A=4f, B=5s. The first shell can give q alone.
    assert convert_eissner_to_standard("51951A51B") == "4d14f15s1"
    assert convert_eissner_to_standard("21522") == "1s22s2"
    assert convert_eissner_to_standard("21") == "1s2"
    assert convert_eissner_to_standard("3A52B") == "4f35s2"
    # the order of the specification: 9=4d, 0=4f, A=5s
    assert convert_eissner_to_standard("51951051A", "specification") == "4d14f15s1"
    # "520" has the shell character that only the specification uses. The last three give a
    # shell more electrons than it holds (1s9, 1s14, 2s6).
    for malformed in ("521junk", "501", "651", "520", "591", "641", "62"):
        assert not is_eissner_config(malformed)
        with pytest.raises(ValueError, match="Not an Eissner configuration"):
            convert_eissner_to_standard(malformed)


def test_eissner_total_l_is_possible():
    """The total L of a level shows which order of the Eissner shell characters a file uses."""
    from artisatomic.levelnames import eissner_total_l_is_possible

    # 1s 4f has L = 3, and 1s 5s has L = 0 (the OPEN-ADAS file for He-like C)
    assert eissner_total_l_is_possible("51151A", 3)
    assert eissner_total_l_is_possible("51151B", 0)
    assert not eissner_total_l_is_possible("51151A", 0)
    assert eissner_total_l_is_possible("51151A", 0, "specification")
    # 3p5 4f gives L = 2, 3 or 4 (the Ca III file). With the order of the specification it is 3p5 5s, with L = 1.
    config = "52256352455551A"
    assert [total_l for total_l in range(6) if eissner_total_l_is_possible(config, total_l)] == [2, 3, 4]
    assert [total_l for total_l in range(6) if eissner_total_l_is_possible(config, total_l, "specification")] == [1]
    # closed shells have L = 0, and one hole in a p shell has L = 1
    assert [total_l for total_l in range(3) if eissner_total_l_is_possible("521522563", total_l)] == [0]
    assert [total_l for total_l in range(3) if eissner_total_l_is_possible("521522553", total_l)] == [1]
    # for 3d3 the test is only an upper limit: 2 + 2 + 1
    assert eissner_total_l_is_possible("536", 5)
    assert not eissner_total_l_is_possible("536", 6)
    assert eissner_total_l_is_possible("4p65s2", 0)  # not Eissner notation: no test


def test_expand_standard_config_expands_only_a_real_subshell():
    """A letter is an occupation only in a word that can be a subshell: n > l and q <= 2(2l+1)."""
    assert expand_standard_config("3p6 3da") == "3p6 3d10"
    assert expand_standard_config("4fe") == "4f14"
    assert expand_standard_config("as1") == "10s1"
    # each word decides for itself, so a bare "4s" does not stop the expansion of "3da"
    assert expand_standard_config("3s2 3p6 3da 4s") == "3s2 3p6 3d10 4s"
    assert expand_standard_config("3s2  3p6 3da ") == "3s2  3p6 3d10 "
    # a term with its parity, an occupation above the capacity of the shell, and a label with no digit
    for label in ("3d7 4fo", "2po", "4ff", "3dd", "1ss", "3pa", "grd", "ion", "spd", "3da4p", ""):
        assert expand_standard_config(label) == label


def make_adf04(levels: t.Sequence[str], rows: t.Sequence[str], *, header: str, temperatures: str) -> str:
    """Return the text of a minimal adf04 file: the header, the levels, the temperatures and the collision rows."""
    return "\n".join([header, *levels, "   -1", temperatures, *rows, "  -1", "  -1  -1", ""])


hydrogen_header = "H+ 0         1         1    109679."
hydrogen_levels = ("    1 1S                 (2)0( 0.5)        0.", "    2 2P                 (2)1( 2.5)    82303.")
two_temperatures = " 1.00    3       5.80+03 1.16+04"


def write_hydrogen_adf04(
    tmp_path: Path,
    rows: t.Sequence[str],
    *,
    levels: t.Sequence[str] = hydrogen_levels,
    header: str = hydrogen_header,
    temperatures: str = two_temperatures,
) -> Path:
    """Write a minimal H I adf04 file and return its path."""
    filepath = tmp_path / "1_1.adf04"
    filepath.write_text(make_adf04(levels, rows, header=header, temperatures=temperatures), encoding="utf-8")
    return filepath


def read_hydrogen_adf04(tmp_path: Path, rows: t.Sequence[str], **parts: t.Any) -> dict[tuple[int, int], float]:
    """Write a minimal H I adf04 file and return the upsilon values at 5000 K."""
    return readadasdata.read_adf04(write_hydrogen_adf04(tmp_path, rows, **parts), io.StringIO(), 5000.0, 1, 1)[2]


def test_read_adf04_header():
    """The parent term and the element symbol are optional, and the numbers of the header must agree."""
    read_header = readadasdata._read_adf04_header  # ruff: ignore[private-member-access]
    for line, atomic_number, ion_stage in (
        ("H+ 0         1         1    109679.\n", 1, 1),
        ("H+ 0         1         1    109679\n", 1, 1),
        ("HE+ 0         2         1    109679.0000\n", 2, 1),
        ("C + 3         6         4    109679.0(1S)  2931440.0(3S)\n", 6, 4),
        ("  + 2        26         3    109679.(6S)\n", 26, 3),
    ):
        assert read_header(line, atomic_number, ion_stage, "x.adf04") == pytest.approx(13.5984, abs=1e-3)
    with pytest.raises(ValueError, match="Ion stage"):
        read_header("H+ 0         1         1    109679.\n", 1, 2, "x.adf04")
    # the header gives the ion charge and the ion stage, and they must agree
    with pytest.raises(ValueError, match="ion charge 7"):
        read_header("Sr+ 7        38         1     45932.2036(  )\n", 38, 1, "x.adf04")
    # a number in a different form must not give its first digits
    for number in ("4.10+05(1s)", "1.0E+06(1s)", "1.43175D+04", "109,679."):
        with pytest.raises(ValueError, match="Cannot read the adf04 header line"):
            read_header(f"Ca+ 2        20         3    {number}\n", 20, 3, "x.adf04")


def test_adf04_level_regex():
    """The regex takes the first "(2S+1)L(J)" group, and the energy must be a full number."""
    level_regex = readadasdata.adf04_level_regex
    line = "    7 3D7 4S1            (5)2( 4.0)     439.0279  (3)1( 2) 12"
    levelmatch = level_regex.match(line)
    assert levelmatch is not None
    assert levelmatch.groups() == ("7", "3D7 4S1", "5", "2", "4.0", "439.0279")
    # the free text of the specification can start directly after the energy
    for tail, energy in (("0.0(3P)", "0.0"), ("439.0279X", "439.0279"), ("0.0{1}1.000", "0.0"), ("82303.", "82303.")):
        levelmatch = level_regex.match(f"    1 1S2 2S1            (2)0( 0.5)        {tail}")
        assert levelmatch is not None
        assert levelmatch[6] == energy
    # an energy in exponent form must not give its first digits
    for energy in ("4.39+02", "1.43175+04", "1.0E+03"):
        assert level_regex.match(f"    2 3S2 3P6 3D6       (5)2( 3.0)      {energy}") is None


def test_read_adf04_process_code_and_touching_values(tmp_path):
    """A row with the process code "1" in column 1 is a collision row, and fixed columns separate two values."""
    filepath = write_hydrogen_adf04(tmp_path, ["1  2   1 6.27+08 4.29-01 5.29-01-3.01-02"])
    assert readadasdata.read_adf04(filepath, io.StringIO(), 5000.0, 1, 1)[2] == {(0, 1): pytest.approx(0.429)}
    # the last upsilon touches the Born limit
    assert readadasdata.read_adf04(filepath, io.StringIO(), 1e6, 1, 1)[2] == {(0, 1): pytest.approx(0.529)}


def test_read_adf04_temperature_line(tmp_path):
    """The reader takes ITYP and the temperatures from the fixed columns, and it stops for a different layout."""
    rows = ["   2   1 6.27+08 4.29-01 5.29-01"]
    # ZEFF can be blank, ITYP is a number, and the 6 columns before the temperatures have no use
    for temperatures in (
        "         3       5.80+03 1.16+04",
        " 1.00   03       5.80+03 1.16+04",
        " 1.00    3 zz    5.80+03 1.16+04",
    ):
        assert read_hydrogen_adf04(tmp_path, rows, temperatures=temperatures) == {(0, 1): pytest.approx(0.429)}
    for temperatures, message in (
        (" 1.00    1       5.80+03 1.16+04", "ITYP field must be 3, and it is '1'"),
        (" 1.00    3", "names no temperatures"),
        # free format, and values that are 9 columns wide: the rows of such a file are not in the fixed columns
        (" 1.00    3   5.80+03  1.16+04", "fixed columns of the adf04 specification"),
        (" 1.00    3        5.800+03 1.160+04", "fixed columns of the adf04 specification"),
        (" 1.00    3       5.80+03 1.16+04       X", "temperature that is not a number"),
        (" 1.00    3       1.0D+03 1.16+04", "temperature that is not a number"),
    ):
        with pytest.raises(ValueError, match=message) as excinfo:
            read_hydrogen_adf04(tmp_path, rows, temperatures=temperatures)
        assert "1_1.adf04" in str(excinfo.value)

    truncated = tmp_path / "truncated.adf04"
    truncated.write_text("\n".join([hydrogen_header, *hydrogen_levels, "   -1", ""]), encoding="utf-8")
    with pytest.raises(ValueError, match="ends before the line that gives the temperatures"):
        readadasdata.read_adf04(truncated, io.StringIO(), 5000.0, 1, 1)


def test_read_adf04_file_index_columns(tmp_path):
    """A file index is a Fortran integer at the right of its columns, and the level count decides the use of column 1."""
    levels = [f"{i:5d} 1S                 (2)0( 0.5) {i - 1:12d}." for i in range(1, 1202)]
    wide_header = "H+ 0         1         1  99999999."
    values = " 6.27+08 4.29-01 5.29-01"
    # more than 999 levels: columns 1 to 4 are the file index, and "1 23" is a process code and a file index
    rows = ["1123   5 6.27+08 1.00-01 9.00-01", "1 23   9 6.27+08 2.00-01 9.00-01", "   7   5 6.27+08 3.00-01 9.00-01"]
    upsilondict = read_hydrogen_adf04(tmp_path, rows, levels=levels, header=wide_header)
    assert upsilondict == {(4, 1122): pytest.approx(0.1), (8, 22): pytest.approx(0.2), (4, 6): pytest.approx(0.3)}
    # a file index above the number of levels stops the run. It is not a process code and a smaller file index.
    with pytest.raises(ValueError, match="file indices 6, 3999"):
        read_hydrogen_adf04(tmp_path, ["3999   6" + values], levels=levels, header=wide_header)

    # 200 levels: column 1 is the process code, so "1123" is level 123. The reader cannot parse the
    # other rows. They have a zero at the left, an integer at the left of its columns, the process
    # code "4", and a sign.
    rows = ["1123   5", "2005   1", "42     1", "4 12   1", "   9  +1"]
    filepath = write_hydrogen_adf04(tmp_path, [row + values for row in rows], levels=levels[:200])
    flog = io.StringIO()
    assert sorted(readadasdata.read_adf04(filepath, flog, 5000.0, 1, 1)[2]) == [(4, 122)]
    assert "The reader skipped 4 collision rows that it could not parse." in flog.getvalue()


def test_read_adf04_skips_the_rows_of_a_different_process(tmp_path):
    """The first field of a row shows a different process or a comment, also if it is not in column 1."""
    rows = [
        "   2   1 6.27+08 4.29-01 5.29-01",
        " P  2   1 1.00-10 1.00-10 1.00-10",
        "R  1  +1         3.24-13 3.24-13",
        "C a comment",
    ]
    flog = io.StringIO()
    upsilondict = readadasdata.read_adf04(write_hydrogen_adf04(tmp_path, rows), flog, 5000.0, 1, 1)[2]
    assert upsilondict == {(0, 1): pytest.approx(0.429)}
    assert "The reader skipped 3 collision rows that are not an electron impact excitation." in flog.getvalue()
    assert "could not parse" not in flog.getvalue()


def test_read_adf04_returns_only_the_rows_that_it_can_parse(tmp_path):
    """The caller makes a transition from each returned row, so a row with values in the wrong columns must not be there."""
    rows = [
        "   2   1 1.00+08 5.00-01 5.00-01",
        "   2   1 1.234+05 1.000-01 2.000-01",  # 9 columns for each value: the A-value reads as 1.234
        "   2   1 6.27+08 4.29-01",  # no upsilon at the second temperature
    ]
    flog = io.StringIO()
    _, _, upsilondict, collisiondf = readadasdata.read_adf04(write_hydrogen_adf04(tmp_path, rows), flog, 1e6, 1, 1)
    assert upsilondict == {(0, 1): pytest.approx(0.5)}
    assert collisiondf.columns == ["upper", "lower", "avalue", "upsilon"]
    assert collisiondf["avalue"].to_list() == [1e8, 6.27e8]
    assert "The reader skipped 1 collision rows that it could not parse." in flog.getvalue()
    assert "1 collision rows have no value at the selected temperature." in flog.getvalue()
    assert "WARNING" not in flog.getvalue()


def test_read_adf04_stops_if_no_collision_row_is_readable(tmp_path):
    """Rows that are not in the fixed columns give no value. The ion must not lose each transition silently."""
    for row in ("   1    2  6.27+08  4.29-01  5.29-01", "\t  2   1 6.27+08 4.29-01 5.29-01"):
        with pytest.raises(ValueError, match="could not parse any of the 1 collision rows"):
            read_hydrogen_adf04(tmp_path, [row])

    # A row that stops before the selected temperature is readable, so the reader does not stop. It gives a warning.
    rows = ["   2   1 6.27+08 4.29-01", "   1    2  6.27+08  4.29-01  5.29-01"]
    filepath = write_hydrogen_adf04(tmp_path, rows)
    assert readadasdata.read_adf04(filepath, io.StringIO(), 5000.0, 1, 1)[2] == {(0, 1): pytest.approx(0.429)}
    flog = io.StringIO()
    assert readadasdata.read_adf04(filepath, flog, 1e6, 1, 1)[2] == {}
    assert "WARNING: no collision row has an upsilon at the selected temperature" in flog.getvalue()


def test_append_adas_transition_rejects_equal_file_indices():
    """A transition from a level to itself stops the run in the reader, and the message names the file."""
    levels = [readadasdata.ADASEnergyLevel("a", 1, 1, 0, 0.0, 0.0, 1.0, 0)] * 2
    with pytest.raises(ValueError, match="same file index 2"):
        readadasdata.append_adas_transition(levels, [], 2, 2, 1e8, "x.adf04")


def test_standardise_config():
    """The reader converts an Eissner configuration, and it writes standard notation in lower case."""
    standardise = readadasdata._standardise_config  # ruff: ignore[private-member-access]
    assert standardise(" 522563524565 ", eissner_order="AUTOSTRUCTURE") == "2s22p63s23p6"
    assert standardise("522563524565606", eissner_order="AUTOSTRUCTURE") == "2s22p63s23p63d10"
    assert standardise("51151A", eissner_order="specification") == "1s15s1"
    # a label in a file with Eissner notation keeps its text
    assert standardise("2P", eissner_order="AUTOSTRUCTURE") == "2p"
    for config, expected in (
        ("3P6 3DA", "3p6 3d10"),
        ("3D54P", "3d54p"),
        ("4FA(3H)", "4f10(3H)"),
        ("3S2  3DA (4F)", "3s2  3d10 (4F)"),
        ("2P", "2p"),
        ("3S2 3P6 3D6 4S 4P", "3s2 3p6 3d6 4s 4p"),
        ("5s2", "5s2"),
        ("4P65S2(1S)", "4p65s2(1S)"),
        # the text after a parent term gets the same steps as the text before it
        ("3D6(5D)4DA", "3d6(5D)4d10"),
        ("(5D)4S", "(5D)4s"),
        # the column has 18 characters, so a term can have no closing parenthesis. A term can hold a term.
        ("3P63D5(4P)4S(5P", "3p63d5(4P)4s(5P"),
        ("3D6(5D", "3d6(5D"),
        ("((3P)4D)5S", "((3P)4D)5s"),
        # in a file with standard notation, a label that is also a valid Eissner configuration stays as it is
        ("21", "21"),
    ):
        assert standardise(config, eissner_order=None) == expected


def test_eissner_order_of_file():
    """The notation and the order of the shell characters are properties of the file, not of one level."""
    order_of_file = readadasdata._eissner_order_of_file  # ruff: ignore[private-member-access]
    flog = io.StringIO()
    # each level is its configuration and its total L
    assert order_of_file([("522563524565", 0), ("522563524555516", 1), ("21", 0)], "x.adf04", flog) == "AUTOSTRUCTURE"
    assert order_of_file([("3S2 3P6 3D6", 2), ("21", 0), ("3S2 3P6 3D5 4P1", 1)], "x.adf04", flog) is None
    assert order_of_file([("4p65s2(1S)", 0), ("5s2", 0), ("3D54P", 1), ("2P", 1)], "x.adf04", flog) is None
    assert order_of_file([], "x.adf04", flog) is None
    assert "WARNING" not in flog.getvalue()

    # 1s 4f has L = 3 and 1s 5s has L = 0, so the total L shows which shell "A" is
    assert order_of_file([("521", 0), ("51151A", 3), ("51151B", 0)], "x.adf04", flog) == "AUTOSTRUCTURE"
    assert order_of_file([("521", 0), ("51151A", 0), ("511510", 3)], "x.adf04", flog) == "specification"
    assert "1 of 3 levels agree with the AUTOSTRUCTURE order, 3 of 3 levels agree with the specification order" in (
        flog.getvalue()
    )

    # one level with a wrong L in the file does not stop the run
    flog = io.StringIO()
    levels = [("521", 0), ("51151A", 3), ("51151B", 0), ("51151B", 4)]
    assert order_of_file(levels, "x.adf04", flog) == "AUTOSTRUCTURE"
    assert "WARNING: The shells of 1 levels cannot give their total L." in flog.getvalue()

    # a blank field has no notation, so it does not count in the decision
    assert order_of_file([("521", 0), ("", 0), ("", 0)], "x.adf04", io.StringIO()) == "AUTOSTRUCTURE"
    # a blank field or a label is not a defective Eissner configuration
    flog = io.StringIO()
    assert order_of_file([("521", 0), ("51151A", 3), ("", 0)], "x.adf04", flog) == "AUTOSTRUCTURE"
    assert "WARNING: 1 levels have no Eissner configuration, for example ''." in flog.getvalue()

    # The digits of a defective Eissner configuration must not become the name of a level. "591" is 1s9.
    with pytest.raises(ValueError, match="cannot read the configuration '591'"):
        order_of_file([("521", 0), ("51151A", 3), ("591", 0)], "x.adf04", io.StringIO())


def test_read_adf04_uses_the_shell_order_of_the_file(tmp_path):
    """A file that follows the order of the specification (0=4f, A=5s) gets the level names of that order."""
    levels = [
        "    1 521                (1)0( 0.0)        0.",
        "    2 51151A             (1)0( 0.0)    82303.",
        "    3 511510             (1)3( 3.0)    92303.",
    ]
    filepath = write_hydrogen_adf04(tmp_path, ["   2   1 6.27+08 4.29-01 5.29-01"], levels=levels)
    energylevels = readadasdata.read_adf04(filepath, io.StringIO(), 5000.0, 1, 1)[1]
    assert [level.levelname.split("_")[0] for level in energylevels] == ["1s2", "1s15s1", "1s14f1"]


def test_parse_ion_handlers_accepts_a_renamed_handler():
    """A file from before a handler rename still names the old handler, so the parser must map it."""
    from artisatomic.iondata import handlers
    from artisatomic.ionhandlers import parse_ion_handlers
    from artisatomic.ionhandlers import renamed_handlers

    assert parse_ion_handlers([[38, [[1, "qub_data"]]]]) == [(38, [(1, "adas")])]
    # the ion stage gives the new name of the old cobalt handler
    old_cobalt = [[27, [[1, "qub_cobalt"], [2, "qub_cobalt"], [3, "qub_cobalt"], [4, "qub_cobalt"]]]]
    assert parse_ion_handlers(old_cobalt) == [(27, [(1, "cmfgen"), (2, "cmfgen_qubphixs"), (3, "adas"), (4, "adas")])]
    # a current name passes through unchanged
    assert parse_ion_handlers([[27, [[2, "cmfgen_qubphixs"]]]]) == [(27, [(2, "cmfgen_qubphixs")])]
    # every alias must point at a handler that read_ion_data() can dispatch
    assert {newname for newnames in renamed_handlers.values() for newname in newnames.values()} <= set(handlers)


def test_read_adas_sr1():
    """Sr I is a complete adf04 file: the collision block ends with a "-1" row and a comment block."""
    flog = io.StringIO()
    ionization_energy_ev, energylevels, transitions, upsilondict = readadasdata.read_adas_levels_and_transitions(
        38, 1, flog, argparse.Namespace(electrontemperature=5000.0)
    )
    assert abs(ionization_energy_ev - 5.694867) < 1e-5
    assert len(energylevels) == 57
    # the file holds 1596 collision rows between the temperature header and the "-1" row
    assert len(upsilondict) == 1596
    assert len(transitions) == 1372
    assert energylevels[0].levelname.startswith("4p65s2")


def test_write_adata_level_comment():
    """The level comment is the level's name, with no padding.

    artistools reads the comment as `line.split(maxsplit=4)[4].strip("'")`, which strips quotes but
    not whitespace, so the reported level name would contain any padding written here. The
    writer puts the name as the reader gave it.
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
    write_adata(buf, 26, 2, dfhillier, 10.0, [0], io.StringIO())
    hillier_line = buf.getvalue().splitlines()[1]
    assert hillier_line.endswith(" someion_gs")
    assert hillier_line.split(maxsplit=4)[4] == "someion_gs"

    # the writer puts a level name with spaces unchanged, and it reads back as everything after
    # the fourth field
    spacedlevelname = "3Pe index 1 '2s22p2'"
    dfspaced = leveltuples_to_pldataframe(
        pl.DataFrame({"levelname": [spacedlevelname], "energyabovegsinpercm": [0.0], "g": [9.0]})
    )
    buf = io.StringIO()
    write_adata(buf, 8, 1, dfspaced, 13.6, [0], io.StringIO())
    spaced_line = buf.getvalue().splitlines()[1]
    assert spaced_line.endswith(" " + spacedlevelname)
    assert spaced_line.split(maxsplit=4)[4] == spacedlevelname


# the format string that FAC writes one transition row with. The reader cuts its fields.
FAC_TRANSITION_FORMAT = "%6d %2d %6d %2d %13.6E %13.6E %13.6E %13.6E"


def fac_levels_header(code: str, nlev: int) -> list[str]:
    """Return the header of an FAC or cFAC level file, with its blank row and its NLEV row."""
    return [
        f"{code} 1.1.5[60.0.0]",
        "Endian\t= 0",
        "TSess\t= 1693231739",
        "Type\t= 1",
        "Verbose\t= 1",
        "La Z\t=  57.0",
        "NBlocks\t= 1",
        "E0\t= 0, -2.30803280E+05",
        "",
        "NELE\t= 56",
        f"NLEV\t= {nlev}",
        "  ILEV  IBASE    ENERGY       P   VNL   2J",
    ]


def fac_lines_header(code: str, ntrans: int) -> list[str]:
    """Return the header of an FAC or cFAC transition file, with its blank row and its NTRANS row."""
    return [
        f"{code} 1.1.5[60.0.0]",
        "Endian\t= 0",
        "TSess\t= 1693231739",
        "Type\t= 2",
        "Verbose\t= 1",
        "La Z\t=  57.0",
        "NBlocks\t= 1",
        "",
        "NELE\t= 56",
        f"NTRANS\t= {ntrans}",
        "MULTIP\t= -1",
        "GAUGE\t= 2",
        "MODE\t= 1",
    ]


def write_fac_fixture(tmp_path):
    """Write one FAC and one cFAC pair of ascii files, in the fixed-width layout of each code.

    Every field sits at the character positions that the code writes. The transition rows come
    from FAC's own format string, so the fixture cannot put a character where FAC never puts one.
    The header is the header of a real file. The tables carry these cases:

    - an occupation of 1, and an occupation of 10 or more;
    - two levels of one energy;
    - a negative A beside a positive monopole, and a positive A beside a negative monopole;
    - a file index of six digits, in fac_wide.tr.asc.
    """

    def row(width: int, fields: list[tuple[str, int, int]]) -> str:
        chars = [" "] * width
        for text, start, end in fields:
            chars[start:end] = list(f"{text:>{end - start}}")
        return "".join(chars).rstrip()

    # FAC levels: Ilev (0,7) Energy_ev (14,30) P (30,31) 2J (38,43) Configs (76,125)
    faclevels = fac_levels_header("FAC", 3)
    faclevels += [
        row(125, [("0", 0, 7), ("0.0000000000E+00", 14, 30), ("0", 30, 31), ("0", 38, 43), ("4f1.6s1", 76, 84)]),
        row(125, [("1", 0, 7), ("5.0000000000E+00", 14, 30), ("1", 30, 31), ("4", 38, 43), ("4f14.6s2", 76, 85)]),
        # the same energy as the level before it, so the stable sort must keep the file's order
        row(125, [("2", 0, 7), ("5.0000000000E+00", 14, 30), ("0", 30, 31), ("2", 38, 43), ("5d1.6s1", 76, 84)]),
        "",  # FAC writes a blank line after the table
    ]
    (tmp_path / "fac.lev.asc").write_text("\n".join(faclevels) + "\n")

    # FAC transitions: Upper (0,6) Lower (10,16) A (48,61), from FAC_TRANSITION_FORMAT
    factransrows = [
        (1, 4, 0, 4, 1.754126e00, 9.521099e-02, 3.14e07, -2.104985e00),
        # a negative monopole sits beside a positive A, and must stay out of the A column
        (2, 6, 0, 6, 1.628146e00, 7.481372e-03, 1.0e06, -6.124631e-01),
        (2, 2, 1, 4, 1.541308e00, 7.456976e-04, -7.77e04, 1.987342e-01),
    ]
    factrans = fac_lines_header("FAC", len(factransrows))
    factrans += [FAC_TRANSITION_FORMAT % fields for fields in factransrows]
    factrans += [""]
    (tmp_path / "fac.tr.asc").write_text("\n".join(factrans) + "\n")

    # a file index of six digits fills its whole field, so no space separates it from the field
    # on its left. It goes into a file of its own, because the levels file above has three levels
    facwide = fac_lines_header("FAC", 1)
    facwide += [FAC_TRANSITION_FORMAT % (999999, 4, 123456, 4, 1.5e00, 1.0e-02, 1.0e05, 1.0e-01), ""]
    (tmp_path / "fac_wide.tr.asc").write_text("\n".join(facwide) + "\n")

    # cFAC levels: the configuration and a second field share the column at (43,150)
    cfaclevels = fac_levels_header("cFAC", 2)
    cfaclevels += [
        row(
            150,
            [("0", 0, 7), ("0.0000000000E+00", 14, 30), ("0", 30, 31), ("0", 38, 43), ("4f1 6s1   4f+1(3)3", 45, 63)],
        ),
        row(
            150,
            [("1", 0, 7), ("5.0000000000E+00", 14, 30), ("1", 30, 31), ("4", 38, 43), ("4f14 6s2   4f+14(0)0", 45, 65)],
        ),
        "",
    ]
    (tmp_path / "cfac.lev.asc").write_text("\n".join(cfaclevels) + "\n")

    # cFAC transitions: Upper (0,6) Lower (10,16) A (61,75)
    cfactrans = fac_lines_header("cFAC", 1)
    cfactrans += [
        row(89, [("1", 0, 6), ("0", 10, 16), ("3.1400000E+07", 61, 75), ("1.0E-03", 75, 89)]),
        "",
    ]
    (tmp_path / "cfac.tr.asc").write_text("\n".join(cfactrans) + "\n")


def test_readfacdata_parses_the_fac_and_cfac_column_layouts(tmp_path):
    """The reader cuts the same values out of the FAC and the cFAC fixed-width layouts.

    Both layouts give the same level table here, because the two files describe the same ion.
    The transition table shows that every field keeps its own window. A negative monopole stays
    out of the A column. A negative A keeps its sign, and a six-digit file index stays whole.
    """
    write_fac_fixture(tmp_path)

    # "6s1" loses its occupation of 1, "4f14" and "6s2" keep theirs, and a dot becomes a space
    expected_levels = {
        "fac": [(0, "4f 6s", 0, 1, 0.0), (1, "4f14 6s2", 1, 5, 5.0), (2, "5d 6s", 0, 3, 5.0)],
        "cfac": [(0, "4f 6s", 0, 1, 0.0), (1, "4f14 6s2", 1, 5, 5.0)],
    }
    for name, expected in expected_levels.items():
        dflevels = readfacdata.GetLevels(tmp_path / f"{name}.lev.asc")
        # the whole table, so a parse that drops or adds a level fails here
        assert list(dflevels.select("Ilev", "Config", "P", "g", "Energy_ev").iter_rows()) == expected, name
        assert dflevels["energypercm"].to_list() == [energy / hc_in_ev_cm for _, _, _, _, energy in expected], name

    dflines = readfacdata.GetLines(tmp_path / "fac.tr.asc")
    assert list(dflines.select("Upper", "Lower", "A").iter_rows()) == [
        (1, 0, 3.14e7),
        (2, 0, 1.0e6),  # the negative monopole beside it stays out of the A column
        (2, 1, -7.77e4),  # a negative A keeps its sign
    ]

    # a file index of six digits touches the field on its left, and must still parse
    assert list(readfacdata.GetLines(tmp_path / "fac_wide.tr.asc").select("Upper", "Lower", "A").iter_rows()) == [
        (999999, 123456, 1.0e5)
    ]

    assert list(readfacdata.GetLines(tmp_path / "cfac.tr.asc").select("Upper", "Lower", "A").iter_rows()) == [
        (1, 0, 3.14e7)
    ]

    # a file that neither code wrote must not parse as either of them
    (tmp_path / "other.lev.asc").write_text("SOMETHINGELSE 1.0\n" + "\n".join(f"h{i}" for i in range(1, 12)) + "\n")
    with pytest.raises(ValueError, match="names neither FAC nor cFAC"):
        readfacdata.GetLevels(tmp_path / "other.lev.asc")


def test_readfacdata_stops_on_a_file_that_holds_fewer_rows_than_its_header_declares(tmp_path):
    """A copy of a shared-drive file can stop between two rows, which every other check accepts.

    FAC writes the row count in the header, as NLEV or NTRANS. The reader compares it with the
    count of rows that it read. Without the comparison, the ion goes to the output with a part of
    its levels or its transitions and no message.
    """
    write_fac_fixture(tmp_path)

    # the same tables, with a header that declares many more rows than the file holds
    for name, key, declared in (("fac.lev", "NLEV", 3), ("fac.tr", "NTRANS", 3)):
        text = (tmp_path / f"{name}.asc").read_text()
        (tmp_path / f"short_{name}.asc").write_text(text.replace(f"{key}\t= {declared}", f"{key}\t= 1200"))

    with pytest.raises(ValueError, match="declares NLEV = 1200 but holds 3 rows"):
        readfacdata.GetLevels(tmp_path / "short_fac.lev.asc")

    with pytest.raises(ValueError, match="declares NTRANS = 1200 but holds 3 rows"):
        readfacdata.GetLines(tmp_path / "short_fac.tr.asc")

    # a header with no count at all means that the file stopped inside its own header
    noheader = (tmp_path / "fac.lev.asc").read_text().replace("NLEV\t= 3", "NOTNLEV\t= 3")
    (tmp_path / "nocount.lev.asc").write_text(noheader)
    with pytest.raises(ValueError, match="no NLEV line"):
        readfacdata.GetLevels(tmp_path / "nocount.lev.asc")


def test_readfacdata_maps_file_indices_to_energy_sorted_ids(tmp_path):
    """The FAC levels sort by energy, and the transitions still name their levels by the file's Ilev.

    The map that read_levels_data() returns carries the transitions onto the sorted ids. The sort
    is stable, so two levels of one energy keep the order of the file.
    """
    write_fac_fixture(tmp_path)
    dflevels = readfacdata.GetLevels(tmp_path / "fac.lev.asc")

    energy_levels, levelid_of_fileindex = readfacdata.read_levels_data(dflevels)

    assert [level.levelname for level in energy_levels] == [
        "4f 6s Ilev=0",
        "4f14 6s2 Ilev=1",
        "5d 6s Ilev=2",
    ]
    assert levelid_of_fileindex == {0: 0, 1: 1, 2: 2}

    dflines = readfacdata.GetLines(tmp_path / "fac.tr.asc")
    transitions = readfacdata.read_lines_data(dflines, levelid_of_fileindex)
    assert [(tr.lowerlevel, tr.upperlevel, tr.A) for tr in transitions] == [
        (0, 1, 3.14e7),
        (0, 2, 1.0e6),
        (1, 2, -7.77e4),
    ]

    # a level above the ionisation energy leaves the level list, so its transitions go too
    flog = io.StringIO()
    dfkeptlines = drop_transitions_of_levels(dflines, "Lower", "Upper", {2}, "The FAC transitions file", flog)
    transitions = readfacdata.read_lines_data(dfkeptlines, levelid_of_fileindex)
    assert [(tr.lowerlevel, tr.upperlevel) for tr in transitions] == [(0, 1)]
    assert "skipped 2 transitions" in flog.getvalue()

    # a transition that names an Ilev the levels file does not have means the two files disagree
    with pytest.raises(ValueError, match="names file index 99"):
        readfacdata.read_lines_data(dflines.with_columns(pl.col("Upper").replace(1, 99)), levelid_of_fileindex)


def test_readfacdata_warns_on_an_ion_whose_transitions_are_all_above_the_ionisation_energy(tmp_path, monkeypatch):
    """66DyIII_calib skips every one of its 1873047 transitions, and writes an ion with none.

    The writer accepts an ion with no transition, so the reader writes a warning and continues.
    readlisbondata gives the same warning for the same case.
    """

    def levelrow(ilev: int, energy_ev: float, twoj: int) -> str:
        chars = [" "] * 125
        for text, start, end in (
            (str(ilev), 0, 7),
            (f"{energy_ev:.10E}", 14, 30),
            ("0", 30, 31),
            (str(twoj), 38, 43),
            ("5d1", 76, 79),
        ):
            chars[start:end] = list(f"{text:>{end - start}}")
        return "".join(chars).rstrip()

    # La II ionises at 11.06 eV, so the third level is above the ionisation energy
    iondir = tmp_path / "OptimizedFAC_lanthanides_calibrated" / "57LaII_calib"
    iondir.mkdir(parents=True)
    levels = fac_levels_header("FAC", 3)
    levels += [levelrow(0, 0.0, 0), levelrow(1, 5.0, 2), levelrow(2, 50.0, 4), ""]
    (iondir / "57LaII_calib.lev.asc").write_text("\n".join(levels) + "\n")

    # every transition names the level above the ionisation energy, so every one goes
    transitionrows = [
        (2, 4, 0, 0, 5.0e01, 1.0e-02, 1.0e05, 1.0e-01),
        (2, 4, 1, 2, 4.5e01, 1.0e-02, 2.0e05, 1.0e-01),
    ]
    lines = fac_lines_header("FAC", len(transitionrows))
    lines += [FAC_TRANSITION_FORMAT % fields for fields in transitionrows]
    lines += [""]
    (iondir / "57LaII_calib.tr.asc").write_text("\n".join(lines) + "\n")

    monkeypatch.setenv("ARTISATOMIC_FAC_PATH", str(tmp_path))
    flog = io.StringIO()
    _, energy_levels, transitions = readfacdata.read_levels_and_transitions(57, 2, flog)

    # the ion keeps its bound levels and goes to the output with no line
    assert len(energy_levels) == 2
    assert transitions == []
    assert "The reader skipped all 2 transitions" in flog.getvalue()
    assert "The reader dropped 1 levels that are above the ionisation energy." in flog.getvalue()


def test_readfacdata_stops_on_an_ion_whose_levels_are_all_above_the_ionisation_energy(tmp_path, monkeypatch):
    """An ion with no bound level stops the run, as it does for the Lisbon reader.

    The two readers share split_levels_above_ionization(), which owns the check. Such an ion
    would go to the output with no level and no line.
    """

    def levelrow(ilev: int, energy_ev: float, twoj: int) -> str:
        chars = [" "] * 125
        for text, start, end in (
            (str(ilev), 0, 7),
            (f"{energy_ev:.10E}", 14, 30),
            ("0", 30, 31),
            (str(twoj), 38, 43),
            ("5d1", 76, 79),
        ):
            chars[start:end] = list(f"{text:>{end - start}}")
        return "".join(chars).rstrip()

    # La II ionises at 11.06 eV, so both levels are above the ionisation energy
    iondir = tmp_path / "OptimizedFAC_lanthanides_calibrated" / "57LaII_calib"
    iondir.mkdir(parents=True)
    levels = fac_levels_header("FAC", 2)
    levels += [levelrow(0, 50.0, 0), levelrow(1, 60.0, 2), ""]
    (iondir / "57LaII_calib.lev.asc").write_text("\n".join(levels) + "\n")

    monkeypatch.setenv("ARTISATOMIC_FAC_PATH", str(tmp_path))
    with pytest.raises(ValueError, match="Every one of the 2 levels"):
        readfacdata.read_levels_and_transitions(57, 2, io.StringIO())


def test_path_for_log_renders_a_path_relative_to_a_directory():
    """A log file must name a data file the same way on every machine, so no path is absolute.

    The default directory is the repository root. A reader that passes its own data folder gets a
    shorter path, and a path outside that folder still falls back to the repository root.
    """
    from artisatomic.base import path_for_log
    from artisatomic.readhillierdata import hillier_datadir

    oscfile = hillier_datadir / "atomic_21jun23" / "COB" / "II" / "19apr23" / "osc_data"

    # the CMFGEN reader names its files relative to its own data folder
    assert path_for_log(oscfile, relative_to=hillier_datadir) == "atomic_21jun23/COB/II/19apr23/osc_data"

    # with no folder given, the same file is relative to the repository root
    assert path_for_log(oscfile) == "atomic-data-hillier/atomic_21jun23/COB/II/19apr23/osc_data"

    # a file outside the given folder falls back to the repository root
    kuruczfile = PYDIR / ".." / "atomic-data-kurucz" / "gfall.dat"
    assert path_for_log(kuruczfile, relative_to=hillier_datadir) == "atomic-data-kurucz/gfall.dat"

    # a path outside the repository comes back unchanged, because no base fits it
    assert path_for_log("/nonexistent/elsewhere/osc_data") == "/nonexistent/elsewhere/osc_data"


def test_path_for_log_keeps_a_symlinked_data_folder_short(tmp_path):
    """A data folder is often a symbolic link to another disk, and the log path must stay short.

    The CMFGEN data is hundreds of megabytes, so a checkout often links it to another disk. A
    comparison that follows the link puts the target outside every base, and the log then holds
    the absolute path of that disk.
    """
    from artisatomic.base import path_for_log

    datadir = tmp_path / "repo" / "atomic-data-hillier"
    external = tmp_path / "external" / "atomic_21jun23"
    (external / "COB" / "II").mkdir(parents=True)
    (external / "COB" / "II" / "osc_data").touch()
    datadir.mkdir(parents=True)
    (datadir / "atomic_21jun23").symlink_to(external)

    oscfile = datadir / "atomic_21jun23" / "COB" / "II" / "osc_data"
    assert oscfile.is_file()  # the link resolves, so the reader can open it

    assert path_for_log(oscfile, relative_to=datadir) == "atomic_21jun23/COB/II/osc_data"

    # the same holds when the data folder itself is the link
    linkeddatadir = tmp_path / "repo" / "linked-data"
    linkeddatadir.symlink_to(tmp_path / "external")
    assert path_for_log(linkeddatadir / "atomic_21jun23" / "COB", relative_to=linkeddatadir) == "atomic_21jun23/COB"


def test_scan_file_lines_reads_each_compressed_form(tmp_path):
    """Every compression form of a file gives the same lines, and skip_lines drops the header."""
    from xopen import xopen

    text = "first\nsecond\n\nfourth\n"
    plainpath = tmp_path / "lines.txt"
    plainpath.write_text(text)
    for suffix in (".gz", ".xz", ".zst"):
        with xopen(f"{plainpath}{suffix}", "wt", encoding="utf-8") as fout:
            fout.write(text)

    # polars reads a plain, a gzip and a zstd file itself, and xopen decompresses the xz form
    for suffix in ("", ".gz", ".xz", ".zst"):
        lines = scan_file_lines(f"{plainpath}{suffix}").collect()["line"].to_list()
        assert lines == ["first", "second", None, "fourth"], suffix
        assert scan_file_lines(f"{plainpath}{suffix}", skip_lines=2).collect()["line"].to_list() == [None, "fourth"]

    # the caller names the plain file, so a name with no file of any form is an error
    with pytest.raises(FileNotFoundError):
        scan_file_lines(tmp_path / "notafile.txt")


def test_rewrite_file_as_utf8_converts_an_iso_8859_1_file(tmp_path):
    """rewrite_file_as_utf8() converts a CMFGEN iso-8859-1 file to utf-8, and leaves a utf-8 file unchanged."""
    text = "Reference: Galav\u00eds M.E. 1998\nsecond line\n"
    filepath = tmp_path / "osc_data"
    filepath.write_bytes(text.encode("iso-8859-1"))

    assert rewrite_file_as_utf8(filepath)
    assert filepath.read_bytes() == text.encode("utf-8")
    assert filepath.read_text(encoding="utf-8") == text

    # the file is utf-8 now, so a second call has nothing to do and says so
    assert not rewrite_file_as_utf8(filepath)
    assert filepath.read_bytes() == text.encode("utf-8")


def test_parse_transition_lines_reads_both_layouts():
    """parse_transition_lines() reads a CMFGEN oscillator table with or without the last two columns."""
    withbars = "3d6_a6De[9/2]           -3d6_a4De[7/2]      1.693E-08   2.090D-03   2.599E+05     1-   2   |    |    |"
    withid = "3d6_a6De[9/2]           -3d6_a4De[7/2]      1.693E-08   2.090E-03   2.599E+05     1-   2       7"
    dftransitions = rhd.parse_transition_lines(pl.LazyFrame({"line": [withbars, withid]}), Path("test_osc"))

    assert dftransitions.height == 2
    assert dftransitions["namefrom"].to_list() == ["3d6_a6De[9/2]"] * 2
    assert dftransitions["nameto"].to_list() == ["3d6_a4De[7/2]"] * 2
    # the files write an exponent as D as well as E
    assert dftransitions["A"].to_list() == [2.090e-3, 2.090e-3]
    assert dftransitions["i"].to_list() == [1, 1]
    assert dftransitions["j"].to_list() == [2, 2]
    # the row with no id column gets its position as its id, and the other keeps the file's id
    assert dftransitions["hilliertransitionid"].to_list() == [1, 7]
    assert dftransitions.schema == rhd.hillier_transition_schema


def test_parse_transition_lines_stops_at_a_second_table():
    """The parser reads only the first table, and a table with no line at all gives an empty frame."""
    transition = "a-b   1.0E-01   2.0E-01   3.0E+03     1-   2       1"
    lines = [transition, "   Oscillator strengths for Fe2", transition]

    assert rhd.parse_transition_lines(pl.LazyFrame({"line": lines}), Path("test_osc")).height == 1
    # a file that ends at the table title leaves no line to read
    emptyframe = pl.LazyFrame({"line": []}, schema={"line": pl.String})
    assert rhd.parse_transition_lines(emptyframe, Path("test_osc")).height == 0


def test_parse_gfall_collapses_label_whitespace(monkeypatch):
    r"""Kurucz labels have fixed-width padding, so the parser must collapse their whitespace runs.

    Expr.replace() swaps whole values that equal a literal. `.replace(r"\s+", " ")` on a label
    column therefore matched nothing, and every run survived into the level names in adata.txt.
    Only `.str.replace_all()` treats the argument as a pattern.
    """
    # the sample, so the answer does not depend on the unpacked corpus or on ARTISATOMIC_TESTMODE
    monkeypatch.setattr(readkuruczdata, "kuruczdatapath", PYDIR / ".." / "atomic-data-kurucz" / "test_sample")
    gfall = readkuruczdata.parse_gfall(str(readkuruczdata.find_gfall(38, 0))).collect()

    labels = pl.concat([gfall["label_lower"], gfall["label_upper"]]).unique().to_list()
    assert labels, "expected some labels for Sr I"
    # no run of two or more spaces survives, and no label keeps padding at either end
    assert not [label for label in labels if "  " in label]
    assert all(label == label.strip() for label in labels)
    # the collapse must join the parts rather than delete the separator ('s4d  1D' -> 's4d 1D')
    assert any(" " in label for label in labels)


def test_get_level_valence_n():
    """Each reader's level names yield the valence electron's principal quantum number."""
    # each handler has its own level-name format and parser
    assert readkuruczdata.get_level_valence_n("s5p  3P,enpercm=14276.381,j=0.0") == 5
    # extendedatoms labels write the first orbital with its n and no count: 5s 11p, not 5s1 1p
    assert readkuruczdata.get_level_valence_n("5s11p 1P,enpercm=44366.42,j=1.0") == 11
    assert readkuruczdata.get_level_valence_n("5s13s 1S,enpercm=44366.42,j=0.0") == 13
    assert readkuruczdata.get_level_valence_n("3d104s 2S,enpercm=0.0,j=0.5") == 4
    assert readtanakajpltdata.get_level_valence_n("2,even,{  4d- 3  4d+ 1  5s+ 1 }") == 5
    # the configuration column glues two orbitals: 4p 5s
    assert (
        readtanakajpltdata.get_level_valence_n("16,odd,3s2_3p6_3d10_4s2_4p5s   3s(2).3p(6).3d(10).4s(2).4p.5s_3P") == 5
    )
    # a two-digit n has one leading space
    assert readtanakajpltdata.get_level_valence_n("6,even,{  6p- 2  6p+ 4 10s+ 1 }") == 10
    # the closed 5s(2).5p(6) shells follow the open 4f shell in the LS term; the configuration
    # column gives the valence orbital
    assert (
        readtanakajpltdata.get_level_valence_n("1,even,4s2_4p6_4f2 4s(2).4p(6).4d(10)1S0_1S.4f(2)3H1_3H.5s(2).5p(6)_3H")
        == 4
    )

    # the 2024 Ge-sequence JPLT files append an LS-coupled term string after the configuration
    assert (
        readtanakajpltdata.get_level_valence_n(
            "1,even,3s2_3p6_3d10_4s2_4p2                  3s(2).3p(6).3d(10).4s(2).4p(2)_3P"
        )
        == 4
    )
    # parent terms may follow a shell ("4p(3)4S") or stand as their own segment ("3P2_3P.7p")
    assert readtanakajpltdata.get_level_valence_n("6,odd,3s2_3p6_3d10_4s4p3   3s(2).3p(6).3d(10).4s.4p(3)4S_5S") == 4
    # the valence orbital comes from the configuration column, so a name that gives the LS term
    # alone gives None. No level name of data_v2.1 has that shape
    assert (
        readtanakajpltdata.get_level_valence_n("60,odd,3d(10)1S0.4s(2).4p(6).4d(10)1S0_1S.5s(2).5p(2)3P2_3P.7p_4D")
        is None
    )
    assert readtanakajpltdata.get_level_valence_n("1,even,5d(10).6s(2).6p(6).7s(2)_1S") is None
    assert readfloers25data.get_level_valence_n("4f10") == 4
    assert readfloers25data.get_level_valence_n("4f9.6s") == 6
    assert readfloers25data.get_level_valence_n("5s2.5p5") == 5
    assert readfacdata.get_level_valence_n("4f9 6s1") == 6
    assert readfacdata.get_level_valence_n("4f10") == 4

    # the floers25 and fac level names carry a suffix that makes the name unique, which the parser must ignore
    assert readfloers25data.get_level_valence_n("4f10 J=8 index=0") == 4
    assert readfloers25data.get_level_valence_n("4f9.6s J=15/2 index=137") == 6
    assert readfloers25data.get_level_valence_n("5s2.5p5 J=3/2 index=2") == 5
    assert readfacdata.get_level_valence_n("4f9 6s1 Ilev=42") == 6
    assert readfacdata.get_level_valence_n("4f10 Ilev=0") == 4
    assert readadasdata.get_level_valence_n("3d7_4Fe[9/2]_id=1") == 3
    assert readadasdata.get_level_valence_n("5s2_1Se[0/2]_id=1") == 5
    # the count of the previous orbital and a two-digit n: 5s1 11s1, not 5s11 1s
    assert readadasdata.get_level_valence_n("5s111s1_2Se[1/2]_id=1") == 11
    assert readadasdata.get_level_valence_n("3d24s_2Se[1/2]_id=1") == 4
    # an adf04 label separates its shells with a space, so the last run holds n alone
    assert readadasdata.get_level_valence_n("3S2 3P6 3D6 4S 4P_5D0[8]_id=1") == 4

    # a Kurucz label can end in a parent term and an odd-parity mark after the valence orbital
    assert readkuruczdata.get_level_valence_n("4f3(4I*)6s6p*(3P*) 5I,enpercm=12345.0,j=2.5") == 6
    # a Kurucz label glues the electron count of a shell to the n of the next: s25p is 5s2 5p
    assert readkuruczdata.get_level_valence_n("f36s *5I,enpercm=0.0,j=4.0") == 6
    assert readkuruczdata.get_level_valence_n("s25p 3P,enpercm=14276.381,j=0.0") == 5
    assert readkuruczdata.get_level_valence_n("d25s 4F,enpercm=1.0,j=1.5") == 5
    assert readkuruczdata.get_level_valence_n("31s 2S,enpercm=1.0,j=0.5") == 31
    assert readkuruczdata.get_level_valence_n("f125d 2D,enpercm=1.0,j=1.5") == 5
    assert readkuruczdata.get_level_valence_n("f145d 2D,enpercm=1.0,j=1.5") == 5
    assert readkuruczdata.get_level_valence_n("d105s 2S,enpercm=1.0,j=0.5") == 5
    assert readkuruczdata.get_level_valence_n("s10d 2D,enpercm=1.0,j=1.5") == 10
    assert readkuruczdata.get_level_valence_n("d5p' 3P,enpercm=37292.106,j=0.0") == 5

    # a name with no readable n gives None, never a guessed n. match_hydrogenic_phixs() then
    # skips the level and logs it. A guess would give a cross section of the wrong size
    assert readkuruczdata.get_level_valence_n(",enpercm=12345.0,j=2.5") is None
    assert readkuruczdata.get_level_valence_n("N(1S)2H 1,enpercm=76765.9,j=4.5") is None
    assert readadasdata.get_level_valence_n("_id=1") is None
    assert readadasdata.get_level_valence_n("(3P)_1Se[0/2]_id=1") is None
    assert readtanakajpltdata.get_level_valence_n("1,even,{  6h+ 1 }") == 6
    assert readtanakajpltdata.get_level_valence_n("1,even,{  h+ 1 }") is None

    # the Floers+25 and FAC names accept any orbital letter, and give None for a bad token
    assert readfloers25data.get_level_valence_n("4f12.6h1 J=5 index=17") == 6
    assert readfacdata.get_level_valence_n("4f12 6h1 Ilev=17") == 6
    assert readfloers25data.get_level_valence_n("4f12.h J=5 index=17") is None
    assert readfloers25data.get_level_valence_n("4f12.66 J=5 index=17") is None


def test_adf04_float_reads_every_exponent_form():
    """adf04 writes 1.23-04 for 1.23e-04, and the reader must not break a sign or an E that is there."""
    lines = pl.DataFrame({"line": ["1.23-04", "4.66+04", "-1.23-04", "1.23E-04", "1.23", "-2.5"]})
    values = lines.select(readadasdata.adf04_float(0, 8)).to_series().to_list()
    assert values == [1.23e-4, 4.66e4, -1.23e-4, 1.23e-4, 1.23, -2.5]


def test_leveltuples_to_pldataframe_empty_list():
    """An ion with no levels gets the columns the writer reads, not an empty frame with only ids."""
    from artisatomic.base import leveltuples_to_pldataframe

    dflevels = leveltuples_to_pldataframe([])
    assert dflevels.height == 0
    assert set(dflevels.columns) >= {"levelid", "levelname", "energyabovegsinpercm", "g", "parity"}


def test_match_hydrogenic_phixs_skips_unreadable_and_out_of_range_n():
    """A level with no readable n, or with n outside the hydrogenic tables, gets no estimate.

    The hydrogenic tables cover n up to max_hyd_gaunt_n (30). A level outside them gave a
    KeyError before.
    """
    rhd.read_hyd_phixsdata()
    assert rhd.max_hyd_gaunt_n == 30

    ionization_energy_ev = 4 * rhd.ryd_to_ev
    dflevels = pl.DataFrame(
        {
            "levelid": [0, 1, 2],
            "energyabovegsinpercm": [0.0, 1000.0, 2000.0],
            "g": [2.0, 2.0, 2.0],
            "levelname": [
                "s1s  1S,enpercm=0.0,j=0.5",  # n=1: gets a table
                "31s 2S,enpercm=1000.0,j=0.5",  # n=31: outside the tables
                "N(1S)2H 1,enpercm=2000.0,j=4.5",  # no readable n
            ],
        }
    )
    args = phixs_args(nlevels_hydrogenic_for_unknown_phixs=100)
    flog = io.StringIO()
    crosssections, targetfractions, thresholds = match_hydrogenic_phixs(
        atomic_number=2,
        energy_levels=dflevels,
        ionization_energy_ev=ionization_energy_ev,
        ion_handler="kurucz",
        get_level_valence_n=readkuruczdata.get_level_valence_n,
        args=args,
        flog=flog,
    )

    assert targetfractions == [[(0, 1.0)], [], []]
    assert not np.isnan(thresholds[0])
    assert np.isnan(thresholds[1])
    assert np.isnan(thresholds[2])
    assert crosssections[0][0] > 0.0
    assert np.all(crosssections[1] == 0.0)
    assert np.all(crosssections[2] == 0.0)

    logtext = flog.getvalue()
    assert "n=31" in logtext
    assert "level name 'N(1S)2H 1,enpercm=2000.0,j=4.5' has no principal quantum number" in logtext


def test_readkuruczdata_drops_repeated_lines_but_not_merged_ones(monkeypatch):
    """Gfall repeats some lines, and separately lists distinct lines that share a level pair.

    A repeat has the same labels at both ends. It is one line given twice, once at its observed
    wavelength and once at the Ritz one. ARTIS adds the A values of two rows that share a level
    pair (input.cc), so both rows together would double the line.

    Rows that share a level pair with DIFFERENT labels are separate transitions whose levels the
    (energy, J) key merged into one. Sr I has 785 of those, two of them strong, and a drop would
    delete real lines. The reader leaves them for ARTIS to combine.
    """
    from artisatomic import readkuruczdata

    # use the committed sample, so the answer does not depend on the unpacked corpus
    monkeypatch.setattr(readkuruczdata, "kuruczdatapath", PYDIR / ".." / "atomic-data-kurucz" / "test_sample")

    gfall = readkuruczdata.parse_gfall(str(readkuruczdata.find_gfall(39, 1))).collect()
    levelkey = ["energyabovegsinpercm_lower", "j_lower", "energyabovegsinpercm_upper", "j_upper"]
    # Y II holds one repeat: s5p z3P -> d5d g3D at both 241.7267 and 241.7308 nm
    assert gfall.height - gfall.unique(levelkey).height == 1
    assert gfall.height - gfall.unique([*levelkey, "label_lower", "label_upper"]).height == 1

    monkeypatch.setattr(readkuruczdata, "kuruczdatapath", PYDIR / ".." / "atomic-data-kurucz" / "test_sample")

    _, dflevels, transitions = readkuruczdata.read_levels_and_transitions(39, 2, io.StringIO())

    # the repeat is gone, and no level pair keeps two rows for ARTIS to add up
    assert transitions.height == gfall.height - 1
    assert transitions.group_by(["lowerlevel", "upperlevel"]).len().filter(pl.col("len") > 1).is_empty()

    # Sr II in the zztar layout has five groups that match on the levels AND both labels. Their
    # loggf differs, by as much as -2.848 against -1.547. Those are separate lines. To keep only the
    # first would discard the stronger of the two, so the strength is part of the key.
    srii = readkuruczdata.parse_gfall(
        str(PYDIR / ".." / "atomic-data-kurucz" / "test_sample" / "zztar" / "gf3801.all.zst")
    ).collect()
    withlabels = [*levelkey, "label_lower", "label_upper"]
    assert srii.height - srii.unique(withlabels).height == 5
    assert srii.height - srii.unique([*withlabels, "loggf"]).height == 0

    # the surviving row keeps one line's A, not the sum of the two
    levelid_of_name = dict(dflevels.select("levelname", "levelid").iter_rows())
    lower = levelid_of_name["s5p z3P,enpercm=23776.241,j=1.0"]
    upper = levelid_of_name["d5d g3D,enpercm=65132.0,j=1.0"]
    kept = transitions.filter((pl.col("lowerlevel") == lower) & (pl.col("upperlevel") == upper))
    assert kept.height == 1
    assert kept["A"].item() == pytest.approx(3.805e8, rel=1e-3)


def test_write_phixs_data_keeps_a_table_with_no_threshold():
    """A cross section is real data; a threshold ARTIS never reads is not a reason to drop it."""
    from artisatomic.output import write_phixs_data

    args = phixs_args(optimaltemperature=3000, nphixspoints=2, phixsnuincrement=0.1)
    crosssections = np.array([[1.0, 0.5], [2.0, 1.0]])
    targetfractions = [[(0, 1.0)], [(0, 1.0)]]
    thresholds = np.array([13.6, np.nan])

    out = io.StringIO()
    write_phixs_data(out, 8, 1, crosssections, targetfractions, thresholds, args, io.StringIO())
    written = out.getvalue()

    # one header line per level: both get a table, where the writer used to discard level 2
    headers = [line for line in written.splitlines() if line.split()[0] == "8"]
    assert len(headers) == 2
    assert headers[0].split()[-2:] == ["1", "1.360000E+01"]
    # the writer puts the unknown threshold as zero, not as a NaN that the reader cannot parse
    assert headers[1].split()[-2:] == ["2", "0.000000E+00"]
    assert "nan" not in written.lower()

    # ...and level 2's cross sections are all there
    assert written.splitlines()[4:] == ["  2.00000000E+00", "  1.00000000E+00"]


def test_readfloers25data_extend_ion_list_skips_hidden_files(tmp_path, monkeypatch):
    """A "._" copy of a level file, as a macOS archive extracted on Linux leaves, names no ion."""
    from artisatomic import readfloers25data

    for name in ("70YbII_levels_calib.txt", "._70YbII_levels_calib.txt", "._57LaIII_levels_uncalib.txt"):
        (tmp_path / name).write_text("", encoding="utf-8")
    monkeypatch.setattr(readfloers25data, "get_basepath", lambda withforbidden: tmp_path)  # ruff: ignore[unused-lambda-argument]
    # Without the test mode, the search of the private folder runs first, and the folder of this
    # test is the public folder and the private folder. The result must not depend on the mode.
    monkeypatch.setattr(readfloers25data, "TESTMODE", True)

    assert readfloers25data.extend_ion_list([]) == [(70, [(2, "floers25calib")])]


def test_readtanakajpltdata_reads_a_transition_with_a_wide_wavelength_field(tmp_path, monkeypatch):
    """A wavelength of 1e9 nm or more is one character wider than its field and moves g_u*A right.

    Fe II 455 -> 454 in data_v2.1 is such a line. A cut at fixed positions read "2.386e-1" from
    "2.386e-13". The reader splits the line on white space instead.
    """
    from artisatomic import readtanakajpltdata

    lines = [
        "# Japan-Lithuania Opacity Database for Kilonova (version 1.1)",
        "# Se I ",
        "# 34 1 ",
        "# 3 2 ",
        "# CLOSED=  1s+  2s+ ",
        "# IP = 9.752 ",
        "# Energy levels ",
        "# num  weight parity      E(eV)      configuration ",
        "      1   5.0  even  0.0000000e+00 {  4s+ 2  4p- 2  4p+ 2 } ",
        "      2   3.0  even  2.3450344e-01 {  4s+ 2  4p- 1  4p+ 3 } ",
        "      3   1.0   odd  3.1167481e-01 {  4s+ 2  4p+ 4 } ",
        "# Transitions ",
        "# num_u   num_l   wavelength(nm)     g_u*A      log(g_l*f)",
        "      3       1       330.290     5.619e+05        -3.037 ",
        "      3       2 1290010779.045     2.386e-13        -8.225 ",
    ]
    (tmp_path / "34_1.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(readtanakajpltdata, "jpltpath", tmp_path)

    _ionization_energy_ev, dflevels, dftransitions = readtanakajpltdata.read_levels_and_transitions(
        34, 1, io.StringIO()
    )

    assert dflevels.height == 3
    assert dftransitions.sort("lowerlevel")["A"].to_list() == pytest.approx([5.619e5, 2.386e-13], rel=1e-12)


def test_readtanakajpltdata_records_the_header_with_a_joined_paper_line(tmp_path, monkeypatch):
    """The v2.1 header of Se III continues its paper line on a line with no #, and the ion name line repeats the title."""
    from artisatomic import readtanakajpltdata
    from artisatomic.base import IonLog

    lines = [
        "# Japan-Lithuania Opacity Database for Kilonova (version 2.1)",
        '# L. Kitoviene, G. Gaigalas, "Theoretical Investigation of the Ge',
        'Sequence" Journal of Physical and Chemical Reference Data 53 (2024) 033101.',
        "# Se III ",
        "# 34 3 ",
        "# 1 0 ",
        "#  ",
        "# IP = 31.697 ",
        "# Energy levels ",
        "# num  weight parity      E(eV)      configuration ",
        "      1   1.0  even  0.0000000e+00 {  4s+ 2 } ",
        "# Transitions ",
        "# num_u   num_l   wavelength(nm)     g_u*A      log(g_l*f)",
    ]
    (tmp_path / "34_3.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(readtanakajpltdata, "jpltpath", tmp_path)
    flog = IonLog(io.StringIO())

    _ionization_energy_ev, dflevels, _dftransitions = readtanakajpltdata.read_levels_and_transitions(34, 3, flog)

    assert dflevels.height == 1
    recorded = [line for line in flog.comments["adata"] if "come from" not in line]
    assert recorded == [
        "Japan-Lithuania Opacity Database for Kilonova (version 2.1)",
        (
            'L. Kitoviene, G. Gaigalas, "Theoretical Investigation of the Ge'
            ' Sequence" Journal of Physical and Chemical Reference Data 53 (2024) 033101.'
        ),
    ]
    assert recorded == [line for line in flog.comments["transitiondata"] if "come from" not in line]


def test_write_adata_writes_a_negative_zero_energy_as_zero():
    """A ground state read as -0.0 cm^-1 writes without a minus sign."""
    dflevels = leveltuples_to_pldataframe(
        pl.DataFrame(
            {
                "levelname": ["gs"],
                "g": [1.0],
                "energyabovegsinpercm": [-0.0],
                "lambdaangstrom": [911.0],
                "hillierlevelid": [1],
                "parity": [0],
            }
        )
    )
    buf = io.StringIO()
    write_adata(buf, 13, 5, dflevels, 100.0, [0], io.StringIO())
    energyfield = buf.getvalue().splitlines()[1].split()[1]
    assert energyfield == "0.0000000000000000"


def test_write_phixs_data_rejects_a_single_target_fraction_below_one():
    """A single target implies a fraction of 1.0 in the short output form, so 0.995 is an error."""
    from artisatomic.output import write_phixs_data

    args = phixs_args(nphixspoints=2, phixsnuincrement=0.1)
    crosssections = np.array([[1.0, 0.5]])
    thresholds = np.array([13.6])
    with pytest.raises(ValueError, match=r"sum to 0\.99500"):
        write_phixs_data(io.StringIO(), 8, 1, crosssections, [[(0, 0.995)]], thresholds, args, io.StringIO())
    # 1.0 within the tolerance passes
    write_phixs_data(io.StringIO(), 8, 1, crosssections, [[(0, 1.0 - 1e-7)]], thresholds, args, io.StringIO())


def test_get_photoiontargetfractions_merges_targets_that_match_one_level():
    """A matched route and a route that falls back to the ground state can come to the same level."""
    from artisatomic import readhillierdata

    dfenergy_levels = pl.DataFrame({"levelid": [0], "levelname": ["3s2_3p6_4s_2Se[1/2]"], "g": [2.0]})
    dfenergy_levels_upperion = pl.DataFrame(
        {
            "levelid": [0, 1],
            "levelname": ["3s2_3p6_1Se[0]", "3s2_3p5_4s_3Po[2]"],
            "g": [1.0, 5.0],
        }
    )
    # the first route matches the ground state, and the second matches nothing and falls back to it
    targetconfigs: list[list[tuple[str, float]] | None] = [[("3s2_3p6_1Se", 0.7), ("3s2_3p6_1So", 0.3)]]

    targetlist = readhillierdata.get_photoiontargetfractions(dfenergy_levels, dfenergy_levels_upperion, targetconfigs)

    # one entry only, with the two fractions added, and not the same target two times
    assert targetlist == [[(0, 1.0)]]


def test_get_photoiontargetfractions_shares_a_target_over_its_j_levels():
    """A matched configuration is still split over its J levels by statistical weight."""
    from artisatomic import readhillierdata

    dfenergy_levels = pl.DataFrame({"levelid": [0], "levelname": ["2s2_2p4_3Pe[2]"], "g": [5.0]})
    dfenergy_levels_upperion = pl.DataFrame(
        {
            "levelid": [0, 1, 2],
            "levelname": ["2s2_2p3_4So[3/2]", "2s2_2p3_2Do[5/2]", "2s2_2p3_2Do[3/2]"],
            "g": [4.0, 6.0, 4.0],
        }
    )
    targetconfigs: list[list[tuple[str, float]] | None] = [[("2s2_2p3_2Do", 1.0)]]

    targetlist = readhillierdata.get_photoiontargetfractions(dfenergy_levels, dfenergy_levels_upperion, targetconfigs)

    assert targetlist == [[(1, 0.6), (2, 0.4)]]


def test_get_photoiontargetfractions_matches_a_parenthesised_target():
    """F II names '2s2_2p3(2Do)' where F III has '2s2_2p3_2Do', so the separators must not decide."""
    from artisatomic import readhillierdata

    dfenergy_levels = pl.DataFrame({"levelid": [0], "levelname": ["2s2_2p4_3Pe[2]"], "g": [5.0]})
    dfenergy_levels_upperion = pl.DataFrame(
        {
            "levelid": [0, 1, 2],
            "levelname": ["2s2_2p3_4So[3/2]", "2s2_2p3_2Do[5/2]", "2s2_2p3_2Do[3/2]"],
            "g": [4.0, 6.0, 4.0],
        }
    )
    targetconfigs: list[list[tuple[str, float]] | None] = [[("2s2_2p3(2Do)", 1.0)]]

    targetlist = readhillierdata.get_photoiontargetfractions(dfenergy_levels, dfenergy_levels_upperion, targetconfigs)

    # the two J levels of 2Do, by statistical weight, and not the ground state fallback
    assert targetlist == [[(1, 0.6), (2, 0.4)]]


def test_get_photoiontargetfractions_matches_a_target_with_no_separators():
    """O IV names '2s2p3Po' where O V has '2s_2p_3Po'."""
    from artisatomic import readhillierdata

    dfenergy_levels = pl.DataFrame({"levelid": [0], "levelname": ["2s2_2p_2Po[1/2]"], "g": [2.0]})
    dfenergy_levels_upperion = pl.DataFrame(
        {
            "levelid": [0, 1],
            "levelname": ["2s2_1Se[0]", "2s_2p_3Po[1]"],
            "g": [1.0, 3.0],
        }
    )
    targetconfigs: list[list[tuple[str, float]] | None] = [[("2s2p3Po", 1.0)]]

    targetlist = readhillierdata.get_photoiontargetfractions(dfenergy_levels, dfenergy_levels_upperion, targetconfigs)

    assert targetlist == [[(1, 1.0)]]


def test_get_photoiontargetfractions_keeps_the_ground_state_for_a_different_term():
    """K I names '3s2_3p6_1So' where K II has '3s2_3p6_1Se'. The parity differs, so it must not match."""
    from artisatomic import readhillierdata

    dfenergy_levels = pl.DataFrame({"levelid": [0], "levelname": ["3s2_3p6_4s_2Se[1/2]"], "g": [2.0]})
    dfenergy_levels_upperion = pl.DataFrame(
        {
            "levelid": [0, 1],
            "levelname": ["3s2_3p6_1Se[0]", "3s2_3p5_4s_3Po[2]"],
            "g": [1.0, 5.0],
        }
    )
    targetconfigs: list[list[tuple[str, float]] | None] = [[("3s2_3p6_1So", 1.0)]]

    targetlist = readhillierdata.get_photoiontargetfractions(dfenergy_levels, dfenergy_levels_upperion, targetconfigs)

    assert targetlist == [[(0, 1.0)]]


def test_get_photoiontargetfractions_rejects_an_ambiguous_stripped_match():
    """Two upper ion names that differ only in their separators give no target that the resolver can share."""
    from artisatomic import readhillierdata

    dfenergy_levels = pl.DataFrame({"levelid": [0], "levelname": ["3d7_a4Fe[9/2]"], "g": [10.0]})
    # the two names are different levels, but they become one string with the separators removed
    dfenergy_levels_upperion = pl.DataFrame(
        {
            "levelid": [0, 1],
            "levelname": ["3d6(5D)4s_a6De[9/2]", "3d6_5D_4s_a6De[9/2]"],
            "g": [10.0, 10.0],
        }
    )
    targetconfigs: list[list[tuple[str, float]] | None] = [[("3d6(5D)4sa6De", 1.0)]]

    with pytest.raises(ValueError, match="matched more than one level name"):
        readhillierdata.get_photoiontargetfractions(dfenergy_levels, dfenergy_levels_upperion, targetconfigs)


def test_get_photoiontargetfractions_matches_a_slash_target_with_separators_removed():
    """Each part of a slash target gets its own second comparison, so two matched names are not ambiguous."""
    from artisatomic import readhillierdata

    dfenergy_levels = pl.DataFrame({"levelid": [0], "levelname": ["2s2_2p4_3Pe[2]"], "g": [5.0]})
    dfenergy_levels_upperion = pl.DataFrame(
        {
            "levelid": [0, 1, 2],
            "levelname": ["2s2_2p3_4So[3/2]", "2s2_2p3_2Do[5/2]", "2s2_2p3_2Po[1/2]"],
            "g": [4.0, 6.0, 2.0],
        }
    )
    # each part matches exactly one name, so the parts share the fraction as the exact comparison does
    targetconfigs: list[list[tuple[str, float]] | None] = [[("2s2_2p3(2Do)/2s2_2p3(2Po)", 1.0)]]

    targetlist = readhillierdata.get_photoiontargetfractions(dfenergy_levels, dfenergy_levels_upperion, targetconfigs)

    assert targetlist == [[(1, 0.75), (2, 0.25)]]


def test_strip_name_separators():
    """The underscore and the parentheses go; nothing else changes."""
    from artisatomic.readhillierdata import strip_name_separators

    assert strip_name_separators("2s2_2p3(2Do)") == "2s22p32Do"
    assert strip_name_separators("2s2_2p3_2Do") == "2s22p32Do"
    assert strip_name_separators("3d6(5D)4sa6De") == "3d65D4sa6De"
    assert strip_name_separators("3d6(5D)4s_a6De") == "3d65D4sa6De"
    # a different term stays different
    assert strip_name_separators("3s2_3p6_1So") != strip_name_separators("3s2_3p6_1Se")


def test_write_phixs_data_rejects_a_duplicated_target():
    """A target level that occurs two times would give the same target two fractions in ARTIS."""
    from artisatomic.output import write_phixs_data

    args = phixs_args(optimaltemperature=3000, nphixspoints=2, phixsnuincrement=0.1)
    crosssections = np.array([[1.0, 0.5]])
    targetfractions = [[(0, 0.4), (1, 0.3), (0, 0.3)]]
    thresholds = np.array([13.6])

    out = io.StringIO()
    with pytest.raises(ValueError, match="occur more than one time"):
        write_phixs_data(out, 8, 1, crosssections, targetfractions, thresholds, args, io.StringIO())

    # the level fails before any part of the ion goes out
    assert not out.getvalue()


def test_write_phixs_data_rejects_a_bad_fraction_sum_before_output():
    """A bad fraction sum fails before any part of the ion goes out."""
    from artisatomic.output import write_phixs_data

    args = phixs_args(optimaltemperature=3000, nphixspoints=2, phixsnuincrement=0.1)
    crosssections = np.array([[1.0, 0.5]])
    targetfractions = [[(0, 0.4), (1, 0.3)]]
    thresholds = np.array([13.6])

    out = io.StringIO()
    with pytest.raises(ValueError, match="sum to"):
        write_phixs_data(out, 8, 1, crosssections, targetfractions, thresholds, args, io.StringIO())

    assert not out.getvalue()


def test_log_degenerate_transitions():
    """ARTIS drops a transition whose two levels have one energy, so artisatomic reports it."""
    from artisatomic.output import log_degenerate_transitions

    dflevels = pl.DataFrame(
        {"levelid": [0, 1, 2], "energyabovegsinpercm": [0.0, 100.0, 100.0]},
        schema={"levelid": pl.Int64, "energyabovegsinpercm": pl.Float64},
    )

    def warnings_for(dftransitions: pl.DataFrame) -> str:
        flog = io.StringIO()
        log_degenerate_transitions(flog, dflevels, dftransitions)
        return flog.getvalue()

    # levels 1 and 2 share an energy, so the log reports that pair and not the other
    degenerate = pl.DataFrame({"lowerlevel": [1], "upperlevel": [2], "A": [0.0], "coll_str": [-2.0]})
    assert "1 transitions connect two levels of the same energy" in warnings_for(degenerate)
    assert "(0 of them with a collision strength)" in warnings_for(degenerate)

    ok = pl.DataFrame({"lowerlevel": [0], "upperlevel": [1], "A": [1.0], "coll_str": [-1.0]})
    assert not warnings_for(ok)

    # a tabulated upsilon on such a pair is real data that ARTIS will not use, so the log counts it
    withupsilon = degenerate.with_columns(coll_str=pl.lit(0.5))
    assert "(1 of them with a collision strength)" in warnings_for(withupsilon)

    # a transitions frame with no coll_str column still reports the pair, and counts none
    nocollstr = pl.DataFrame({"lowerlevel": [1], "upperlevel": [2], "A": [0.0]})
    assert "(0 of them with a collision strength)" in warnings_for(nocollstr)

    # the log reports a pair whose lower id has the higher energy on its own line, not as degenerate
    inverted = pl.DataFrame({"lowerlevel": [1], "upperlevel": [0], "A": [1.0]})
    assert "1 transitions have a lower level id whose energy is above" in warnings_for(inverted)
    assert "same energy" not in warnings_for(inverted)

    # the function cannot check a level frame with no energy column, and must stay silent, not raise
    flog = io.StringIO()
    log_degenerate_transitions(flog, dflevels.drop("energyabovegsinpercm"), degenerate)
    assert not flog.getvalue()


def test_read_photoionizations_without_data_gives_empty_arrays():
    """An ion with no QUB cross sections must give the empty arrays, not zero-filled ones.

    iondata.read_ion_data() reads a zero-filled array as data and then skips the hydrogenic
    estimate. The output would then hold the ion with no cross sections at all.
    """
    args = phixs_args()
    dfenergylevels = pl.DataFrame({"levelname": ["a"] * 5})
    phixs = readadasdata.read_photoionizations(38, 1, dfenergylevels, args, io.StringIO())
    crosssections, targetfractions, thresholds = phixs.crosssections, phixs.targetfractions, phixs.thresholds_ev
    assert targetfractions is not None
    assert crosssections.shape == (0, 100)
    assert targetfractions == []
    assert thresholds.shape == (0,)

    # The QUB Co II tables are for CMFGEN levels. Co II through the "adas" handler has ADAS levels.
    flog = io.StringIO()
    assert readadasdata.read_photoionizations(27, 2, dfenergylevels, args, flog).crosssections.size == 0
    assert "The ADAS data has no photoionisation cross sections for this ion." in flog.getvalue()


def test_fill_missing_phixs_thresholds():
    """fill_missing_phixs_thresholds() derives a missing threshold as ARTIS does."""
    from artisatomic.iondata import IonData
    from artisatomic.output import fill_missing_phixs_thresholds

    def makeion(ion_stage, ionpot, energiespercm, targets, thresholds):
        return IonData(
            ion_stage=ion_stage,
            handler="test",
            is_top_ion=False,
            ionization_energy_ev=ionpot,
            dfenergylevels=pl.DataFrame({"energyabovegsinpercm": energiespercm}),
            dftransitions=pl.DataFrame(),
            upsilondict={},
            photoion_targetconfigs=None,
            photoionization_crosssections=np.zeros((len(energiespercm), 1)),
            photoionization_targetfractions=targets,
            photoionization_thresholds_ev=np.array(thresholds),
        )

    percm_per_ev = 1.0 / hc_in_ev_cm
    # this ion: ground state and a level 2 eV up. Upper ion: ground state and a level 1 eV up.
    ion = makeion(1, 10.0, [0.0, 2.0 * percm_per_ev], [[(0, 1.0)], [(1, 1.0)]], [np.nan, np.nan])
    upperion = makeion(2, 25.0, [0.0, 1.0 * percm_per_ev], [], [])

    filled = fill_missing_phixs_thresholds(ion, upperion, io.StringIO())

    # ionisation energy + target level energy - this level's energy
    assert filled[0] == pytest.approx(10.0 + 0.0 - 0.0)
    assert filled[1] == pytest.approx(10.0 + 1.0 - 2.0)

    # a threshold that the reader did give stays as it is, and so does one with no upper ion to look in
    given = makeion(1, 10.0, [0.0], [[(0, 1.0)]], [7.5])
    assert fill_missing_phixs_thresholds(given, upperion, io.StringIO())[0] == pytest.approx(7.5)
    assert np.isnan(fill_missing_phixs_thresholds(ion, None, io.StringIO())[0])

    # a level at or above the continuum has no edge to describe, so it keeps its NaN
    above = makeion(1, 1.0, [5.0 * percm_per_ev], [[(0, 1.0)]], [np.nan])
    assert np.isnan(fill_missing_phixs_thresholds(above, upperion, io.StringIO())[0])


def test_resolve_coll_str_negative_upsilon_is_a_forbidden_marker():
    """A reader's negative upsilon says "forbidden, no value", and resolve_coll_str() must not read it as permitted.

    readhillierdata writes -2 for the J pairs within a term. Those pairs carry no A. A permitted
    flag would therefore send them to the van Regemorter formula (van Regemorter 1962, ApJ, 136,
    906-915, doi:10.1086/147445) with an oscillator strength of zero, which is no collisional
    coupling at all. The -2 asks instead for the approximation of Axelrod (1980, PhD thesis,
    University of California, Santa Cruz).
    """
    from artisatomic.output import resolve_coll_str

    # neither level has a parity or a J, so nothing but the upsilon can decide
    dflevels = pl.DataFrame(
        {"levelid": [0, 1], "parity": [None, None], "j": [None, None]},
        schema={"levelid": pl.Int64, "parity": pl.Int64, "j": pl.Float64},
    )
    dftransitions = pl.DataFrame({"lowerlevel": [0], "upperlevel": [1], "A": [0.0]})
    joined = add_level_ids_forbidden(dflevels, dftransitions)

    # add_level_ids_forbidden() cannot tell on its own
    assert joined["forbidden"].to_list() == [False]

    # a negative upsilon makes the flag true, and coll_str repeats it
    resolved = resolve_coll_str(joined.with_columns(upsilon=pl.lit(-2.0)))
    assert resolved["forbidden"].to_list() == [True]
    assert resolved["coll_str"].to_list() == [-2.0]
    assert "upsilon" not in resolved.columns

    # a real upsilon passes through and leaves the flag alone
    real = resolve_coll_str(joined.with_columns(upsilon=pl.lit(0.5)))
    assert real["forbidden"].to_list() == [False]
    assert real["coll_str"].to_list() == [0.5]

    # only a missing upsilon reaches -1, and a zero is a real collision strength
    assert resolve_coll_str(joined.with_columns(upsilon=pl.lit(None, dtype=pl.Float64)))["coll_str"].to_list() == [-1.0]
    assert resolve_coll_str(joined.with_columns(upsilon=pl.lit(0.0)))["coll_str"].to_list() == [0.0]

    # ...and a missing upsilon on a pair the parities already forbid gives -2
    sameparity = pl.DataFrame(
        {"levelid": [0, 1], "parity": [1, 1], "j": [None, None]},
        schema={"levelid": pl.Int64, "parity": pl.Int64, "j": pl.Float64},
    )
    forbidden = add_level_ids_forbidden(sameparity, dftransitions).with_columns(upsilon=pl.lit(None, dtype=pl.Float64))
    assert resolve_coll_str(forbidden)["coll_str"].to_list() == [-2.0]


def test_fill_missing_phixs_thresholds_treats_a_negative_as_missing():
    """A reader marks a threshold it does not have in two ways, and both have to count.

    The arrays start as NaN. threshold_is_known() also counts a value at or below zero as missing.
    readadasdata wrote -1.0 at one time, and only NaN counted then. That left every ADAS level with
    its -1 and made the calculation dead code for the one reader that asks for it.
    """
    from artisatomic.iondata import IonData
    from artisatomic.output import fill_missing_phixs_thresholds

    def makeion(ion_stage, ionpot, energiespercm, targets, thresholds):
        return IonData(
            ion_stage=ion_stage,
            handler="test",
            is_top_ion=False,
            ionization_energy_ev=ionpot,
            dfenergylevels=pl.DataFrame({"energyabovegsinpercm": energiespercm}),
            dftransitions=pl.DataFrame(),
            upsilondict={},
            photoion_targetconfigs=None,
            photoionization_crosssections=np.zeros((len(energiespercm), 1)),
            photoionization_targetfractions=targets,
            photoionization_thresholds_ev=np.array(thresholds),
        )

    upperion = makeion(2, 25.0, [0.0], [], [])
    # the two marks for "no value": NaN and readadasdata's -1
    ion = makeion(1, 17.084, [0.0, 0.0], [[(0, 1.0)], [(0, 1.0)]], [np.nan, -1.0])

    filled = fill_missing_phixs_thresholds(ion, upperion, io.StringIO())

    # a ground state that ionises to the upper ion's ground state has the ionisation energy itself
    assert filled[0] == pytest.approx(17.084)
    assert filled[1] == pytest.approx(17.084)


def test_get_nist_ionization_energies_ev():
    """The NIST loader gives a float for each listed energy, and no entry for a blank one.

    The NIST table ends with footnotes and leaves the energy blank when NIST lists none. Both
    must stay out of the result, so a caller gets a KeyError and not a NaN.
    """
    from artisatomic.base import get_nist_ionization_energies_ev
    from artisatomic.base import get_nist_ionization_provenance

    energies = get_nist_ionization_energies_ev()

    assert energies[1, 1] == pytest.approx(13.598434599702)
    assert energies[26, 2] == pytest.approx(16.19921)
    assert all(isinstance(z, int) and isinstance(stage, int) for z, stage in energies)
    assert all(isinstance(value, float) and np.isfinite(value) for value in energies.values())
    # Rf XV has a blank energy
    assert (104, 15) not in energies
    # the table names its source, so the output files can quote it
    assert any("NIST" in line for line in get_nist_ionization_provenance())


def test_parse_nist_ionization_table():
    """The parser keeps the provenance lines, stops at the footnotes, and rejects a row that does not parse."""
    from artisatomic.base import parse_nist_ionization_table

    header = "At. num\tSp. Name\tIon Charge\tPrefix\tIonization Energy (a) (eV)\tSuffix\n"
    rows = '"26"\t"Fe I"\t"0"\t""\t"7.9"\t""\n"26"\t"Fe II"\t"+1"\t""\t"16.2"\t""\n"26"\t"Fe III"\t"+2"\t""\t""\t""\n'
    # the footnote holds a tab, which would break the column count if the parser reached it
    footer = "Notes:\n(a) Uncertainty\tof the listed value is unknown.\n"

    provenance, energies = parse_nist_ionization_table("# Source: NIST ASD\n# Date: 2022\n" + header + rows + footer)
    assert provenance == ["Source: NIST ASD", "Date: 2022"]
    assert energies == {(26, 1): 7.9, (26, 2): 16.2}

    with pytest.raises(ValueError, match="does not parse"):
        parse_nist_ionization_table(header + rows + '"26"\t"Fe IV"\t"+3"\t""\t"abc"\t""\n')
    with pytest.raises(ValueError, match="non-finite"):
        parse_nist_ionization_table(header + rows + '"26"\t"Fe IV"\t"+3"\t""\t"nan"\t""\n')


def test_readmonsdata_reads_the_sample(monkeypatch):
    """Read Ce V from the committed sample and check the first level and transition."""
    monkeypatch.setattr(readmonsdata, "datafilepath", PYDIR / ".." / "atomic-data-mons" / "test_sample")
    ionization_energy_ev, dflevels, dftransitions = readmonsdata.read_levels_and_transitions(58, 5, io.StringIO())

    assert ionization_energy_ev == 65.55  # the NIST table gives Ce V exactly this value
    assert dflevels.height == 450
    assert dftransitions.height == 17139
    # the level file has no energy order, so the reader finds the ground state by its energy
    assert dflevels["energyabovegsinpercm"][0] == 0.0
    assert dflevels["g"][0] == 1.0
    assert dflevels["j"][0] == 0.0
    assert dflevels["energyabovegsinpercm"][1] == pytest.approx(123734.52825648028)
    assert dflevels["g"][1] == 3.0
    assert dflevels["j"][1] == 1.0
    assert dflevels["parity"].is_null().all()

    # first line of the transition file: 269.5076 Angstrom from the ground state with gf=2.588659e-03.
    # The upper level is therefore the highest level of the sample (371047.04 cm^-1, g=3).
    # A = gf / (1.49919e-16 g_upper lambda^2), where the third column of the file gives gf and not f.
    first = dftransitions.row(0, named=True)
    assert first["lowerlevel"] == 0
    assert first["upperlevel"] == 449
    assert first["A"] == pytest.approx(7.9241887e07, rel=1e-6)

    # 32 lines of the sample transition file have the ground state as their lower level, and none
    # has it as the upper level
    counts = transition_count_of_level(dftransitions, dflevels.height)
    assert counts[0] == 32
    assert sum(counts) == 2 * dftransitions.height


def test_get_nearest_level_indices():
    """Check the binary search for the two closest levels, including the ends of the range and a tie."""
    sorted_energies = np.array([0.0, 10.0, 20.0, 30.0])
    energies = np.array([-5.0, 0.4, 4.9, 5.0, 5.1, 19.9, 35.0])
    nearest, secondnearest = readmonsdata.get_nearest_level_indices(sorted_energies, energies)
    assert nearest.tolist() == [0, 0, 0, 0, 1, 2, 3]
    assert secondnearest.tolist() == [1, 1, 1, 1, 0, 1, 2]

    # a table of one level cannot give an index of minus one
    onelevel, onelevel_second = readmonsdata.get_nearest_level_indices(np.array([5.0]), np.array([1.0, 9.0]))
    assert onelevel.tolist() == [0, 0]
    assert onelevel_second.tolist() == [0, 0]


def test_readfloers25data_pertype_merge_swap_and_forbidden(monkeypatch, tmp_path):
    """The per-type Floers+25 files merge duplicate level pairs and set the forbidden flag.

    A reversed row (Lower > Upper) swaps into energy order first, so both orientations of a
    pair merge into one row with the summed A. A merged row is forbidden only when no E1 line
    contributes to it. The committed test_sample data has none of these cases, so this test
    builds the files itself.
    """
    header = "Test table\n--\n--\n--\n"
    (tmp_path / "57LaII_levels_calib.txt").write_text(
        header + " Index Energy J Parity Configuration\n 0 0.0 0 0 5d1\n 1 100.0 1 1 5p1\n 2 200.0 2 0 4f1\n"
    )
    transheader = header + " Lower Upper A Type\n"
    (tmp_path / "57LaII_transitions_calib_E1.txt").write_text(transheader + " 0 1 1.0e+06 E1\n")
    # the row to level 9 references a level that the levels file does not list: discard it
    (tmp_path / "57LaII_transitions_calib_M1.txt").write_text(transheader + " 0 2 2.0e+00 M1\n 0 9 4.0e+00 M1\n")
    # the E2 row is in reverse order on purpose: it must swap and then merge with the M1 row
    (tmp_path / "57LaII_transitions_calib_E2.txt").write_text(transheader + " 2 0 3.0e+00 E2\n")

    from artisatomic import readfloers25data

    monkeypatch.setattr(readfloers25data, "get_basepath", lambda **_kwargs: tmp_path)
    flog = io.StringIO()
    _, dflevels, dftransitions = readfloers25data.read_levels_and_transitions(
        57, 2, flog, calibrated=True, withforbidden=True
    )
    assert "The reader discarded 1 transitions" in flog.getvalue()

    assert dflevels.height == 3
    rows = {
        (lower, upper): (A, forbidden)
        for upper, lower, A, forbidden in dftransitions[["upperlevel", "lowerlevel", "A", "forbidden"]].iter_rows()
    }
    assert rows == {(0, 1): (1.0e6, False), (0, 2): (5.0, True)}
    assert dflevels["levelname"].to_list() == ["5d1 J=0 index=0", "5p1 J=1 index=1", "4f1 J=2 index=2"]
    assert transition_count_of_level(dftransitions, dflevels.height) == [2, 1, 1]

    # an unknown transition type must stop the run rather than count as forbidden
    (tmp_path / "57LaII_transitions_calib_E2.txt").write_text(transheader + " 2 0 3.0e+00 XX\n")
    with pytest.raises(ValueError, match="Unknown transition type"):
        readfloers25data.read_levels_and_transitions(57, 2, io.StringIO(), calibrated=True, withforbidden=True)


def test_readfloers25data_ragged_rows(monkeypatch, tmp_path):
    """A short row keeps the columns in front of its empty cell. A long row stops the run.

    The Floers+25 tables right-align each value in a cell of a minimum width. A value that is
    wider than its cell moves the rest of the line. A column is not at a fixed character position,
    and the reader splits each line on whitespace. The level tables leave the LS2 cell of a high-l
    level empty, which gives a row of nine tokens against a header of ten. The reader takes no
    column after LS2, so it reads such a row. A row with an extra token moves every value into the
    column on its left, so the reader rejects it.
    """
    from artisatomic import readfloers25data

    header = "Test table\n--\n--\n--\n"
    levelheader = " Index       Z  Charge      Energy       J  Parity  Configuration     LS    LS2    Method\n"
    # the second level has an empty LS2 cell, as a high-l level of the published tables does
    levels = (
        "     0      57       2        0.00     3/2       0            5d1     2D     2D   uncalib\n"
        "     1      57       2   113810.70     9/2       0            5g1     2G          uncalib\n"
    )
    (tmp_path / "57LaII_levels_calib.txt").write_text(header + levelheader + levels)
    transheader = header + " Lower Upper A Type\n"
    (tmp_path / "57LaII_transitions_calib_E1.txt").write_text(transheader + " 0 1 1.0e+06 E1\n")

    monkeypatch.setattr(readfloers25data, "get_basepath", lambda **_kwargs: tmp_path)
    _, dflevels, _ = readfloers25data.read_levels_and_transitions(
        57, 2, io.StringIO(), calibrated=True, withforbidden=True
    )

    # the short row keeps its Energy, J, Parity and Configuration, which all precede LS2
    assert dflevels["levelname"].to_list() == ["5d1 J=3/2 index=0", "5g1 J=9/2 index=1"]
    assert dflevels["energyabovegsinpercm"].to_list() == [0.0, 113810.70]
    assert dflevels["g"].to_list() == [4, 10]

    # a row with an extra token would move every value into the column on its left
    (tmp_path / "57LaII_levels_calib.txt").write_text(
        header
        + levelheader
        + levels
        + "     2      57       2   200000.00     1/2       1  5f1  2F  2F  uncalib  EXTRA\n"
    )
    with pytest.raises(ValueError, match="have more than 10 tokens"):
        readfloers25data.read_levels_and_transitions(57, 2, io.StringIO(), calibrated=True, withforbidden=True)


def test_readfloers25data_degenerate_tables(monkeypatch, tmp_path):
    """A table of no data row gives no transition. A broken table names its own file.

    A per-type file holds the transitions of one type, so an ion with no line of that type has a
    header and no data row. Such a file must add nothing rather than stop the run. A blank line
    between the third rule and the column names is also permitted.
    """
    from artisatomic import readfloers25data

    header = "Test table\n--\n--\n--\n"
    (tmp_path / "57LaII_levels_calib.txt").write_text(
        header + " Index Energy J Parity Configuration\n 0 0.0 0 0 5d1\n 1 100.0 1 1 5p1\n"
    )
    transheader = " Lower Upper A Type\n"
    (tmp_path / "57LaII_transitions_calib_E1.txt").write_text(header + transheader + " 0 1 1.0e+06 E1\n")
    # the M1 file holds no data row, because the ion has no M1 line
    (tmp_path / "57LaII_transitions_calib_M1.txt").write_text(header + transheader)
    # the E2 file has a blank line between the third rule and the column names
    (tmp_path / "57LaII_transitions_calib_E2.txt").write_text(header + "\n" + transheader)

    monkeypatch.setattr(readfloers25data, "get_basepath", lambda **_kwargs: tmp_path)
    _, dflevels, dftransitions = readfloers25data.read_levels_and_transitions(
        57, 2, io.StringIO(), calibrated=True, withforbidden=True
    )
    assert dftransitions.height == 1
    assert not dftransitions["forbidden"][0]
    assert dflevels["levelname"].to_list()[:2] == ["5d1 J=0 index=0", "5p1 J=1 index=1"]
    assert transition_count_of_level(dftransitions, dflevels.height) == [1, 1] + [0] * (dflevels.height - 2)

    # a file that names none of the columns that the reader takes must say so
    (tmp_path / "57LaII_transitions_calib_M1.txt").write_text(header + " Lower Upper A\n 0 1 1.0e+00\n")
    with pytest.raises(ValueError, match=r"has no \['Type'\]"):
        readfloers25data.read_levels_and_transitions(57, 2, io.StringIO(), calibrated=True, withforbidden=True)

    # a file with no column header at all is not a data table
    (tmp_path / "57LaII_transitions_calib_M1.txt").write_text(header)
    with pytest.raises(ValueError, match="Did not find the expected data table"):
        readfloers25data.read_levels_and_transitions(57, 2, io.StringIO(), calibrated=True, withforbidden=True)


def test_iondata_handlers_registry():
    """Each handler must keep its own level-name parser and its own return shape.

    The parsers used to be a second table in phixs.py, and the length of the reader's result
    used to give the return shape. Both are registry fields now, so nothing else checks them. A
    parser registered against the wrong handler would give an ion the hydrogenic cross sections
    of another data source. A wrong returns_upsilondict would make the unpack fail on the first
    run.
    """
    from artisatomic import groundstatesonlynist
    from artisatomic import readboyledata
    from artisatomic import readdreamdata
    from artisatomic import readlisbondata
    from artisatomic.iondata import handlers

    expected_parsers = {
        "cmfgen": readhillierdata.get_level_valence_n,
        "cmfgen_qubphixs": readhillierdata.get_level_valence_n,
        "kurucz": readkuruczdata.get_level_valence_n,
        "fac": readfacdata.get_level_valence_n,
        "floers25calibwithforbidden": readfloers25data.get_level_valence_n,
        "floers25calib": readfloers25data.get_level_valence_n,
        "floers25uncalib": readfloers25data.get_level_valence_n,
        "tanakajplt": readtanakajpltdata.get_level_valence_n,
        "adas": readadasdata.get_level_valence_n,
    }
    assert {
        name: handler.get_level_valence_n for name, handler in handlers.items() if handler.get_level_valence_n
    } == expected_parsers

    # these handlers have no parser in the registry, so their ions get no hydrogenic estimate
    # and match_hydrogenic_phixs() writes a warning. A registered parser is what changes that.
    assert {name for name, handler in handlers.items() if handler.get_level_valence_n is None} == {
        "boyle",
        "dream",
        "lisbon",
        "mons",
        "gsnist",
    }

    # only the ADAS reader returns collision strengths beside the levels and the transitions. Only
    # it takes args, for the temperature that selects the tabulated collision strengths
    assert {name for name, handler in handlers.items() if handler.returns_upsilondict} == {"adas"}
    assert {name for name, handler in handlers.items() if handler.reader_takes_args} == {"adas"}

    # CMFGEN is the one data source with collision strengths in its own file. CMFGEN and the QUB
    # Co data are the two with cross sections.
    assert {name: handler.read_coldata for name, handler in handlers.items() if handler.read_coldata} == {
        "cmfgen": readhillierdata.read_coldata,
        "cmfgen_qubphixs": readhillierdata.read_coldata,
    }
    assert {name: handler.read_phixs for name, handler in handlers.items() if handler.read_phixs} == {
        "cmfgen": readhillierdata.read_phixs_tables,
        "cmfgen_qubphixs": readadasdata.read_cmfgen_qubphixs_photoionizations,
        "adas": readadasdata.read_photoionizations,
    }

    # the readers that the registry calls with (atomic_number, ion_stage, flog[, args])
    expected_readers = {
        "cmfgen": readhillierdata.read_levels_and_transitions,
        "cmfgen_qubphixs": readhillierdata.read_levels_and_transitions,
        "kurucz": readkuruczdata.read_levels_and_transitions,
        "dream": readdreamdata.read_levels_and_transitions,
        "lisbon": readlisbondata.read_levels_and_transitions,
        "fac": readfacdata.read_levels_and_transitions,
        "mons": readmonsdata.read_levels_and_transitions,
        "tanakajplt": readtanakajpltdata.read_levels_and_transitions,
        "gsnist": groundstatesonlynist.read_ground_levels,
        "adas": readadasdata.read_adas_levels_and_transitions,
        "boyle": readboyledata.read_levels_and_transitions,
    }
    for name, reader in expected_readers.items():
        assert handlers[name].read_levels_and_transitions is reader, name

    # the three floers25 entries call one reader and select the data set by keyword, so each
    # entry must bind its own keywords
    expected_keywords = {
        "floers25calibwithforbidden": {"calibrated": True, "withforbidden": True},
        "floers25calib": {"calibrated": True},
        "floers25uncalib": {"calibrated": False},
    }
    for name, keywords in expected_keywords.items():
        reader = handlers[name].read_levels_and_transitions
        assert isinstance(reader, functools.partial)
        assert reader.func is readfloers25data.read_levels_and_transitions
        assert reader.keywords == keywords


def test_console_script_entry_points_resolve():
    """Each console script must name a module and a function that exist.

    The lint job of CI starts three of the four console scripts with --help. makeartisgammaspecfiles
    has no argument parser, so CI does not start it. A wrong module path for that script fails
    only when a user runs the installed command.
    """
    from importlib.metadata import entry_points

    declared = {ep.name: ep for ep in entry_points(group="console_scripts") if ep.module.startswith("artisatomic")}
    assert set(declared) == {
        "makeartisatomicfiles",
        "makeartischargetransferfile",
        "makeartisgammaspecfiles",
        "makeartisrecombratefile",
    }

    # the module that defines main(), not the package root: the root re-exports nothing
    assert declared["makeartisatomicfiles"].value == "artisatomic.cli:main"

    for name, entrypoint in declared.items():
        assert callable(entrypoint.load()), name


def test_lchars_orbital_letters_skip_j_p_and_s():
    """The orbital letter sequence gives every letter its own l, so q, r and the letters after t parse correctly.

    The sequence skips J. It also skips P and S at l = 12 and l = 14, because those letters are
    already l = 1 and l = 0. A table that repeated them read a 13q orbital as l = 13. That
    inverted its parity and made the l >= n merge test fire on a real orbital.
    """
    orbitals, twosplusone, term_l, parity, _ = interpret_configuration("13q_2Q")
    assert (orbitals, twosplusone, term_l, parity) == (["13q"], 2, 12, 0)
    assert get_config_parity("13q_2Q") == 0
    assert not has_merged_orbital("13q_2Q")
    assert get_config_parity("14r_2R") == 1
    # w and z are CMFGEN's merge markers at any n. 18w is the whole n = 18 shell (g = 2 x 18^2).
    # l = 17 < 18 must therefore not make it a single orbital with a parity
    assert has_merged_orbital("4z_2Z")
    assert has_merged_orbital("2s2_13w_2W")
    assert has_merged_orbital("2s2_2p3(4So)18w_5W")
    assert has_merged_orbital("3s2_30w_2W")
    assert readhillierdata.get_level_parity("3s2_30w_2W") == -1


def test_interpret_configuration_empty_name():
    """A name with nothing before its J bracket has no parity, and does not raise."""
    assert interpret_configuration("") == ([], -1, -1, -1, -1)
    assert interpret_configuration("[1/2]") == ([], -1, -1, -1, -1)
    assert get_config_parity("") is None
    assert not has_merged_orbital("[3/2]")


def test_parse_ion_handlers_rejects_unknown_handlers_and_bad_entries():
    """A misspelt handler or a malformed entry fails at parse time, before any output file exists."""
    from artisatomic.ionhandlers import parse_ion_handlers

    # the renamed handler is still accepted under its old name
    assert parse_ion_handlers([[26, [[2, "cmfgen"], [3, "qub_data"]]]]) == [(26, [(2, "cmfgen"), (3, "adas")])]

    with pytest.raises(ValueError, match="unknown handler 'cmfgne'"):
        parse_ion_handlers([[26, [[2, "cmfgne"]]]])
    with pytest.raises(TypeError, match="names no handler"):
        parse_ion_handlers([[26, [2]]])
    with pytest.raises(TypeError, match=r"not an \[ion_stage, handler\] pair"):
        parse_ion_handlers([[26, [["2"]]]])
    with pytest.raises(TypeError, match=r"not an \[ion_stage, handler\] pair"):
        parse_ion_handlers([[26, ["2"]]])


def test_add_level_ids_forbidden_name_joins():
    """Name-keyed transitions get their ids in the reader's row order, and a bad name fails.

    An inner join drops a transition whose name matches no level without a word, and a name that
    two levels share multiplies its rows. Both used to print a warning; adata.txt then disagreed
    with transitiondata.txt.
    """
    dflevels = pl.DataFrame({"levelid": [0, 1, 2], "levelname": ["a", "b", "c"], "parity": [0, 1, 0]})
    dftransitions = pl.DataFrame({"namefrom": ["b", "a"], "nameto": ["c", "c"], "A": [1.0, 2.0]})

    result = add_level_ids_forbidden(dflevels, dftransitions)
    assert result["lowerlevel"].to_list() == [1, 0]
    assert result["upperlevel"].to_list() == [2, 2]
    assert result["forbidden"].to_list() == [False, True]

    with pytest.raises(ValueError, match="nameto join changed the transition count from 1 to 0"):
        add_level_ids_forbidden(dflevels, pl.DataFrame({"namefrom": ["a"], "nameto": ["x"], "A": [1.0]}))

    dflevels_duplicate = pl.DataFrame({"levelid": [0, 1, 2], "levelname": ["a", "b", "a"], "parity": [0, 1, 0]})
    with pytest.raises(ValueError, match="namefrom join changed the transition count from 1 to 2"):
        add_level_ids_forbidden(dflevels_duplicate, pl.DataFrame({"namefrom": ["a"], "nameto": ["b"], "A": [1.0]}))


def test_transition_count_of_level():
    """Both levels of every transition count, by level id, and an id outside the level list fails."""
    dftransitions = pl.DataFrame({"lowerlevel": [0, 0, 1], "upperlevel": [1, 2, 2], "A": [1.0, 1.0, 1.0]})
    assert transition_count_of_level(dftransitions, 4) == [2, 2, 2, 0]
    assert transition_count_of_level(pl.DataFrame(), 3) == [0, 0, 0]
    with pytest.raises(ValueError, match="name level ids 0 to 2, but the ion has 2 levels"):
        transition_count_of_level(dftransitions, 2)


def test_write_transition_data_format():
    """The %-format writer gives the bytes of the f-string it replaced, for every column type."""
    dftransitions = pl.DataFrame(
        {
            "lowerlevel": [0, 1],
            "upperlevel": [1, 2],
            "A": [1.5e-3, 2.0e8],
            "coll_str": [-1.0, 3.5],
            "forbidden": [False, True],
        }
    )
    out = io.StringIO()
    write_transition_data(out, 26, 2, dftransitions, io.StringIO())

    expected = f"{26:7d}{2:7d}{2:12d}\n"
    expected += f"{1:4d} {2:4d} {1.5e-3:11.5e} {-1.0:9.2e} {0:d}\n"
    expected += f"{2:4d} {3:4d} {2.0e8:11.5e} {3.5:9.2e} {1:d}\n"
    expected += "\n"
    assert out.getvalue() == expected


def test_readhillierdata_bare_proton_has_one_state():
    """The H II placeholder level has g = 1: ARTIS divides the H I to H II Saha ratio by it."""
    ionization_energy_ev, dflevels, dftransitions = readhillierdata.read_levels_and_transitions(1, 2, io.StringIO())
    assert ionization_energy_ev == 0.0
    assert dflevels["g"].to_list() == [1.0]
    assert dftransitions.is_empty()


def test_get_level_valence_n_glued_digit_runs():
    """A digit run after an orbital letter splits into a count and an n by one rule.

    The count must fit the shell before it, the n must not start with 0, and the valence orbital
    must have l < n. A two-digit run that fails the rule is a two-digit n. A three-digit run
    tries a two-digit count first and a one-digit count second. The ADAS parser took the first digit alone as the
    count, so 4f145d gave n = 45. The Kurucz parser always split a three-digit run as 2 + 1, so
    s210d gave n = 0.
    """
    assert readadasdata.get_level_valence_n("4f145d_2De[3/2]_id=1") == 5
    assert readadasdata.get_level_valence_n("4f146s_2Se[1/2]_id=2") == 6
    assert readadasdata.get_level_valence_n("5s210d_2De[3/2]_id=3") == 10
    assert readadasdata.get_level_valence_n("3d104s_2Se[1/2]_id=4") == 4
    assert readadasdata.get_level_valence_n("3d24s_x") == 4
    assert readadasdata.get_level_valence_n("5s10d_x") == 10

    assert readkuruczdata.get_level_valence_n("s210d 2D,enpercm=1.0,j=0.5") == 10
    assert readkuruczdata.get_level_valence_n("f125d 2D,enpercm=1.0,j=0.5") == 5
    assert readkuruczdata.get_level_valence_n("s25p 3P,enpercm=1.0,j=0.0") == 5
    assert readkuruczdata.get_level_valence_n("s10d 1D,enpercm=1.0,j=2.0") == 10
    assert readkuruczdata.get_level_valence_n("p610s 2S,enpercm=1.0,j=0.5") == 10


def test_readdreamdata_rejects_an_unknown_level_type():
    """A level type other than (o) or (e) fails, and does not become even parity."""
    from artisatomic import readdreamdata

    row = {"Lower_Level": 0, "Lower_Type": "(o)", "Lower_g": 1}
    assert readdreamdata.energytuplefromrow(row, "Lower").parity == 1
    with pytest.raises(ValueError, match=r"level type '\(x\)'"):
        readdreamdata.energytuplefromrow({**row, "Lower_Type": "(x)"}, "Lower")


def test_groundstatesonlynist_names_a_missing_ion():
    """An ion that the NIST ground-state table lacks fails with the ion in the message."""
    from artisatomic import groundstatesonlynist

    with pytest.raises(ValueError, match="no row for Z=1 ion_stage 1"):
        groundstatesonlynist.read_ground_levels(1, 1, io.StringIO())


def test_photfilereader_short_block_and_unknown_type(tmp_path):
    """The reader stores a short tabulated block as read, and an unknown type does not clear the next block's name.

    Both cases have no blank line between the blocks. The reader used to fill a short block with
    zero rows up to the declared count, which the downsample then read as sorted energies. It
    also used to clear the name of the block after an unknown type on that block's own
    "!Configuration name" line.
    """
    from artisatomic.readhillierdata import PhotFileReader

    header = (
        "\n*****\n  header comment\n12-Oct-2009                             !Date\n"
        "3                                       !Number of energy levels\n"
        "2.0D0                                   !Screened nuclear charge\n"
        "5s2_5p6_1Se                             !Final state in ion\n"
        "Megabarns                               !Cross-section unit\n"
        "False                                   !Split J levels\n"
    )
    body = (
        "A                                       !Configuration name\n"
        "10                                      !Type of cross-section\n"
        "2                                       !Number of cross-section points\n"
        "1.0\n2.0\n"
        "B                                       !Configuration name\n"
        "20                                      !Type of cross-section\n"
        "3                                       !Number of cross-section points\n"
        "1.0 2.0\n1.5 1.0\n"
        "C                                       !Configuration name\n"
        "20                                      !Type of cross-section\n"
        "2                                       !Number of cross-section points\n"
        "1.0 4.0\n1.5 3.0\n"
    )
    photfile = tmp_path / "phot_test"
    photfile.write_text(header + body)

    flog = io.StringIO()
    levelindices = {"A": 0, "B": 1, "C": 2}
    reader = PhotFileReader(56, 2, 1, [911.0, 455.5, 300.0], levelindices, levelindices, flog)
    reader.read_file(0, photfile, photfile.name)

    assert reader.unknown_phixs_types == [10]
    assert reader.phixs_type_levels[10] == {"A"}
    assert reader.phixs_type_levels[20] == {"B", "C"}
    assert set(reader.phixstables[0]) == {"B", "C"}
    # the short block keeps its two rows, and the log says so. The file gives the energy as a
    # multiple of the level's threshold, and the table holds it in Rydberg.
    threshold_b = hc_in_ev_angstrom / 455.5 / ryd_to_ev
    assert reader.phixstables[0]["B"].shape == (2, 2)
    assert reader.phixstables[0]["B"][:, 0].tolist() == pytest.approx([1.0 * threshold_b, 1.5 * threshold_b])
    assert reader.phixstables[0]["B"][:, 1].tolist() == [2.0, 1.0]
    assert "B declares 3 cross section rows but the block ends after 2" in flog.getvalue()
    assert reader.phixstables[0]["C"].shape == (2, 2)
    assert reader.phixstables[0]["C"][:, 1].tolist() == [4.0, 3.0]
    # the reader accepts the D exponent of the screened nuclear charge, and 2 matches the ion stage
    assert "screened nuclear charge" not in flog.getvalue()


def test_photfilereader_excitation_energy_units(tmp_path):
    """The reader converts the excitation energy of the target, in either unit, and shifts the edge.

    The first phot file of an ion is CMFGEN's photoionisation route 1, and that route always has
    an excitation energy of zero (rdphot_gen_v2.f line 282). The reader therefore discards the
    header value of the first file. NIT/I/19apr23/phot_data_A writes 88.89 there, and CMFGEN
    reads none of it.

    A later file resolves the energy from the levels of the ion above, and the header value is
    the fallback. CMFGEN has no oscillator file for Ba III, so this Ba II file takes the fallback
    and logs it. The header value is in cm^-1 above 10 and in 10^15 Hz below it. The C II
    19apr23 phot_data_B file writes 52419.42 (cm^-1) and the O I one writes 0.804D0 (10^15 Hz).
    """
    from artisatomic.readhillierdata import excitation_energy_ev_of_header_value
    from artisatomic.readhillierdata import PhotFileReader

    assert excitation_energy_ev_of_header_value("52419.42") == pytest.approx(52419.42 * hc_in_ev_cm)
    assert excitation_energy_ev_of_header_value("0.804D0") == pytest.approx(0.804e15 * h_in_ev_seconds)
    assert excitation_energy_ev_of_header_value("0.0D0") == 0.0

    def read_one_file(excitationline: str, filenum: int = 1) -> tuple[PhotFileReader, str]:
        header = (
            "\n*****\n  header comment\n12-Oct-2009                             !Date\n"
            "1                                       !Number of energy levels\n"
            "2.0D0                                   !Screened nuclear charge\n"
            "5s2_5p6_1Se                             !Final state in ion\n"
            f"{excitationline}"
            "Megabarns                               !Cross-section unit\n"
            "False                                   !Split J levels\n"
        )
        body = (
            "A                                       !Configuration name\n"
            "20                                      !Type of cross-section\n"
            "2                                       !Number of cross-section points\n"
            "1.0 4.0\n2.0 1.0\n\n"
        )
        photfile = tmp_path / "phot_test"
        photfile.write_text(header + body)
        flog = io.StringIO()
        reader = PhotFileReader(56, 2, 2, [911.0], {"A": 0}, {"A": 0}, flog)
        reader.read_file(filenum, photfile, photfile.name)
        # the level block resolved the energy, so the sentinel of an unresolved file is gone
        assert reader.excitation_energy_ev is not None
        return reader, flog.getvalue()

    # no excitation line at all, so the edge is the level's own threshold wavelength
    plain, plainlog = read_one_file("")
    assert plain.excitation_energy_ev == 0.0
    assert plain.edge_lambda_angstrom() == 911.0
    assert "CMFGEN has no oscillator file for the ion above" in plainlog

    # the first file of an ion is route 1, so the reader discards a non-zero header value there
    firstfile, firstfilelog = read_one_file(
        "88.89                                   !Excitation energy of final state\n", filenum=0
    )
    assert firstfile.excitation_energy_ev == 0.0
    assert firstfile.edge_lambda_angstrom() == 911.0
    assert "the ion above" not in firstfilelog
    assert firstfile.phixstables[0]["A"][:, 0].tolist() == pytest.approx(
        [hc_in_ev_angstrom / 911.0 / ryd_to_ev, 2.0 * hc_in_ev_angstrom / 911.0 / ryd_to_ev]
    )

    gs_threshold_ev = hc_in_ev_angstrom / 911.0
    for excitationline, expected_ev in (
        ("52419.42                                !Excitation energy of final state\n", 52419.42 * hc_in_ev_cm),
        ("0.804D0                                 !Excitation energy of final state\n", 0.804e15 * h_in_ev_seconds),
        (
            "6.51014                                 !Excitation energy of final state (10^15 Hz)\n",
            6.51014e15 * h_in_ev_seconds,
        ),
    ):
        reader, log = read_one_file(excitationline)
        assert reader.excitation_energy_ev == pytest.approx(expected_ev)
        assert "CMFGEN has no oscillator file for the ion above" in log
        # the fits take the edge of the route, which is the ground-state edge plus the excitation
        assert reader.edge_lambda_angstrom() == pytest.approx(hc_in_ev_angstrom / (gs_threshold_ev + expected_ev))
        # the first column of a tabulated block is a multiple of that edge, not of the level's own
        expected_ryd = (gs_threshold_ev + expected_ev) / ryd_to_ev
        assert reader.phixstables[1]["A"][:, 0].tolist() == pytest.approx([expected_ryd, 2.0 * expected_ryd])


def test_photfilereader_negative_threshold_wavelength(tmp_path):
    """A level with a negative Lam(A) gets a table only where the excitation energy lifts its edge.

    CMFGEN writes a negative Lam(A) for a level above the ionisation limit. The edge of a route
    is EDGE = GS_EDGE + EXC_FREQ, and the reader keeps the sign of GS_EDGE. Route 1 has
    EXC_FREQ = 0, so such a level has an edge of GS_EDGE, which is below zero. A later route
    gives it the edge EXC_FREQ - |GS_EDGE|, which is above zero for a high enough target. An edge
    of zero or below needs a division by a negative edge, so the reader stores no table and
    counts the level in the summary of the file.
    """
    from artisatomic.readhillierdata import PhotFileReader

    def read_one_file(filenum: int) -> tuple[PhotFileReader, str]:
        header = (
            "\n*****\n  header comment\n12-Oct-2009                             !Date\n"
            "1                                       !Number of energy levels\n"
            "2.0D0                                   !Screened nuclear charge\n"
            "5s2_5p6_1Se                             !Final state in ion\n"
            "200000.0                                !Excitation energy of final state\n"
            "Megabarns                               !Cross-section unit\n"
            "False                                   !Split J levels\n"
        )
        body = (
            "A                                       !Configuration name\n"
            "20                                      !Type of cross-section\n"
            "2                                       !Number of cross-section points\n"
            "1.0 4.0\n2.0 1.0\n\n"
        )
        photfile = tmp_path / "phot_test"
        photfile.write_text(header + body)
        flog = io.StringIO()
        # CMFGEN has no oscillator file for Ba III, so the reader takes the header value
        reader = PhotFileReader(56, 2, 2, [-911.0], {"A": 0}, {"A": 0}, flog)
        reader.read_file(filenum, photfile, photfile.name)
        return reader, flog.getvalue()

    gs_edge_ev = hc_in_ev_angstrom / -911.0
    excitation_ev = 200000.0 * hc_in_ev_cm
    assert gs_edge_ev < 0.0 < excitation_ev + gs_edge_ev

    # a later route, whose excitation energy is above the size of the negative ground-state edge
    later, laterlog = read_one_file(1)
    assert not later.levels_without_edge
    assert "threshold energy of zero or below" not in laterlog
    edge_ev = float(later.phixstables[1]["A"][0, 0]) * ryd_to_ev
    assert edge_ev == pytest.approx(excitation_ev + gs_edge_ev)
    # the literal edge in eV, so a lost sign of Lam(A) fails here
    assert edge_ev == pytest.approx(11.18716, abs=1e-5)
    assert later.edge_lambda_angstrom() == pytest.approx(1108.2725, abs=1e-4)
    assert float(later.phixstables[1]["A"][1, 0]) == pytest.approx(2.0 * edge_ev / ryd_to_ev)
    assert later.phixstables[1]["A"][:, 1].tolist() == [4.0, 1.0]

    # the first file of the ion is route 1, whose excitation energy is zero
    first, firstlog = read_one_file(0)
    assert first.excitation_energy_ev == 0.0
    assert not first.phixstables[0]
    assert list(first.levels_without_edge) == ["A"]
    # the guard itself, which the reader calls at the start of every block of the level
    assert first.edge_lambda_angstrom() is None
    assert "1 level names of phot_test have a threshold energy of zero or below" in firstlog
    assert "The first is A." in firstlog


def test_readhillierdata_get_level_valence_n():
    """The last orbital of a CMFGEN configuration gives n; a merged shell gives its n; no orbital gives None."""
    assert readhillierdata.get_level_valence_n("2s2_2p3(4So)3p_5Pe[1]") == 3
    assert readhillierdata.get_level_valence_n("3d6(5D)4s_a6De[9/2]") == 4
    assert readhillierdata.get_level_valence_n("2s2_18w_2W") == 18
    assert readhillierdata.get_level_valence_n("3d5(4D)4po[3]") == 4
    # the seniority digit and parity letter of a term are not an orbital
    assert readhillierdata.get_level_valence_n("3d7_a2D2e[5/2]") == 3
    assert readhillierdata.get_level_valence_n("3d5_4s2_2F1e[5/2]") == 4
    assert readhillierdata.get_level_valence_n("1___") is None
    assert readhillierdata.get_level_valence_n("8SNG") is None


def test_cmfgen_fit_functions():
    """Each analytic fit gives its formula's value at the threshold and at one point above it.

    The grid is 1 + 20 x^2 times the threshold for x from 0 to 1 in steps of 0.001. Index 500 is
    therefore 6 times the threshold.
    """
    lambda_angstrom = 911.753  # the H I edge
    threshold_ev = hc_in_ev_angstrom / lambda_angstrom
    threshold_ryd = threshold_ev / ryd_to_ev

    # type 1: sigma_t (beta + (1 - beta) / u) u^-s
    table = readhillierdata.get_seaton_phixstable(lambda_angstrom, 2.0, 0.5, 3.0)
    assert table.shape == (1000, 2)
    assert table[0, 0] == pytest.approx(threshold_ryd)
    assert table[0, 1] == 2.0
    u = table[500, 0] / threshold_ryd
    assert u == pytest.approx(6.0)
    assert table[500, 1] == pytest.approx(2.0 * (0.5 + 0.5 / u) * u**-3.0)

    # type 7: the same fit with the edge moved up by nu_o, so the cross section is zero below it.
    # nu_o = E_th / h puts the edge at twice the threshold energy.
    nu_o_1e15hz = threshold_ev / h_in_ev_seconds / 1e15
    table7 = readhillierdata.get_seaton_phixstable(lambda_angstrom, 2.0, 0.5, 3.0, nu_o=nu_o_1e15hz)
    assert table7[0, 1] == 0.0
    assert table7[200, 1] == 0.0  # u = 1.8
    assert table7[300, 1] > 0.0  # u = 2.8

    # type 5: 10^(a + b x + c x^2 + d x^3) with x = log10(min(u, e)), times (e / u)^2 above e
    table5 = readhillierdata.get_opproject_phixstable(lambda_angstrom, 1.0, 0.5, 0.0, 0.0, 3.0)
    assert table5[0, 1] == pytest.approx(10.0)
    x = np.log10(3.0)
    assert table5[500, 1] == pytest.approx(10 ** (1.0 + 0.5 * x) * (3.0 / 6.0) ** 2)

    # type 4: a polynomial in 1 / u, from the C IV 5s_2Se block of phot_data_A. At the threshold
    # it is the sum of the coefficients. A negative value of the polynomial becomes zero.
    coefficients = (-3.780e-3, 5.227e-2, 1.949, 1.223, -2.259, 1.122)
    table4 = readhillierdata.get_leibowitz_phixstable(lambda_angstrom, *coefficients)
    assert table4[0, 1] == pytest.approx(sum(coefficients))
    ru = 1 / 6.0
    expected4 = sum(coefficient * ru**power for power, coefficient in enumerate(coefficients))
    assert table4[500, 1] == pytest.approx(expected4)
    assert np.all(readhillierdata.get_leibowitz_phixstable(lambda_angstrom, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0)[:, 1] == 0.0)
    assert readhillierdata.phixs_fit_functions[4] == (6, readhillierdata.get_leibowitz_phixstable)

    # type 6: a cubic in x = log10(u) below the break e, a straight line 10^(f + g x) above it
    table6 = readhillierdata.get_hummer_phixstable(lambda_angstrom, 1.0, -1.0, 0.0, 0.0, 0.5, 0.0, -2.0, 0.0)
    assert table6[0, 1] == pytest.approx(10.0)
    assert table6[500, 1] == pytest.approx(6.0**-2.0)

    # type 9: the H I ground-state fit of Verner & Yakovlev (1995, A&AS, 109, 125-133, bibcode
    # 1995A&AS..109..125V) gives the 6.3 Mb threshold cross section
    fit = readhillierdata.VY95PhixsFitRow(
        n=1, l=0, E_th_eV=13.6, E_0=0.4298, sigma_0=5.475e4, y_a=32.88, P=2.963, y_w=0.0
    )
    table9 = readhillierdata.get_vy95_phixstable(lambda_angstrom, [fit])
    y = threshold_ev / 0.4298
    q = 5.5 - 0.5 * 2.963
    expected = 5.475e4 * (y - 1) ** 2 * y**-q * (1 + np.sqrt(y / 32.88)) ** -2.963
    assert table9[0, 1] == pytest.approx(expected)
    assert table9[0, 1] == pytest.approx(6.3, rel=0.02)


def test_read_adf04_selects_the_nearest_temperature():
    """The collision strengths come from the tabulated temperature nearest to -electrontemperature.

    The Co III grid runs from 3150 K to 7590 K in steps of about 5 percent. 6000 K therefore
    selects the 6030 K column, and a low temperature selects the first column. A hard-coded string
    per element chose 5010 K before, whatever the command line said.
    """
    flog = io.StringIO()
    _, _, upsilons_6000, _ = readadasdata.read_adf04(adf04_sample_path(), flog, 6000.0, 27, 3)
    assert "The collision strengths are the values at 6030 K." in flog.getvalue()

    flog = io.StringIO()
    _, _, upsilons_low, _ = readadasdata.read_adf04(adf04_sample_path(), flog, 1000.0, 27, 3)
    assert "The collision strengths are the values at 3150 K." in flog.getvalue()

    assert set(upsilons_6000) == set(upsilons_low)
    assert any(upsilons_6000[key] != upsilons_low[key] for key in upsilons_6000)

    assert readadasdata.adf04_number("5.01+03") == 5010.0
    assert readadasdata.adf04_number("1.00-02") == 0.01


def test_readhillierdata_warns_when_ground_lambda_disagrees_with_header(monkeypatch, tmp_path):
    """The ground level's Lam(A) must give the header's ionisation energy to four significant figures.

    The header value is the one adata.txt gets. A difference above that precision means the header
    and the level table disagree, which the log file must say. The test copies the H I file with
    its header value raised by one percent; the unchanged file gives no warning.
    """
    import contextlib

    ionfiles = readhillierdata.ions_data[1, 1]
    with xopen_check_extension(readhillierdata.hillier_osc_filename(1, 1)) as fosc:
        lines = fosc.readlines()
    for index, line in enumerate(lines):
        if line.rstrip().endswith("!Ionization energy"):
            value = line.split()[0]
            lines[index] = line.replace(value, f"{float(value) * 1.01:.4f}", 1)
            break
    else:
        pytest.fail("the H I oscillator file has no '!Ionization energy' line")
    (tmp_path / "hi_osc.dat").write_text("".join(lines))

    flog = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()):
        ionization_energy_ev, _, _ = readhillierdata.read_levels_and_transitions(1, 1, flog)
    assert "WARNING: the ground level Lam(A)" not in flog.getvalue()

    # an absolute folder path replaces the ion folder in the joined path
    monkeypatch.setitem(
        readhillierdata.ions_data,
        (1, 1),
        ionfiles._replace(folder=str(tmp_path), levelstransitionsfilename="hi_osc.dat"),
    )
    flog = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()):
        ionization_energy_raised_ev, _, _ = readhillierdata.read_levels_and_transitions(1, 1, flog)
    assert ionization_energy_raised_ev == pytest.approx(ionization_energy_ev * 1.01, rel=1e-5)
    assert "WARNING: the ground level Lam(A)" in flog.getvalue()


def test_readhillierdata_rejects_a_file_with_no_ionization_energy(monkeypatch, tmp_path):
    """A file with no '!Ionization energy' line fails with that message, not with a division by zero."""
    import contextlib

    ionfiles = readhillierdata.ions_data[1, 1]
    with xopen_check_extension(readhillierdata.hillier_osc_filename(1, 1)) as fosc:
        lines = [line for line in fosc.readlines() if not line.rstrip().endswith("!Ionization energy")]
    (tmp_path / "hi_osc.dat").write_text("".join(lines))
    monkeypatch.setitem(
        readhillierdata.ions_data,
        (1, 1),
        ionfiles._replace(folder=str(tmp_path), levelstransitionsfilename="hi_osc.dat"),
    )
    with contextlib.redirect_stdout(io.StringIO()), pytest.raises(ValueError, match="no '!Ionization energy' line"):
        readhillierdata.read_levels_and_transitions(1, 1, io.StringIO())


def test_clear_files_removes_phixsdata_with_nophixs(tmp_path):
    """--nophixs writes no cross sections, so clear_files() removes phixsdata_v2.txt of an earlier run.

    The level ids in that file belong to the earlier adata.txt, so the file must not stay beside a
    new one.
    """
    from artisatomic.output import clear_files

    phixspath = tmp_path / "phixsdata_v2.txt"
    phixspath.write_text("100\n 3.0000000e-02\n26 2 0 1 10 1.0\n", encoding="utf-8")
    (tmp_path / "adata.txt").write_text("an earlier run\n", encoding="utf-8")

    clear_files(phixs_args(nophixs=True, output_folder=str(tmp_path)))

    assert not phixspath.exists()
    # the other two files always start again, with their file comment only
    for filename in ("adata.txt", "transitiondata.txt"):
        lines = (tmp_path / filename).read_text(encoding="utf-8").splitlines()
        assert lines
        assert all(line.startswith("#") for line in lines)
        assert "an earlier run" not in lines
    transitioncomment = (tmp_path / "transitiondata.txt").read_text(encoding="utf-8")
    assert "temperature closest to 6000 K" in transitioncomment
    # each file comment explains each field of its file
    for field in ("Z", "ion_stage", "ntransitions", "lower, upper", "A", "coll_str", "forbidden"):
        assert f"\n# {field} " in transitioncomment, field
    adatacomment = (tmp_path / "adata.txt").read_text(encoding="utf-8")
    # each file comment says that its level numbers and ion stages start at 1 and not at 0
    for filecomment in (transitioncomment, adatacomment):
        assert "\n# NUMBERS START AT 1\n" in filecomment
        assert "start at 1, and not at 0" in filecomment
    for field in ("Z", "ion_stage", "nlevels", "ionisation_energy", "level_number", "energy", "g", "level_name"):
        assert f"\n# {field} " in adatacomment, field
    # a folder with no phixsdata_v2.txt is fine too
    clear_files(phixs_args(nophixs=True, output_folder=str(tmp_path)))
    assert not phixspath.exists()

    # a run that writes cross sections truncates the file and writes the header for the ions
    clear_files(phixs_args(nophixs=False, output_folder=str(tmp_path), optimaltemperature=5500))
    lines = phixspath.read_text(encoding="utf-8").splitlines()
    # ARTIS reads the first two numbers with no comment skip, so the file comment comes after them
    assert lines[:2] == ["100", " 3.0000000e-02"]
    assert len(lines) > 2
    assert all(line.startswith("#") for line in lines[2:])
    assert any("T=5500 K (option -optimaltemperature)" in line for line in lines)
    assert "# NUMBERS START AT 1" in lines
    assert any("Only the point number i of a table starts at 0" in line for line in lines)
    assert phixspath.read_text(encoding="utf-8").isascii()


@pytest.mark.parametrize(
    ("modulename", "pathname", "handler", "dataname", "strayname"),
    [
        ("readtanakajpltdata", "jpltpath", "tanakajplt", "26_1.txt.zst", "26_1 2.txt.zst"),
        ("readadasdata", "adaspath", "adas", "38_1.adf04.zst", "38_1 2.adf04.zst"),
    ],
)
def test_extend_ion_list_skips_a_file_name_that_names_no_ion(
    modulename, pathname, handler, dataname, strayname, tmp_path, monkeypatch, capsys
):
    """A stray file that the glob matches gives a warning, and the ion selection continues.

    A sync client leaves a conflict copy, e.g. "26_1 2.txt.zst", beside the data file. int() on
    the parts of that name stopped every run that used the built-in ion selection. The message
    named neither the reader nor the file.
    """
    module = importlib.import_module(f"artisatomic.{modulename}")
    (tmp_path / dataname).touch()
    (tmp_path / strayname).touch()
    monkeypatch.setattr(module, pathname, tmp_path)

    result = module.extend_ion_list([])

    atomic_number = int(dataname.split("_")[0])
    assert result == [(atomic_number, [(1, handler)])]
    assert strayname in capsys.readouterr().out


def test_read_adas_levels_and_transitions_sorts_the_level_ids(tmp_path, monkeypatch):
    """A collision row that gives the lower level first still gives a transition with the lower id first.

    read_adf04() sorts each collision pair, so a reversed transition pair matches no upsilon. The
    join in write_output_files() then misses, and write_transition_data() raises after adata.txt
    already holds the ion.
    """
    import contextlib

    adf04 = make_adf04(
        [
            "    1          4p65s2(1S)   (1)0( 0.0)            0.0000",
            "    2       4p65s15p1(3P)   (3)1( 0.0)        14317.5023",
        ],
        ["   1   2 1.00+08 5.00-01 5.00-01"],
        header="Xx+ 0        99         1     45932.2036(  )",
        temperatures=" 1.00    3       1.00+03 1.00+04",
    )
    (tmp_path / "99_1.adf04").write_text(adf04, encoding="utf-8")
    monkeypatch.setattr(readadasdata, "adaspath", tmp_path)

    with contextlib.redirect_stdout(io.StringIO()):
        _, _, adas_transitions, upsilondict = readadasdata.read_adas_levels_and_transitions(
            99, 1, io.StringIO(), phixs_args()
        )

    # the ids are zero-based in memory, and both the transition and the upsilon name the same pair
    assert list(upsilondict) == [(0, 1)]
    assert not isinstance(adas_transitions, pl.DataFrame)  # this reader returns a list of rows
    assert [(tr.lowerlevel, tr.upperlevel) for tr in adas_transitions] == [(0, 1)]


def test_get_ion_handlers_builds_the_built_in_selection(tmp_path, monkeypatch):
    """A run with no ion handlers file selects ions that compositiondata.txt can hold.

    Every tests/*/ set supplies an ion handlers file, so no checksum set takes this branch. The
    test pins the properties of the selection and not the number of ions, which the available
    data sets decide.
    """
    import contextlib

    from artisatomic.base import check_ion_stages_contiguous
    from artisatomic.base import sort_ion_handlers
    from artisatomic.ionhandlers import get_ion_handlers

    if not Path(readhillierdata.hillier_ion_folder(26, 2)).is_dir():
        pytest.skip("the CMFGEN data set is not available here")
    # the built-in selection asks four readers, and the Floers+25 reader stops without its data
    if not readfloers25data.get_basepath(withforbidden=False).is_dir():
        pytest.skip("the Floers+25 test sample is not available here")

    # a directory with no artisatomicionhandlers.json, so the function builds the selection
    monkeypatch.chdir(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        ion_handlers = get_ion_handlers(1, 5, None)
        unlimited = get_ion_handlers(None, None, None)

    assert ion_handlers
    # compositiondata.txt gives the lowest and the highest ion stage of an element, and no list
    check_ion_stages_contiguous(ion_handlers)
    assert all(1 <= ion_stage <= 5 for _, listions in ion_handlers for ion_stage, _ in listions)
    assert ion_handlers == sort_ion_handlers(ion_handlers)

    # the limits remove ions. They never add one, and they never change the handler of an ion
    ions = {(atomic_number, ion) for atomic_number, listions in ion_handlers for ion in listions}
    ions_unlimited = {(atomic_number, ion) for atomic_number, listions in unlimited for ion in listions}
    assert ions <= ions_unlimited


def test_log_comment_records_for_an_ionlog_only():
    """A plain stream gets the log line and records nothing, so the older callers stay valid."""
    from artisatomic.base import IonLog
    from artisatomic.base import log_comment

    stream = io.StringIO()
    flog = IonLog(stream)
    log_comment(flog, ("adata", "phixsdata"), "Reading a file")
    assert stream.getvalue() == "Reading a file\n"
    assert flog.comments == {"adata": ["Reading a file"], "transitiondata": [], "phixsdata": ["Reading a file"]}

    # a second IonLog with the same dictionary adds to the same lists, as the write pass does
    log_comment(IonLog(io.StringIO(), flog.comments), ("adata",), "second pass")
    assert flog.comments["adata"] == ["Reading a file", "second pass"]

    plainstream = io.StringIO()
    log_comment(plainstream, ("adata",), "Reading a file")
    assert plainstream.getvalue() == "Reading a file\n"


def test_log_source_labels_the_log_line_and_records_a_plain_source_line():
    """Each pass logs its own source, so the log says what the source is for. The block needs "source:"."""
    from artisatomic.base import IonLog
    from artisatomic.base import log_source

    stream = io.StringIO()
    flog = IonLog(stream)
    log_source(flog, ("phixsdata",), "the cross sections", "a data set")
    assert stream.getvalue() == "source of the cross sections: a data set\n"
    assert flog.comments == {"adata": [], "transitiondata": [], "phixsdata": ["source: a data set"]}

    plainstream = io.StringIO()
    log_source(plainstream, ("adata",), "the levels", "a data set")
    assert plainstream.getvalue() == "source of the levels: a data set\n"


def test_read_adf04_logs_the_file_before_its_origin(tmp_path):
    """The block names the file first, and then the sentence about who made it."""
    from artisatomic import readadasdata
    from artisatomic.base import IonLog

    filepath = write_hydrogen_adf04(tmp_path, ["   2   1 1.00+08 5.00-01 5.00-01"])
    flog = IonLog(io.StringIO())
    readadasdata.read_adf04(
        filepath, flog, 5000.0, 1, 1, contents="The levels and the collision strengths", origin="A group made the file."
    )
    lines = flog.comments["adata"]
    assert lines[0].startswith("The levels and the collision strengths come from ")
    assert lines[0].endswith("/1_1.adf04.")
    assert lines[1] == "A group made the file."
    assert flog.comments["transitiondata"][:2] == lines[:2]


def test_write_comment_block_gives_every_part_of_a_line_a_hash():
    """ARTIS reads a line with no # as data, so a line break in a log line must not end the comment."""
    from artisatomic.base import IonLog
    from artisatomic.output import write_comment_block

    flog = IonLog(io.StringIO())
    flog.comments["transitiondata"].extend(["Temperatures:\n0.1, 0.2", "source: a data set"])
    out = io.StringIO()
    write_comment_block(out, "transitiondata", ("Z=26 Fe II",), flog)
    # the source line comes directly after the title lines, wherever a reader recorded it
    assert out.getvalue() == "# Z=26 Fe II\n# source: a data set\n# Temperatures:\n# 0.1, 0.2\n"

    # no title line and a plain stream: the writer tests that pin the whole output depend on this
    out = io.StringIO()
    write_comment_block(out, "transitiondata", (), io.StringIO())
    assert not out.getvalue()


def test_write_comment_block_needs_exactly_one_source_line():
    """A block with no source line means that a reader does not state its source, so the run must stop."""
    from artisatomic.base import IonLog
    from artisatomic.output import write_comment_block

    flog = IonLog(io.StringIO())
    flog.comments["phixsdata"].append("Reading a file")
    with pytest.raises(ValueError, match="needs one source line but has 0"):
        write_comment_block(io.StringIO(), "phixsdata", ("Z=26 Fe II", "handler: cmfgen"), flog)

    flog.comments["phixsdata"] += ["source: a data set", "source: a second data set"]
    with pytest.raises(ValueError, match="needs one source line but has 2"):
        write_comment_block(io.StringIO(), "phixsdata", ("Z=26 Fe II", "handler: cmfgen"), flog)


def artis_noncommentline(lines: list[str], pos: int) -> int:
    """Return the index of the next line that get_noncommentline() of ARTIS (input.h) returns."""
    while not lines[pos].strip() or lines[pos].lstrip().startswith("#"):
        pos += 1
    return pos


def two_level_iondata(ion_stage: int, nphixspoints: int, *, has_phixs: bool, is_top_ion: bool = False):
    """Build an ion with two levels and one collision strength, with or without cross section tables."""
    import dataclasses

    from artisatomic.base import empty_comments

    comments = empty_comments()
    # the source line is not the first line here, and the writer must put it first
    comments["adata"] += ["Reading osc_data", "source: the source of the levels"]
    comments["transitiondata"] += ["source: the source of the levels", "Temperatures:\n0.1, 0.2"]
    comments["phixsdata"] += ["source: the source of the cross sections"]
    return dataclasses.replace(
        make_iondata(ion_stage, is_top_ion=is_top_ion),
        dfenergylevels=pl.DataFrame(
            {
                "levelid": [0, 1],
                "energyabovegsinpercm": [0.0, 1000.0],
                "g": [9.0, 7.0],
                "parity": [0, 1],
                "levelname": [f"gs{ion_stage}", f"excited{ion_stage} # not a comment"],
            }
        ),
        upsilondict={(0, 1): 0.5},
        photoionization_crosssections=np.ones((2, nphixspoints)) if has_phixs else np.empty((0, nphixspoints)),
        # one target for level id 0 and two targets for level id 1, so both table forms occur
        photoionization_targetfractions=[[(0, 1.0)], [(0, 0.25), (1, 0.75)]] if has_phixs else [],
        photoionization_thresholds_ev=np.array([10.0, 9.0]) if has_phixs else np.empty(0),
        comments=comments,
    )


def test_output_files_with_comment_blocks_follow_the_artis_read_rules(tmp_path):
    """Read the output files as ARTIS does (input.cc), so a comment at a wrong position fails here.

    ARTIS skips a comment line only before the header of an ion or of a cross section table.
    Inside a block it takes a fixed count of lines, whatever they hold.
    """
    from artisatomic.output import clear_files
    from artisatomic.output import write_output_files

    nphixspoints = 3
    tmpargs = phixs_args(output_folder=str(tmp_path), nphixspoints=nphixspoints, phixsnuincrement=0.1)

    # Fe II has no cross section table, and it is not the top ion
    ionstages = [1, 2, 3, 4]
    iondatalist = [
        two_level_iondata(ion_stage, nphixspoints, has_phixs=ion_stage in {1, 3}, is_top_ion=ion_stage == 4)
        for ion_stage in ionstages
    ]
    clear_files(tmpargs)
    write_output_files(26, iondatalist, tmpargs)

    # all ions share one log file, and each pass names its ion
    logtext = (tmp_path / "artisatomiclog.txt").read_text(encoding="utf-8")
    for ionstr in ("Fe I", "Fe II", "Fe III", "Fe IV"):
        assert logtext.count(f"Z=26 {ionstr} output:") == 1

    adatatext = (tmp_path / "adata.txt").read_text(encoding="utf-8")
    assert "# Z=26 Fe II\n# handler: cmfgen\n# source: the source of the levels\n# Reading osc_data\n" in adatatext
    lines = adatatext.splitlines()
    pos = 0
    for ion_stage in ionstages:
        pos = artis_noncommentline(lines, pos)
        atomic_number, ion_stage_in, nlevels, _ionpot = lines[pos].split()
        assert (int(atomic_number), int(ion_stage_in), int(nlevels)) == (26, ion_stage, 2)
        for levelnumber, levelline in enumerate(lines[pos + 1 : pos + 1 + 2], start=1):
            levelnumber_in, energy, g, ntransitions = levelline.split()[:4]
            assert (int(levelnumber_in), float(g), int(ntransitions)) == (levelnumber, [9.0, 7.0][levelnumber - 1], 1)
            assert float(energy) >= 0.0
        pos += 1 + 2

    transitiontext = (tmp_path / "transitiondata.txt").read_text(encoding="utf-8")
    assert "# Temperatures:\n# 0.1, 0.2\n" in transitiontext
    assert "# source: the source of the levels\n" in transitiontext
    lines = transitiontext.splitlines()
    pos = 0
    for ion_stage in ionstages:
        pos = artis_noncommentline(lines, pos)
        assert [int(field) for field in lines[pos].split()] == [26, ion_stage, 1]
        # ARTIS counts the columns of the first row, so the row must hold five numbers and no more
        assert [float(field) for field in lines[pos + 1].split()] == [1.0, 2.0, 0.0, 0.5, 0.0]
        pos += 1 + 1

    phixstext = (tmp_path / "phixsdata_v2.txt").read_text(encoding="utf-8")
    assert phixstext.count("# source: the source of the cross sections\n") == 2
    # an ion with no table gets no block, because a block must come directly before a table header
    assert "# Z=26 Fe II\n" not in phixstext
    assert not phixstext.splitlines()[-1].startswith("#")
    # the file holds the two grid numbers and the tables of each ion already
    assert "Downsample" not in phixstext
    assert "Writing" not in phixstext
    lines = phixstext.splitlines()
    # ARTIS reads the first two numbers with no comment skip
    assert int(lines[0]) == nphixspoints
    assert float(lines[1]) == 0.1
    pos = 2
    tables = []
    while any(line.strip() and not line.lstrip().startswith("#") for line in lines[pos:]):
        pos = artis_noncommentline(lines, pos)
        _, upperionstage, targetlevel, lowerionstage, lowerlevel, _threshold = lines[pos].split()
        assert int(upperionstage) == int(lowerionstage) + 1
        pos += 1
        if int(targetlevel) == -1:
            pos = artis_noncommentline(lines, pos)
            ntargets = int(lines[pos])
            pos += 1
            for _ in range(ntargets):
                pos = artis_noncommentline(lines, pos)
                assert len(lines[pos].split()) == 2
                pos += 1
        # ARTIS reads the points with >>, which stops at a #
        assert [float(line) for line in lines[pos : pos + nphixspoints]] == [1.0] * nphixspoints
        pos += nphixspoints
        tables.append((int(lowerionstage), int(lowerlevel)))
    assert tables == [(1, 1), (1, 2), (3, 1), (3, 2)]


def test_write_comment_block_writes_ascii_only():
    """A program that opens an output file with the encoding of an ASCII locale must not stop on an author name."""
    from artisatomic.base import IonLog
    from artisatomic.output import write_comment_block

    flog = IonLog(io.StringIO())
    flog.comments["adata"].append("source: Flörs, A., Martínez-Pinedo, G., Kitovienė, L.")
    out = io.StringIO()
    write_comment_block(out, "adata", (), flog)
    assert out.getvalue() == "# source: Flors, A., Martinez-Pinedo, G., Kitoviene, L.\n"
    assert out.getvalue().isascii()


def test_ionlog_drops_the_comments_of_a_read_that_runs_again():
    """The CMFGEN reader reads a file a second time after a rewrite as utf-8, and no comment line must occur twice."""
    from artisatomic.base import IonLog
    from artisatomic.base import log_comment

    flog = IonLog(io.StringIO())
    log_comment(flog, ("adata",), "source: a data set")
    counts = flog.comment_counts()
    log_comment(flog, ("adata", "transitiondata"), "Reading osc_data")
    flog.drop_comments_after(counts)
    log_comment(flog, ("adata", "transitiondata"), "Reading osc_data")
    assert flog.comments == {
        "adata": ["source: a data set", "Reading osc_data"],
        "transitiondata": ["Reading osc_data"],
        "phixsdata": [],
    }


def test_cmfgen_reader_records_each_comment_one_time_after_the_utf8_retry(monkeypatch):
    """The run that rewrites a file as utf-8 must give the same comment block as each later run."""
    from artisatomic.base import IonLog
    from artisatomic.base import log_comment

    attempts = []

    def fake_read(_atomic_number, _ion_stage, flog):
        log_comment(flog, ("adata", "transitiondata"), "Reading osc_data")
        attempts.append(1)
        if len(attempts) == 1:
            encoding = "utf-8"
            raise UnicodeDecodeError(encoding, b"\xed", 0, 1, "invalid continuation byte")
        return 1.0, pl.DataFrame(), pl.DataFrame()

    monkeypatch.setattr(readhillierdata, "read_levels_and_transitions_from_file", fake_read)
    monkeypatch.setattr(readhillierdata, "rewrite_file_as_utf8", lambda _filename: True)

    flog = IonLog(io.StringIO())
    readhillierdata.read_levels_and_transitions(26, 2, flog)
    assert len(attempts) == 2
    assert flog.comments["adata"] == ["Reading osc_data"]
    assert flog.comments["transitiondata"] == ["Reading osc_data"]


def test_each_reader_of_the_registry_takes_a_log():
    """read_ion_data() gives each reader (atomic_number, ion_stage, flog), so each one must take a third argument."""
    import inspect

    from artisatomic.iondata import handlers

    for name, handler in handlers.items():
        parameters = list(inspect.signature(handler.read_levels_and_transitions).parameters)
        assert parameters[:3] == ["atomic_number", "ion_stage", "flog"], name
        assert handler.description, name


def test_hydrogenic_estimate_records_its_source_only_with_a_table():
    """An ion that got no hydrogenic table must not name the estimate as the source of its cross sections."""
    from artisatomic.base import IonLog

    levels = pl.DataFrame({"levelid": [0], "energyabovegsinpercm": [0.0], "g": [2.0], "levelname": ["3s_2Se"]})
    args = phixs_args()

    flog = IonLog(io.StringIO())
    crosssections, _, _ = match_hydrogenic_phixs(11, levels, 5.139, "mons", None, args, flog)
    assert len(crosssections) == 0
    assert flog.comments["phixsdata"] == []

    flog = IonLog(io.StringIO())
    crosssections, _, _ = match_hydrogenic_phixs(
        11, levels, 5.139, "cmfgen", readhillierdata.get_level_valence_n, args, flog
    )
    assert len(crosssections) == 1
    assert len(flog.comments["phixsdata"]) == 1
    sourceline = flog.comments["phixsdata"][0]
    assert sourceline.startswith("source: the hydrogenic estimate of artisatomic")
    assert "gbf_n_data.dat" in sourceline
    assert "/Users/" not in sourceline


def run_main_with_no_ion_read(monkeypatch, outputfolder) -> None:
    """Run cli.main() with no read of an ion. The read needs the data sets, and the callers test the files of the run only."""
    from artisatomic import cli

    monkeypatch.setattr(cli, "process_files", lambda _ion_handlers, _args: None)
    monkeypatch.setattr("sys.argv", ["makeartisatomicfiles", "-output_folder", str(outputfolder)])
    cli.main()


def test_main_writes_one_log_file_and_the_handlers_record_beside_the_output_files(tmp_path, monkeypatch):
    """The log of all ions is artisatomiclog.txt, and it sits with the record of the ion handlers beside adata.txt."""
    handlers = [[38, [[1, "kurucz"], [2, "kurucz"]]]]
    (tmp_path / "artisatomicionhandlers.json").write_text(json.dumps(handlers), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    outputfolder = tmp_path / "artis_files"
    outputfolder.mkdir()
    # a log of an earlier run must not stay in the file
    (outputfolder / "artisatomiclog.txt").write_text("the log of an earlier run\n", encoding="utf-8")
    run_main_with_no_ion_read(monkeypatch, outputfolder)

    assert {path.name for path in outputfolder.iterdir()} == {
        "adata.txt",
        "compositiondata.txt",
        "transitiondata.txt",
        "phixsdata_v2.txt",
        "artisatomiclog.txt",
        "artisatomicionhandlers_used.json",
    }
    assert json.loads((outputfolder / "artisatomicionhandlers_used.json").read_text(encoding="utf-8")) == handlers
    assert not (outputfolder / "artisatomiclog.txt").read_text(encoding="utf-8")


def test_main_into_the_working_directory_leaves_the_input_file_of_the_ion_handlers_alone(tmp_path, monkeypatch):
    """get_ion_handlers() reads ./artisatomicionhandlers.json, so the record of a run must not get that name."""
    from artisatomic import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "get_ion_handlers", lambda **_limits: [(38, [(1, "kurucz"), (2, "kurucz")])])
    run_main_with_no_ion_read(monkeypatch, ".")

    assert not (tmp_path / "artisatomicionhandlers.json").exists()
    record = json.loads((tmp_path / "artisatomicionhandlers_used.json").read_text(encoding="utf-8"))
    assert record == [[38, [[1, "kurucz"], [2, "kurucz"]]]]


def test_main_removes_the_log_folder_of_an_earlier_release(tmp_path):
    """The per-ion logs of an earlier release must not stay beside the log file of a new run."""
    from artisatomic.cli import remove_old_log_folder

    oldfolder = tmp_path / "atomic_data_logs"
    oldfolder.mkdir()
    (oldfolder / "fe2.txt").write_text("an old log\n", encoding="utf-8")
    (oldfolder / "artisatomicionhandlers.json").write_text("[]", encoding="utf-8")
    remove_old_log_folder(tmp_path)
    assert not oldfolder.exists()

    # a file that the earlier release did not write stays, and so does its folder
    oldfolder.mkdir()
    (oldfolder / "fe2.txt").write_text("an old log\n", encoding="utf-8")
    (oldfolder / "notes.md").write_text("my notes\n", encoding="utf-8")
    remove_old_log_folder(tmp_path)
    assert [path.name for path in oldfolder.iterdir()] == ["notes.md"]

    # no folder is fine too
    remove_old_log_folder(tmp_path / "artis_files")
    assert not (tmp_path / "artis_files").exists()


def test_file_comment_gives_the_creation_time_in_utc(tmp_path, monkeypatch):
    """Each output file names its creation time, and the time of a checksum run is the same for each run."""
    import re

    from artisatomic import base
    from artisatomic.output import clear_files

    # CI sets the test mode for all tests, and the test mode comes before SOURCE_DATE_EPOCH
    monkeypatch.setattr(base, "TESTMODE", False)
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1790000000")
    clear_files(phixs_args(output_folder=str(tmp_path)))
    for filename in ("adata.txt", "transitiondata.txt", "phixsdata_v2.txt"):
        assert "wrote this file at 2026-09-21T14:13:20Z (UTC)." in (tmp_path / filename).read_text(encoding="utf-8")

    for badvalue in ("", "abc", "1e9", "-1", "99999999999999999999"):
        monkeypatch.setenv("SOURCE_DATE_EPOCH", badvalue)
        with pytest.raises(ValueError, match="SOURCE_DATE_EPOCH must be a count of seconds"):
            base.creation_time_utc()

    # The test mode gives a time of zero, so the checksum recipe needs no other variable. A build
    # environment can set SOURCE_DATE_EPOCH for its own use, and that must not change the checksums.
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1790000000")
    monkeypatch.setattr(base, "TESTMODE", True)
    assert base.creation_time_utc() == "1970-01-01T00:00:00Z"

    # a normal run gives the time of the run
    monkeypatch.setattr(base, "TESTMODE", False)
    monkeypatch.delenv("SOURCE_DATE_EPOCH")
    assert re.fullmatch(r"20\d{2}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", base.creation_time_utc())


def test_to_ascii_keeps_a_dash_and_the_letter_of_an_accent():
    """A character with no ASCII form must not go away, because the text on its two sides would then join."""
    from artisatomic.base import comment_lines
    from artisatomic.base import to_ascii

    # chr() and not the characters, because ruff rejects a string with a character that looks like an ASCII one
    endash, minussign, timessign, greaterequal, alpha, lineseparator = (
        chr(0x2013),
        chr(0x2212),
        chr(0x00D7),
        chr(0x2265),
        chr(0x03B1),
        chr(0x2028),
    )
    # the JPLT files of Pr II and Tb II name these two ranges of elements, with an en dash and a minus sign
    assert to_ascii(f"Elements. I. Pr{endash}Gd") == "Elements. I. Pr-Gd"
    assert to_ascii(f"Elements. II. Tb{minussign}Yb") == "Elements. II. Tb-Yb"
    assert to_ascii("Flörs, Martínez-Pinedo, Kitovienė") == "Flors, Martinez-Pinedo, Kitoviene"
    assert to_ascii(f"1{timessign}10 and {greaterequal}5 and {alpha}") == "1x10 and >=5 and ?"
    for text in (f"Pr{endash}Gd", f"a{lineseparator}b", alpha, ""):
        for commentline in comment_lines([text]):
            assert commentline.startswith("#")
            assert commentline.endswith("\n")
            assert commentline.count("\n") == 1
            assert commentline.isascii()


def test_path_in_data_folder_starts_with_the_repository_name_of_the_folder(tmp_path):
    """A data folder can be a symbolic link, and the name of the link target must not go into an output file."""
    from artisatomic.base import path_in_data_folder

    target = tmp_path / "bulk_disk" / "kurucz_2024"
    (target / "zztar").mkdir(parents=True)
    datafolder = tmp_path / "repository" / "atomic-data-kurucz"
    datafolder.parent.mkdir()
    datafolder.symlink_to(target, target_is_directory=True)

    # a reader has the path of the file after resolve(), so it holds the name of the link target
    resolvedfile = (datafolder / "zztar" / "gf3800.all").resolve()
    assert "kurucz_2024" in str(resolvedfile)
    assert path_in_data_folder(resolvedfile, datafolder) == "atomic-data-kurucz/zztar/gf3800.all"

    # a reader takes a plain file or a compressed file, and the output must be the same for the two
    compressedfile = (datafolder / "zztar" / "gf3800.all.zst").resolve()
    assert path_in_data_folder(compressedfile, datafolder) == "atomic-data-kurucz/zztar/gf3800.all"

    # a file that is not in the data folder keeps its own path, with no folder name before it
    otherfile = tmp_path / "repository" / "other.txt"
    assert path_in_data_folder(otherfile, datafolder) == str(otherfile)


def test_ion_label():
    """The comment blocks and the log file name an ion in the same way."""
    from artisatomic.base import ion_label

    assert ion_label(26, 2) == "Z=26 Fe II"


def test_resolve_pass_records_a_target_that_matches_no_level(tmp_path):
    """The fallback to the ground state sets the upper level of the tables, so the comment block must show it."""
    from artisatomic.iondata import resolve_photoion_targetfractions

    lower = make_iondata(1, is_top_ion=False, targetconfigs=[[("no such level", 1.0)]])
    logpath = tmp_path / "artisatomiclog.txt"
    resolve_photoion_targetfractions([lower, make_iondata(2, is_top_ion=True)], 26, logpath)

    assert lower.photoionization_targetfractions == [[(0, 1.0)]]
    warning = "WARNING: photoionisation target 'no such level' matched no level of the upper ion"
    assert [line for line in lower.comments["phixsdata"] if line.startswith(warning)]
    logtext = logpath.read_text(encoding="utf-8")
    assert "Z=26 Fe I photoionisation targets:" in logtext
    assert warning in logtext

    # an ion with no target names has nothing to resolve, so the log file gets no empty section
    logpath.write_text("", encoding="utf-8")
    resolve_photoion_targetfractions([make_iondata(1, is_top_ion=False), make_iondata(2, is_top_ion=True)], 26, logpath)
    assert not logpath.read_text(encoding="utf-8")


def test_hydrogenic_estimate_gives_one_summary_of_the_levels_with_no_table():
    """The log file gets a warning for each such level, and the comment block gets one count line."""
    from artisatomic.base import IonLog

    levels = pl.DataFrame(
        {
            "levelid": [0, 1, 2],
            "energyabovegsinpercm": [0.0, 1000.0, 1.0e6],
            "g": [2.0, 2.0, 2.0],
            "levelname": ["3s_2Se", "a name with no quantum number", "9s_2Se"],
        }
    )
    flog = IonLog(io.StringIO())
    match_hydrogenic_phixs(11, levels, 5.139, "cmfgen", readhillierdata.get_level_valence_n, phixs_args(), flog)
    summaries = [line for line in flog.comments["phixsdata"] if "got no hydrogenic table" in line]
    expected = (
        "2 of the lowest 3 levels got no hydrogenic table: 1 are at or above the ionisation energy, 1 have a level"
        " name with no principal quantum number, and 0 have an n outside the hydrogenic tables."
    )
    assert summaries == [expected]


def test_phot_file_reader_rejects_a_target_that_a_second_file_or_line_repeats(tmp_path):
    """Each phot file of an ion has one target, and two files of an ion must not name the same target.

    combine_phixs_routes() takes each file as one route to its target. Two routes to one target,
    or a file with two targets, would give cross sections to the wrong target with no message.
    """
    from artisatomic.readhillierdata import PhotFileReader

    def phot_text(targetlines: str) -> str:
        return (
            "\n*****\n  header comment\n12-Oct-2009                             !Date\n"
            "1                                       !Number of energy levels\n"
            "2.0D0                                   !Screened nuclear charge\n"
            f"{targetlines}"
            "Megabarns                               !Cross-section unit\n"
            "False                                   !Split J levels\n"
            "A                                       !Configuration name\n"
            "20                                      !Type of cross-section\n"
            "2                                       !Number of cross-section points\n"
            "1.0 4.0\n2.0 1.0\n\n"
        )

    onetarget = "5s2_5p6_1Se                             !Final state in ion\n"
    othertarget = "5s2_5p5_2Po                             !Final state in ion\n"
    for name, targetlines in (("phot_A", onetarget), ("phot_B", onetarget), ("phot_C", onetarget + othertarget)):
        (tmp_path / name).write_text(phot_text(targetlines), encoding="utf-8")

    # the second file of the ion names the target of the first file
    reader = PhotFileReader(56, 2, 2, [911.0], {"A": 0}, {"A": 0}, io.StringIO())
    reader.read_file(0, tmp_path / "phot_A", "phot_A")
    assert reader.phixstargets == ["5s2_5p6_1Se", ""]
    with pytest.raises(ValueError, match="Multiple phixs files for the same target configuration 5s2_5p6_1Se"):
        reader.read_file(1, tmp_path / "phot_B", "phot_B")

    # one file with two target lines
    reader = PhotFileReader(56, 2, 1, [911.0], {"A": 0}, {"A": 0}, io.StringIO())
    with pytest.raises(ValueError, match="phot_C has more than one '!Final state in ion' line"):
        reader.read_file(0, tmp_path / "phot_C", "phot_C")


def test_log_detail_gives_one_count_line_for_each_kind():
    """The log file gets each detail line, and the comment block gets the first line of a kind with its count."""
    from artisatomic.base import IonLog
    from artisatomic.base import log_comment
    from artisatomic.base import log_detail

    stream = io.StringIO()
    flog = IonLog(stream)
    log_comment(flog, ("transitiondata",), "source: a data set")
    for upper in ("B", "C", "D"):
        log_detail(
            flog, ("transitiondata",), "swapped transition levels", f"WARNING: Swapped transition levels A -> {upper}"
        )
    log_detail(flog, ("transitiondata",), "discarded upsilon", "Discarded upsilon=0.500 for A -> B")
    log_comment(flog, ("transitiondata",), "Read 4 effective collision strengths")

    assert stream.getvalue().count("Swapped transition levels") == 3
    assert flog.comments["transitiondata"] == [
        "source: a data set",
        "WARNING: Swapped transition levels A -> B (the first of 3 such lines in the log file)",
        # one line of a kind goes into the block as it is
        "Discarded upsilon=0.500 for A -> B",
        "Read 4 effective collision strengths",
    ]

    # a read that runs again starts each count again, so no count line shows the lines of two reads
    flog = IonLog(io.StringIO())
    counts = flog.comment_counts()
    log_detail(flog, ("adata",), "level name with no LS term", "The Hillier level name 'x' has no LS term")
    log_detail(flog, ("adata",), "level name with no LS term", "The Hillier level name 'y' has no LS term")
    flog.drop_comments_after(counts)
    log_detail(flog, ("adata",), "level name with no LS term", "The Hillier level name 'x' has no LS term")
    assert flog.comments["adata"] == ["The Hillier level name 'x' has no LS term"]

    # a plain stream records nothing
    plainstream = io.StringIO()
    log_detail(plainstream, ("adata",), "kind", "a detail line")
    assert plainstream.getvalue() == "a detail line\n"
