#!/usr/bin/env python3
import numpy as np
# from astropy import constants as const
from astropy import units as u

import artisatomic.readhillierdata as rhd
from artisatomic import (
    get_term_as_tuple, interpret_configuration, interpret_parent_term, reduce_configuration, score_config_match)


def test_reduce_configuration():
    assert reduce_configuration('3d64s  (6D ) 8p  j5Fo') == '3d64s8p_5Fo'
    assert reduce_configuration('3d6_3P2e') == '3d6_3Pe'


def text_interpret_term():
    assert get_term_as_tuple('3d5(6S)4s(7S)4d6De') == (6, 2, 0)
    assert get_term_as_tuple('3d6_3P2e') == (3, 1, 0)


def test_interpret_parent_term():
    assert interpret_parent_term('(3P2)') == (3, 1, 2)
    assert interpret_parent_term('(b2D)') == (2, 2, -1)


def test_interpret_configuration():
    assert interpret_configuration('3d7(4F)6d_5Pbe') == (['3d7', '(4F)', '6d'], 5, 1, 2, -1)
    assert interpret_configuration('3d6(5D)6d4Ge[9/2]') == (['3d6', '(5D)', '6d'], 4, 4, 0, -1)
    assert interpret_configuration('3d6(3G)4s4p_w5Go[4]') == (['3d6', '(3G)', '4s', '4p'], 5, 4, 1, 4)
    assert interpret_configuration('Eqv st (0S ) 0s  a4P') == ([], 4, 1, 0, 1)
    assert interpret_configuration('3d6    (5D ) 4p  z6Do') == (['3d6', '(5D)', '4p'], 6, 2, 1, 1)
    assert interpret_configuration('3d7b2Fe') == (['3d7'], 2, 3, 0, 2)
    assert interpret_configuration('3d6_3P2e') == (['3d6'], 3, 1, 0, -1)


def test_score_config_match():
    assert score_config_match('3d64s  (4P ) 4p  w5Do', '3d6(3P)4s4p_w5Do[4]') == 100
    match1 = score_config_match('3d64s  (6D ) 5g  i5F ', '3d6(5D)4s5g_5Fe[4]')
    assert match1 >= 49
    assert score_config_match('3d64s  (6D ) 5g  (1S) i5F ', '3d6(5D)4s5g_5Fe[4]') == match1
    assert score_config_match('3d6    (5D ) 6s  e6D ',
                              '3d6(5D)6se6De[9/2]') > score_config_match('3d6    (5D ) 6s  e6D ', '3d6(5D)5s6De[9/2]')

    assert score_config_match('Eqv st (0S ) 0s  a4P', '3d5_4Pe[4]') == 5
    assert score_config_match('3d6    (5D ) 0s  b2F ', '3d7b2Fe') > 12
    assert score_config_match('3d5    (2D ) 4p  v3Po', '3d5(b2D)4p_3Po') == 98


def test_hydrogenic_phixs():
    rhd.read_hyd_phixsdata()

    lambda_angstrom = rhd.hc_in_ev_angstrom / (1.0 * u.rydberg.to('eV'))
    expected_n1 = np.array([
        [1.0,          6.30341644],
        [1.1,         4.88284569],
        [1.21,        3.77314939],
        [1.331,       2.90845266],
        [1.4641,      2.23644386],
        [1.61051,     1.71560775],
        [1.771561,    1.31303106],
        [1.9487171,   1.00268611],
        [2.1435888,   0.76405918],
        [2.35794768,  0.58102658]])

    phixstable_nl = rhd.get_hydrogenic_nl_phixstable(lambda_angstrom, 1, 0, 0)
    assert get_maxabs_elratio(expected_n1, phixstable_nl[:10]) <= 0.01

    phixstable_n = rhd.get_hydrogenic_n_phixstable(lambda_angstrom, 1)
    assert get_maxabs_elratio(expected_n1, phixstable_n[:10]) <= 0.01

    lambda_angstrom = rhd.hc_in_ev_angstrom / (5 ** 2 * u.rydberg.to('eV'))
    expected_n5 = np.array([
        [2.50000000e+01,   5.91880525e-02],
        [2.75000000e+01,   4.48991991e-02],
        [3.02500000e+01,   3.40407216e-02],
        [3.32750000e+01,   2.57948374e-02],
        [3.66024999e+01,   1.95370282e-02],
        [4.02627499e+01,   1.47907913e-02],
        [4.42890249e+01,   1.11930170e-02],
        [4.87179274e+01,   8.46718091e-03],
        [5.35897201e+01,   6.40292666e-03],
        [5.89486921e+01,   4.84036618e-03]])
    phixstable_nl = rhd.get_hydrogenic_nl_phixstable(lambda_angstrom, 5, 0, 4)
    assert get_maxabs_elratio(expected_n5, phixstable_nl[:10]) <= 0.01

    phixstable_n = rhd.get_hydrogenic_n_phixstable(lambda_angstrom, 5)
    assert get_maxabs_elratio(expected_n5, phixstable_n[:10]) <= 0.01


def get_maxabs_elratio(a1, a2):
    ratio = np.divide(a1, a2)
    maxabsratio = ratio.max() - 1.0
    maxabsratio = max(maxabsratio, 1 - ratio.min())
    return maxabsratio


if __name__ == "__main__":
    print('3d7b2Fe', interpret_configuration('3d7b2Fe'))
    print(reduce_configuration('3d6_3P2e'))
    test_hydrogenic_phixs()
