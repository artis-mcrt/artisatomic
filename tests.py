#!/usr/bin/env python3
from makeartisatomicfiles import get_term_as_tuple, interpret_configuration, score_config_match, reduce_configuration, interpret_parent_term


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
    assert score_config_match('3d6    (5D ) 6s  e6D ', '3d6(5D)6se6De[9/2]') > score_config_match('3d6    (5D ) 6s  e6D ', '3d6(5D)5s6De[9/2]')

    assert score_config_match('Eqv st (0S ) 0s  a4P', '3d5_4Pe[4]') == 5
    assert score_config_match('3d6    (5D ) 0s  b2F ', '3d7b2Fe') > 90
    assert score_config_match('3d5    (2D ) 4p  v3Po', '3d5(b2D)4p_3Po') == 98


if __name__ == "__main__":
    print('3d7b2Fe', interpret_configuration('3d7b2Fe'))
    print(reduce_configuration('3d6_3P2e'))
