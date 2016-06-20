#!/usr/bin/env python3
import makeartisatomicfiles as maf


def text_interpret_term():
    assert maf.get_term_as_tuple('3d5(6S)4s(7S)4d6De') == (6, 2, 0)


def test_interpret_configuration():
    assert maf.interpret_configuration('3d7(4F)6d_5Pbe') == (['3d7', '(4F)', '6d'], 5, 1, 2, -1)
    assert maf.interpret_configuration('3d6(5D)6d4Ge[9/2]') == (['3d6', '(5D)', '6d'], 4, 4, 0, -1)
    assert maf.interpret_configuration('3d6(3G)4s4p_w5Go[4]') == (['3d6', '(3G)', '4s', '4p'], 5, 4, 1, 4)
    assert maf.interpret_configuration('Eqv st (0S ) 0s  a4P') == ([], 4, 1, 0, 1)


def test_score_config_match():
    assert maf.score_config_match('3d64s  (4P ) 4p  w5Do', '3d6(3P)4s4p_w5Do[4]') == 100
    match1 = maf.score_config_match('3d64s  (6D ) 5g  i5F ', '3d6(5D)4s5g_5Fe[4]')
    assert match1 >= 49
    assert maf.score_config_match('3d64s  (6D ) 5g  (1S) i5F ', '3d6(5D)4s5g_5Fe[4]') == match1
    assert maf.score_config_match('3d6    (5D ) 6s  e6D ', '3d6(5D)6se6De[9/2]') > maf.score_config_match('3d6    (5D ) 6s  e6D ', '3d6(5D)5s6De[9/2]')

    assert maf.score_config_match('Eqv st (0S ) 0s  a4P', '3d5_4Pe[4]') == 5


if __name__ == "__main__":
    pass
