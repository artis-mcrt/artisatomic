#!/usr/bin/env python3
# make sure there is no leading or trailing whitespace in the keys
nahar_configuration_replacements = {
    # O I groundstate
    "Eqv st (0S ) 0s  a3P": "2s2_2p4_3Pe",  # O I groundstate
    # O II
    "2s22p2 (3P ) 0s  z4So": "2s2_2p3_4So",  # O II groundstate
    "2s22p2 (1D ) 0s  a4P": "2s_2p4_4Pe",
    "2s22p2 (1D ) 0s  a2S": "2s_2p4_2Se",
    "2s22p2 (1D ) 3d  e2P": "2s_2p4_2Pe",
    # O III
    "2s22p  (2Po) 0s  a3P": "2s2_2p2_3Pe",  # O III groundstate
    # Fe I
    "Eqv st (0S ) 0s  a5D": "3d6_4s2_5De",  # Fe I groundstate
    "3d64s  (6D ) 5s  b5D": "3d6(5D)4s5s_e5De",
    "3d64s  (6D ) 4d  c5D": "3d6(5D)4s4d_f5De",
    "3d64s  (6D ) 4d  a5G": "3d6(5D)4s4d_e5Ge",
    "3d64s  (6D ) 4d  b5P": "3d6(5D)4s4d_e5Pe",
    "3d7    (4F ) 4d  c5P": "3d7(4F)4d_f5Pe",
    "3d64s  (6D ) 5d  d5P": "3d6(5D)4s5d_5Pe",
    "3d64s  (4D ) 4d  e5P": "3d6(5D)4s4d_5Pe",
    "3d64s  (4G ) 4p  t5Fo": "3d6(3D)4s4p_5Fo",
    "3d64s  (6D ) 5s  a7D": "3d6(5D)4s5s_e7De",
    "3d64s  (6D ) 6s  c7D": "3d6(5D)4s6s_g7De",
    "3d7    (4F ) 5s  b5F": "3d7(4F)5s_e5Fe",
    # Fe II
    "3d6    (5D ) 0s  a4P": "3d7_a4Pe",
    "3d6    (5D ) 0s  a4F": "3d7_a4Fe",
    "3d6    (3P ) 0s  b4G": "3d54s2b4Ge",
    "3d6    (3P ) 0s  a6S": "3d54s2a6Se",
    "3d6    (5D ) 0s  b2F": "3d7b2Fe",
    "3d6    (3P ) 0s  c2H": "3d54s22He",
    "3d6    (3P ) 0s  d4F": "3d54s24Fe",
    "3d54s  (5G ) 0s  f2F": "3d54s22Fe",
    "3d6    (5D ) 5p  v6Po": "3d5(4P)4s4p(3P)6Po",
    "3d6    (3P ) 0s  e2G": "3d54s22Ge",
    "3d5    (6S ) 0s  d1S": "3d4(1S)4s2_1Se",
    # Fe III
    "3d5    (6S ) 0s  a5D": "3d6_5De",  # Fe III groundstate
    "Eqv st (0S ) 0s  a4P": "3d5_4Pe",
    "Eqv st (0S ) 0s  a4F": "3d5_4Fe",
    "3d5    (6S ) 0s  a3G": "3d6_3Ge",
    "3d5    (2G ) 4s  c3G": "3d5(2G2)4s_3Ge",
    "3d5    (2G ) 4s  d3G": "3d5(2G1)4s_3Ge",
    "3d5    (6S ) 0s  a3P": "3d6_3P2e",
    "3d5    (6S ) 0s  b3P": "3d6_3P1e",
    "3d5    (6S ) 0s  a3D": "3d6_3De",
    "3d5    (4D ) 4s  b3D": "3d5(4D)4s_3De",
    "3d5    (2D ) 4s  c3D": "3d5(2D3)4s_3De",
    "3d5    (2D ) 4s  d3D": "3d5(2D2)4s_3De",
    # "3d5    (6S ) 0s  a3F": "3d6_3Fe",
    "3d5    (6S ) 0s  a3F": "3d6_3F2e",  # duplicate -- what to do?
    "3d5    (6S ) 0s  b3F": "3d6_3F1e",
    "3d5    (2F ) 4s  c3F": "3d5(2F2)4s_3Fe",
    "3d5    (2F ) 4s  e3F": "3d5(2F1)4s_3Fe",
    "3d5    (6S ) 0s  a3H": "3d6_3He",
    "3d5    (6S ) 0s  a1D": "3d6_1D2e",
    "3d5    (2D ) 4s  b1D": "3d5(2D3)4s_1De",
    "3d5    (6S ) 0s  c1D": "3d6_1De",
    "3d5    (2D ) 4s  d1D": "3d5(2D2)4s_1De",
    "3d5    (6S ) 0s  a1G": "3d6_1G2e",
    "3d5    (6S ) 0s  b1G": "3d6_1G1e",
    "3d5    (6S ) 0s  a1I": "3d6_1Ie",
    "3d5    (6S ) 0s  a1S": "3d6_1S2e",
    "3d5    (6S ) 0s  b1S": "3d6_1Se",
    "3d5    (6S ) 0s  a1F": "3d6_1Fe",
    "3d5    (2F ) 4s  b1F": "3d5(2F2)4s_1Fe",
    "3d5    (2F ) 4s  c1F": "3d5(2F1)4s_1Fe",
    "3d5    (2D ) 4p  x3Fo": "3d5(a2D)4p_3Fo",
    "3d5    (2D ) 4p  z1Do": "3d5(a2F)4p_1Do",
    "3d5    (2F ) 4p  y1Do": "3d5(b2F)4p_1Do",
    "3d5    (2D ) 4p  w1Do": "3d5(b2D)4p_1Do",
    "3d5    (2D ) 4p  x3Po": "3d5(a2D)4p_3Po",
    "3d5    (2S ) 4p  w3Po": "3d5(2S)4p_3Po",
    "3d5    (2D ) 4p  v3Po": "3d5(b2D)4p_3Po",
    "3d5    (2D ) 4p  z1Po": "3d5(a2D)4p_1Po",
    "3d5    (2F ) 4p  z1Go": "3d5(a2F)4p_1Go",
    "3d5    (2G ) 4p  y1Go": "3d5(a2G)4p_1Go",
    "3d5    (2F ) 4p  w1Go": "3d5(b2F)4p_1Go",
    # Fe IV
    "Eqv st (0S ) 0s  a6S": "3d5_6Se",  # Fe IV groundstate
    "Eqv st (0S ) 0s  a4G": "3d5_4Ge",
    "3d4    (3P ) 4s  b4P": "3d4(3P2)4s_4Pe",
    "3d4    (3P ) 4s  c4P": "3d4(3P1)4s_4Pe",
    "3d4    (3P ) 4p  y4Do": "3d4(3P2)4p_4Do",
    "3d4    (3F ) 4p  x4Do": "3d4(3F2)4p_4Do",
    "3d4    (3D ) 4p  w4Do": "3d4(3D)4p_4Do",
    "3d4    (3F ) 4p  v4Do": "3d4(3F1)4p_4Do",
    "3d4    (3P ) 4p  u4Do": "3d4(3P1)4p_4Do",
    "3d4    (3F ) 4s  b4F": "3d4(3F2)4s_4Fe",
    "3d4    (3F ) 4s  c4F": "3d4(3F1)4s_4Fe",
    "3d4    (3P ) 4p  y4Po": "3d4(3P2)4p_4Po",
    "3d4    (3P ) 4p  w4Po": "3d4(3P1)4p_4Po",
    "3d4    (3P ) 4p  z4So": "3d4(3P1)4p_4So",
    "3d4    (3P ) 4p  y4So": "3d4(3P)5p_4So",
    "3d4    (3F ) 4p  y4Fo": "3d4(3F2)4p_4Fo",
    "3d4    (3H ) 4p  z4Go": "3d4(3H)4p_4Go",
    "3d4    (3F ) 4p  y4Go": "3d4(3F2)4p_4Go",
    "3d4    (3G ) 4p  x4Go": "3d4(3G)4p_4Go",
    "3d4    (3F ) 4p  w4Go": "3d4(3F1)4p_4Go",
    "3d4    (3F ) 4p  v4Fo": "3d4(3F1)4p_4Fo",
}

# this is necessary if the original Hiller term doesn't match
hillier_name_replacements = {
    "2s2_2p3(4So)5z_5Z": "2s22p3 (4So) 5g  z5Go",
    "2s2_2p3(4So)5z_3Z": "2s22p3 (4So) 5g  z3Go",
    # '2s2_2p3(4So)6z_3Z': '2s22p3 (4So) 6g  y3Go',
    # '2s2_2p3(4So)6z_3Z': '2s22p3 (4So) 6h  a3H',
    # '2s2_2p3(4So)6z_5Z': '2s22p3 (4So) 6g  y5Go',
    # '2s2_2p3(4So)6z_5Z': '2s22p3 (4So) 6h  a5H',
    # '2s2_2p3(4So)8z_3Z': '',
}
