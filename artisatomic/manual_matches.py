# Hillier level names whose LS term the parser cannot read. get_term_as_tuple() is applied to
# the replacement instead, which is what gives these levels their parity, and so decides which
# of their transitions are marked forbidden.
hillier_name_replacements = {
    "2s2_2p3(4So)5z_5Z": "2s22p3 (4So) 5g  z5Go",
    "2s2_2p3(4So)5z_3Z": "2s22p3 (4So) 5g  z3Go",
    # '2s2_2p3(4So)6z_3Z': '2s22p3 (4So) 6g  y3Go',
    # '2s2_2p3(4So)6z_3Z': '2s22p3 (4So) 6h  a3H',
    # '2s2_2p3(4So)6z_5Z': '2s22p3 (4So) 6g  y5Go',
    # '2s2_2p3(4So)6z_5Z': '2s22p3 (4So) 6h  a5H',
    # '2s2_2p3(4So)8z_3Z': '',
}
