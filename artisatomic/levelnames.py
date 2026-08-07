"""Parse level names: split a configuration into orbitals and a term, and derive the parity."""

alphabets = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "
reversedalphabets = "zyxwvutsrqponmlkjihgfedcbaZYXWVUTSRQPONMLKJIHGFEDCBA "
lchars = "SPDFGHIKLMNOPQRSTUVWXYZ"


def interpret_configuration(instr_orig: str) -> tuple[list[str], int, int, int, int]:
    """Split a level name into its orbitals and term.

    Returns (orbitals, 2S+1, L, parity, index in symmetry), where orbitals is the configuration
    split into occupied orbitals and parent terms (kept in parentheses), and the index in
    symmetry comes from the seniority letter if the name has one. Term components that cannot be
    read come back as -1.
    """
    max_n = 20  # maximum possible principle quantum number n
    instr = instr_orig
    instr = instr.split("[")[0]  # remove trailing bracketed J value

    if instr[-1] in lchars:
        term_parity = 0  # even
    else:
        term_parity = [0, 1][(instr[-1] == "o")]
        if all(char not in lchars for char in instr):
            # This will be an incorrectly formatted QUB file with no term
            print("Warning: Check QUB file formatting")
        else:
            # Preserve previous behaviour
            instr = instr[:-1]

    term_twosplusone = -1
    term_l = -1
    indexinsymmetry = -1

    while instr:
        if instr[-1] in lchars:
            term_l = lchars.index(instr[-1])
            instr = instr[:-1]
            break
        if not str.isdigit(instr[-1]):
            term_parity += (
                2  # this accounts for things like '3d7(4F)6d_5Pbe' in the Hillier levels. Shouldn't match these
            )
        instr = instr[:-1]
        if all(char not in lchars for char in instr):
            print("Warning: Check QUB file formatting")
            break

    if instr and str.isdigit(instr[-1]):
        term_twosplusone = int(instr[-1])
        instr = instr[:-1]

    if not instr:
        pass
    elif instr[-1] == "_":
        instr = instr[:-1]
    elif instr[-1] in alphabets and (
        (len(instr) < 2 or not str.isdigit(instr[-2])) or (len(instr) < 3 or instr[-3] in lchars.lower())
    ):
        # to catch e.g., '3d6(5D)6d4Ge[9/2]' occupation piece 6d, not index d
        # and 3d7b2Fe is at index b, (keep it from conflicting into the orbital occupation)
        indexinsymmetry = reversedalphabets.index(instr[-1]) + 1 if term_parity == 1 else alphabets.index(instr[-1]) + 1
        instr = instr[:-1]

    def is_two_digit_n(strn: str) -> bool:
        """Test whether two digits are a principal quantum number, not one digit of something else.

        n is written 10 to 19 when it takes two digits, so a leading zero rules it out: the '0' of
        '3d104s' belongs to the occupation number of the 3d shell, giving 4s and not 04s.
        """
        return strn.isdigit() and 10 <= int(strn) < max_n

    electron_config: list[str] = []
    if not instr.startswith("Eqv st"):
        while instr:
            if instr[-1].upper() in lchars:
                # Orbital with no occupation number, e.g. the '10d' of '3d6(5D)10d_5Pe'. A
                # digit-letter-letter run keeps its digit, so '4sp(3P)_7Po[2]' gives '4sp'.
                startpos = (
                    -3
                    if len(instr) >= 3
                    and (is_two_digit_n(instr[-3:-1]) or (instr[-3].isdigit() and not instr[-2].isdigit()))
                    else -2
                )

                electron_config.insert(0, instr[startpos:])
                instr = instr[:startpos]
            elif instr[-1] == ")":
                left_bracket_pos = instr.rfind("(")
                str_parent_term = instr[left_bracket_pos:].replace(" ", "")
                electron_config.insert(0, str_parent_term)
                instr = instr[:left_bracket_pos]
            elif str.isdigit(instr[-1]):  # the number of electrons in an orbital
                if len(instr) >= 2 and instr[-2].upper() in lchars:
                    # Single-digit occupation. A two-digit n survives ('10d1') only where the
                    # digits cannot belong to a preceding orbital: '3d14s2' is ambiguous
                    # (3d1 4s2 or 3d 14s2), and there the occupation-1 reading wins.
                    two_digit_n = (
                        len(instr) >= 4
                        and is_two_digit_n(instr[-4:-2])
                        and (len(instr) == 4 or not (instr[-5].isdigit() or instr[-5].upper() in lchars))
                    )
                    startpos = -4 if two_digit_n else -3
                    electron_config.insert(0, instr[startpos:])
                    instr = instr[:startpos]
                elif len(instr) >= 3 and str.isdigit(instr[-2]) and instr[-3].upper() in lchars:
                    # Two-digit occupation, e.g. the closed shells '3d10' and '4f14'. This is
                    # unambiguous: trailing digits after the orbital letter are the occupation.
                    startpos = -4 if len(instr) >= 4 and str.isdigit(instr[-4]) else -3
                    electron_config.insert(0, instr[startpos:])
                    instr = instr[:startpos]
                else:
                    instr = instr[:-1]
            elif instr[-1] in {"_", " "}:
                instr = instr[:-1]
            else:
                instr = instr[:-1]

    return electron_config, term_twosplusone, term_l, term_parity, indexinsymmetry


def get_parity_from_config(instr) -> int:
    """Parity of a level from its configuration: the sum of l over the occupied orbitals, mod 2.

    Returns 0 for even and 1 for odd. Parent terms in parentheses are not occupied orbitals and
    are skipped, as are merge markers (see the l >= n check below).
    """
    configsplit = interpret_configuration(instr)[0]
    lchars_lower = lchars.lower()
    lsum = 0
    for orbitalstr in configsplit:
        if orbitalstr.startswith("("):
            continue  # a parent term such as '(5D)', not an occupied orbital
        # the orbital letter is the first non-digit, allowing a two-digit principal quantum number
        lpos = next((pos for pos, char in enumerate(orbitalstr) if char in lchars_lower), None)
        if lpos is None:
            # don't fail silently: a skipped orbital means the parity (and hence the forbidden
            # flags of every transition involving this level) could come out wrong
            print(f"WARNING: could not read an orbital from '{orbitalstr}' in '{instr}', skipping it for the parity")
            continue
        l = lchars_lower.index(orbitalstr[lpos])
        nelec = int(orbitalstr[lpos + 1 :]) if len(orbitalstr[lpos + 1 :]) > 0 else 1

        # An orbital must satisfy l <= n - 1. CMFGEN's merged high-l levels ('2s2_13w_2W',
        # '2s2_2p3(4So)5z_5Z') fail this: the letter is a merge marker spanning several l of both
        # parities, so it has no definite parity and only the real orbitals count.
        principalquantumnumber = int(orbitalstr[:lpos]) if orbitalstr[:lpos].isdigit() else 0
        if l >= principalquantumnumber:
            continue

        lsum += l * nelec

    return lsum % 2
