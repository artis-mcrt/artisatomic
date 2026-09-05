"""Parse level names: split a configuration into orbitals and a term, and derive the parity."""

import string
from collections.abc import Iterator

alphabets = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "
reversedalphabets = "zyxwvutsrqponmlkjihgfedcbaZYXWVUTSRQPONMLKJIHGFEDCBA "
# the orbital letters in order of l. The sequence skips J, and it skips P and S at l = 12 and
# l = 14, because those letters are l = 1 and l = 0.
lchars = "SPDFGHIKLMNOQRTUVWXYZ"

# CMFGEN names a level that lumps the high-l orbitals of one n with one of these letters, whatever
# the n. '3s2_18w_2W' has g = 648 = 2 x 18^2, the whole n = 18 shell, and '3s2_7z_2Z' lumps the
# l >= 6 orbitals. No CMFGEN file names a single orbital of l = 17 (w) or l = 20 (z). Such a
# level spans both parities, so it has none.
merged_orbital_letters = frozenset("wz")


def parse_orbital_n(orbital: str) -> int | None:
    """Principal quantum number of one orbital token such as "6s", "4f10" or "6h2".

    The token is the principal quantum number, one orbital letter of any l, and an optional
    electron count. Returns None for any other form, so a caller can skip the level rather
    than guess.
    """
    part = orbital.rstrip(string.digits)
    if len(part) < 2 or part[-1] not in lchars.lower():
        return None
    n_text = part[:-1]
    return int(n_text) if n_text.isdigit() else None


def _split_orbitals(instr: str) -> list[str]:
    """Split the configuration part of a level name into occupied orbitals and parent terms.

    Parent terms keep their parentheses, so callers can tell them from occupied orbitals. The
    caller has already removed any term from the end, so all of this is configuration.
    """
    max_n = 20  # maximum possible principal quantum number n

    def is_two_digit_n(strn: str) -> bool:
        """Test whether two digits are a principal quantum number, not one digit of another number.

        A two-digit n is 10 to 19, so a leading zero rules it out. The '0' of '3d104s' belongs to
        the occupation number of the 3d shell, which gives 4s and not 04s.
        """
        return strn.isdigit() and 10 <= int(strn) < max_n

    electron_config: list[str] = []
    if instr.startswith("Eqv st"):
        return electron_config

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
                # digits cannot belong to a preceding orbital. '3d14s2' is ambiguous
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
        else:
            instr = instr[:-1]

    return electron_config


def interpret_configuration(
    instr_orig: str, warn: bool = True, hasterm: bool = True
) -> tuple[list[str], int, int, int, int]:
    """Split a level name into its orbitals and term.

    Returns (orbitals, 2S+1, L, parity, index in symmetry). orbitals is the configuration split
    into occupied orbitals and parent terms (kept in parentheses). The index in symmetry comes
    from the seniority letter if the name has one. A term component that the function cannot read
    comes back as -1.

    warn=False silences the malformed-name message for callers that expect names this cannot
    split. CMFGEN's merged levels ('1___', '8SNG') are such names by design, and there are
    enough of them to bury a real warning.

    hasterm=False reads the whole string as a configuration, for sources whose level name carries
    no term. An ADAS adf04 file keeps 2S+1 and L in their own columns, so its '5s2' is an orbital
    and not a term to strip. Its '3S2 3P6 3D5 4P1' would otherwise lose the 4P1 in that way. All
    the term components come back as -1, because there is no term to read.
    """
    instr = instr_orig.split("[", maxsplit=1)[0]  # remove trailing bracketed J value

    if not instr:
        # a name with nothing before its J bracket has no orbital and no term to read
        return [], -1, -1, -1, -1

    if not hasterm:
        return _split_orbitals(instr), -1, -1, -1, -1

    if instr[-1] in lchars:
        term_parity = 0  # even
    else:
        term_parity = [0, 1][(instr[-1] == "o")]
        if all(char not in lchars for char in instr):
            # This will be an incorrectly formatted QUB file with no term
            if warn:
                print("Warning: Check QUB file formatting")
        else:
            # drop the parity letter, so the term parse below sees the term only
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
                2  # this accounts for names such as '3d7(4F)6d_5Pbe' in the Hillier levels. These must not match
            )
        instr = instr[:-1]
        if all(char not in lchars for char in instr):
            if warn:
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
        # This catches, for example, the occupation piece 6d of '3d6(5D)6d4Ge[9/2]', which is not an index d.
        # '3d7b2Fe' has the index b. The test keeps the index separate from the orbital occupation.
        indexinsymmetry = reversedalphabets.index(instr[-1]) + 1 if term_parity == 1 else alphabets.index(instr[-1]) + 1
        instr = instr[:-1]

    return _split_orbitals(instr), term_twosplusone, term_l, term_parity, indexinsymmetry


def _iter_occupied_orbitals(instr, warn: bool, hasterm: bool = True) -> Iterator[tuple[int, int, bool]]:
    """Walk the occupied orbitals of a configuration and yield (l, number of electrons, merged).

    Parent terms in parentheses are not occupied orbitals, and the walk skips them. An orbital
    must satisfy l <= n - 1. CMFGEN's merged high-l levels ('2s2_13w_2W', '2s2_2p3(4So)5z_5Z')
    fail this test, because the letter is a merge marker that spans several l and not one
    orbital. A w or a z is such a marker at any n. '3s2_18w_2W' lumps the whole n = 18 shell, and
    l = 17 < 18 would pass the test. The walk yields those with merged=True, so callers can tell
    the two cases apart.

    One token can hold more than one orbital, because interpret_configuration() keeps a
    digit-letter-letter run together. '4sp(3P)_7Po' gives '4sp', which is 4s and 4p with one
    shared principal quantum number. The walk therefore takes each letter in turn, with the
    digits that follow it as its occupation and 1 where it has none.
    """
    lchars_lower = lchars.lower()
    for orbitalstr in interpret_configuration(instr, warn=warn, hasterm=hasterm)[0]:
        if orbitalstr.startswith("("):
            continue  # a parent term such as '(5D)', not an occupied orbital

        # the leading digits are the principal quantum number, which may be two digits long
        nend = 0
        while nend < len(orbitalstr) and orbitalstr[nend].isdigit():
            nend += 1
        principalquantumnumber = int(orbitalstr[:nend]) if nend else 0

        pos = nend
        foundorbital = False
        while pos < len(orbitalstr):
            # Only a name with a term gets a case-sensitive read. CMFGEN writes orbitals in lower
            # case and keeps upper case for terms. An upper-case letter there is a term symbol or
            # a parent term, not an orbital. To read '8SNG' (He I's merged singlets) as an 8s
            # orbital would be wrong, as would the mangled '3H' of '3d4(3H)s44p_x3Io' as a merge
            # marker. A bare configuration has no term to confuse it with, and adf04 writes its
            # orbitals in upper case ('3S2 3P6 3D5 4P1'). There the letter is unambiguous.
            orbitalchar = orbitalstr[pos] if hasterm else orbitalstr[pos].lower()
            if orbitalchar not in lchars_lower:
                pos += 1
                continue
            l = lchars_lower.index(orbitalchar)
            pos += 1
            nstart = pos
            while pos < len(orbitalstr) and orbitalstr[pos].isdigit():
                pos += 1
            nelec = int(orbitalstr[nstart:pos]) if pos > nstart else 1
            foundorbital = True
            # l >= n identifies a merge marker, but only where the token actually carried an n.
            # Tokens such as the 'sp' of '3d8(2H)sp_2Go' have none, and to call those merged
            # would discard the parity that their name states.
            merged = orbitalchar in merged_orbital_letters or (nend > 0 and l >= principalquantumnumber)
            yield l, nelec, merged

        if not foundorbital and warn:
            # Do not fail silently: a skipped orbital means the parity (and therefore the forbidden
            # flags of every transition of this level) could come out wrong.
            print(f"WARNING: could not read an orbital from '{orbitalstr}' in '{instr}', skipping it for the parity")


def has_merged_orbital(instr, hasterm: bool = True) -> bool:
    """Whether the configuration contains a merge marker, i.e. an orbital with l >= n.

    CMFGEN writes its merged high-l levels in this way ('2s2_13w_2W', '10z_2Z'). The letter
    stands for several l of both parities at once. The level therefore has no parity, not an
    unreadable one, and no suffix on the name can supply it.
    """
    return any(merged for _l, _nelec, merged in _iter_occupied_orbitals(instr, warn=False, hasterm=hasterm))


def get_config_parity(instr, warn: bool = False, hasterm: bool = True) -> int | None:
    """Parity of a configuration (0 even, 1 odd), or None when it does not determine one.

    None means that no orbital in the name had a readable l to sum. Examples are CMFGEN's merged
    n-levels '1___' and '13___' (g = 2n^2, every l of that n) and He I's merged '8SNG' and
    '8TRP'. An empty sum is 0, which is a real parity and the wrong answer for those. This
    function therefore reports None and leaves the decision to the caller. A merge marker stays
    out of the sum and does not make the whole result None. Check has_merged_orbital() as well to
    recognise a level that has no definite parity at all.

    Unreadable names are the expected case for the callers that need the None, so this function
    is quiet by default. Pass warn=True to get the per-orbital diagnostics.
    """
    lsum = 0
    readable = False
    for l, nelec, merged in _iter_occupied_orbitals(instr, warn=warn, hasterm=hasterm):
        if not merged:
            lsum += l * nelec
            readable = True

    return lsum % 2 if readable else None
