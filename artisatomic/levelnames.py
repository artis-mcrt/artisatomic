"""Parse level names: split a configuration into orbitals and a term, and derive the parity."""

from collections.abc import Iterator

alphabets = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "
reversedalphabets = "zyxwvutsrqponmlkjihgfedcbaZYXWVUTSRQPONMLKJIHGFEDCBA "
lchars = "SPDFGHIKLMNOPQRSTUVWXYZ"


def _split_orbitals(instr: str) -> list[str]:
    """Split the configuration part of a level name into occupied orbitals and parent terms.

    Parent terms keep their parentheses, so callers can tell them from occupied orbitals. Any
    term has already been taken off the end by the caller, so all of this is configuration.
    """
    max_n = 20  # maximum possible principle quantum number n

    def is_two_digit_n(strn: str) -> bool:
        """Test whether two digits are a principal quantum number, not one digit of something else.

        n is written 10 to 19 when it takes two digits, so a leading zero rules it out: the '0' of
        '3d104s' belongs to the occupation number of the 3d shell, giving 4s and not 04s.
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

    return electron_config


def interpret_configuration(
    instr_orig: str, warn: bool = True, hasterm: bool = True
) -> tuple[list[str], int, int, int, int]:
    """Split a level name into its orbitals and term.

    Returns (orbitals, 2S+1, L, parity, index in symmetry), where orbitals is the configuration
    split into occupied orbitals and parent terms (kept in parentheses), and the index in
    symmetry comes from the seniority letter if the name has one. Term components that cannot be
    read come back as -1.

    warn=False silences the malformed-name message for callers that expect names this cannot
    split. CMFGEN's merged levels ('1___', '8SNG') are such names by design, and there are
    enough of them to bury a real warning.

    hasterm=False reads the whole string as a configuration, for sources whose level name carries
    no term. An ADAS adf04 file keeps 2S+1 and L in their own columns, so its '5s2' is an orbital
    rather than a term to strip, and its '3S2 3P6 3D5 4P1' would otherwise lose the 4P1 that way.
    All the term components come back as -1, there being no term to read.
    """
    instr = instr_orig.split("[", maxsplit=1)[0]  # remove trailing bracketed J value

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
        # to catch e.g., '3d6(5D)6d4Ge[9/2]' occupation piece 6d, not index d
        # and 3d7b2Fe is at index b, (keep it from conflicting into the orbital occupation)
        indexinsymmetry = reversedalphabets.index(instr[-1]) + 1 if term_parity == 1 else alphabets.index(instr[-1]) + 1
        instr = instr[:-1]

    return _split_orbitals(instr), term_twosplusone, term_l, term_parity, indexinsymmetry


def _iter_occupied_orbitals(instr, warn: bool, hasterm: bool = True) -> Iterator[tuple[int, int, bool]]:
    """Walk the occupied orbitals of a configuration, yielding (l, number of electrons, merged).

    Parent terms in parentheses are not occupied orbitals and are skipped. An orbital must
    satisfy l <= n - 1; CMFGEN's merged high-l levels ('2s2_13w_2W', '2s2_2p3(4So)5z_5Z') fail
    this, because the letter is a merge marker spanning several l rather than one orbital. Those
    are yielded with merged=True so callers can tell the two cases apart.

    One token can hold more than one orbital, because interpret_configuration() keeps a
    digit-letter-letter run together: '4sp(3P)_7Po' gives '4sp', which is 4s and 4p sharing the
    one principal quantum number. Each letter is therefore taken in turn, with the digits that
    follow it as its occupation and 1 where it has none.
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
            # Only a name with a term is read case-sensitively. CMFGEN writes orbitals in lower
            # case and keeps upper case for terms, so an upper-case letter there is a term symbol
            # or a parent term, not an orbital: reading '8SNG' (He I's merged singlets) as an 8s
            # orbital, or the mangled '3H' of '3d4(3H)s44p_x3Io' as a merge marker, would be
            # wrong. A bare configuration has no term to confuse it with, and adf04 writes its
            # orbitals in upper case ('3S2 3P6 3D5 4P1'), so there the letter is unambiguous.
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
            # Tokens such as the 'sp' of '3d8(2H)sp_2Go' have none, and calling those merged
            # would discard the parity their name states outright.
            yield l, nelec, nend > 0 and l >= principalquantumnumber

        if not foundorbital and warn:
            # don't fail silently: a skipped orbital means the parity (and hence the forbidden
            # flags of every transition involving this level) could come out wrong
            print(f"WARNING: could not read an orbital from '{orbitalstr}' in '{instr}', skipping it for the parity")


def has_merged_orbital(instr, hasterm: bool = True) -> bool:
    """Whether the configuration contains a merge marker, i.e. an orbital with l >= n.

    CMFGEN writes its merged high-l levels this way ('2s2_13w_2W', '10z_2Z'): the letter stands
    for several l of both parities at once, so the level has no parity rather than an unreadable
    one, and no suffix on the name can supply it.
    """
    return any(merged for _l, _nelec, merged in _iter_occupied_orbitals(instr, warn=False, hasterm=hasterm))


def get_config_parity(instr, warn: bool = False, hasterm: bool = True) -> int | None:
    """Parity of a configuration (0 even, 1 odd), or None when it does not determine one.

    None means no orbital in the name had a readable l to sum, as for CMFGEN's merged n-levels
    '1___' and '13___' (g = 2n^2, every l of that n) and He I's merged '8SNG' and '8TRP'. That is
    different from the 0 those would otherwise get from an empty sum. Merge markers are left out
    of the sum as in get_parity_from_config(), so check has_merged_orbital() as well when a level
    with no definite parity has to be recognised.

    Unreadable names are the expected case for the callers that need the None, so this is quiet
    by default; pass warn=True to get the per-orbital diagnostics.
    """
    lsum = 0
    readable = False
    for l, nelec, merged in _iter_occupied_orbitals(instr, warn=warn, hasterm=hasterm):
        if not merged:
            lsum += l * nelec
            readable = True

    return lsum % 2 if readable else None


def get_parity_from_config(instr) -> int:
    """Parity of a level from its configuration: the sum of l over the occupied orbitals, mod 2.

    Returns 0 for even and 1 for odd. Merge markers have no definite l, so they are left out of
    the sum rather than making the whole result unknown. Prefer get_config_parity() where a level
    with no definite parity has to be told apart from an even one.
    """
    return sum(l * nelec for l, nelec, merged in _iter_occupied_orbitals(instr, warn=True) if not merged) % 2
