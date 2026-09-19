"""Parse level names: split a configuration into orbitals and a term, and derive the parity."""

import re
import string
from collections.abc import Iterator
from itertools import starmap

alphabets = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "
reversedalphabets = "zyxwvutsrqponmlkjihgfedcbaZYXWVUTSRQPONMLKJIHGFEDCBA "
# the orbital letters in order of l. The sequence skips J, and it skips P and S at l = 12 and
# l = 14, because those letters are l = 1 and l = 0.
lchars = "SPDFGHIKLMNOQRTUVWXYZ"
lchars_lower = lchars.lower()


def subshell_capacity(l: int) -> int:
    """Return the largest number of electrons of a subshell, 2(2l+1)."""
    return 4 * l + 2


# CMFGEN names a level that merges the high-l orbitals of one n with one of these letters, whatever
# the n. '3s2_18w_2W' has g = 648 = 2 x 18^2, the whole n = 18 shell, and '3s2_7z_2Z' merges the
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
    if len(part) < 2 or part[-1] not in lchars_lower:
        return None
    n_text = part[:-1]
    return int(n_text) if n_text.isdigit() else None


def _split_orbitals(instr: str) -> list[str]:
    """Split the configuration part of a level name into occupied orbitals and parent terms.

    Parent terms keep their parentheses, so callers can tell them from occupied orbitals. The
    caller has already removed any term from the end, so all of this is configuration.
    """
    max_n = 20  # is_two_digit_n() accepts an n below this value only

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
            # a name with no term letter. The ADAS reader passes hasterm=False and never reaches this.
            if warn:
                print(f"WARNING: the level name '{instr_orig}' has no term letter")
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
            # A letter between the term and the parity letter, as in CMFGEN's '3d7(4F)6d_5Pbe', adds
            # 2 to the parity. Such a level then matches no level of parity 0 or 1.
            term_parity += 2
        instr = instr[:-1]
        if all(char not in lchars for char in instr):
            if warn:
                print(f"WARNING: the level name '{instr_orig}' has no term letter")
            break

    if instr and str.isdigit(instr[-1]):
        term_twosplusone = int(instr[-1])
        instr = instr[:-1]

    # '2s8z1Z' and '1s5z3Zo' end in a merge marker whose letter is the term letter, not in an
    # index. An index letter never equals the term letter of its own name in CMFGEN.
    ends_in_merge_marker = (
        bool(instr) and instr[-1] in merged_orbital_letters and term_l == lchars.index(instr[-1].upper())
    )
    if not instr:
        pass
    elif instr[-1] == "_":
        instr = instr[:-1]
    elif instr[-1] in alphabets and not ends_in_merge_marker and _last_letter_is_index(instr):
        # This catches, for example, the occupation piece 6d of '3d6(5D)6d4Ge[9/2]', which is not an index d.
        # '3d7b2Fe' has the index b. The test keeps the index separate from the orbital occupation.
        indexinsymmetry = reversedalphabets.index(instr[-1]) + 1 if term_parity == 1 else alphabets.index(instr[-1]) + 1
        instr = instr[:-1]

    return _split_orbitals(instr), term_twosplusone, term_l, term_parity, indexinsymmetry


def _last_letter_is_index(instr: str) -> bool:
    """Whether the last letter of the configuration is an index in the symmetry, not an orbital."""
    return len(instr) < 3 or not str.isdigit(instr[-2]) or instr[-3] in lchars_lower


def _iter_occupied_orbitals(instr, warn: bool, hasterm: bool = True) -> Iterator[tuple[int, int, bool]]:
    """Walk the occupied orbitals of a configuration and yield (l, number of electrons, merged).

    Parent terms in parentheses are not occupied orbitals, and the walk skips them. An orbital
    must satisfy l <= n - 1. CMFGEN's merged high-l levels ('2s2_13w_2W', '2s2_2p3(4So)5z_5Z')
    fail this test, because the letter is a merge marker that spans several l and not one
    orbital. A w or a z is such a marker at any n. '3s2_18w_2W' merges the whole n = 18 shell, and
    l = 17 < 18 would pass the test. The walk yields those with merged=True, so callers can tell
    the two cases apart.

    One token can hold more than one orbital, because _split_orbitals() keeps a
    digit-letter-letter run together. '4sp(3P)_7Po' gives '4sp', which is 4s and 4p with one
    shared principal quantum number. The walk therefore takes each letter in turn, with the
    digits that follow it as its occupation and 1 where it has none.
    """
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
            print(f"WARNING: could not read an orbital from '{orbitalstr}' in '{instr}'. The parity ignores it.")


def has_merged_orbital(instr, hasterm: bool = True) -> bool:
    """Whether the configuration contains a merge marker: a w or z orbital, or an orbital with l >= n.

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


def split_count_and_n(previousorbital: str, digits: str, orbital: str) -> int | None:
    """Read the principal quantum number n from the digits between two orbital letters of a label.

    previousorbital is the orbital letter before the digits, or "" when the digits start the
    label. The digits then hold n only. Otherwise they start with the electron count of that
    orbital. The function takes a count-plus-n reading only when it is physical: the count fits
    the previous orbital, and the valence orbital has l < n. So "5s111s1" (adf04, 5s1 11s1) gives
    11 and not 1. For two digits, the function falls back to a two-digit n. For three digits, it
    returns None when no reading is physical. A run of four or more digits gives None.

    Every caller matches the run of digits with a pattern, so the run holds digits only.
    """
    if not previousorbital:
        return int(digits)
    l_previous = lchars_lower.find(previousorbital)
    l_valence = lchars_lower.find(orbital)

    def physical(count: str, n: str) -> bool:
        return (
            n[0] != "0"
            and (l_previous < 0 or int(count) <= subshell_capacity(l_previous))
            and (l_valence < 0 or l_valence < int(n))
        )

    if len(digits) == 1:
        return int(digits)
    if len(digits) == 2:
        # a count and a one-digit n ("s25p"), else a two-digit n ("s10d", "s11p")
        return int(digits[1:]) if physical(digits[:1], digits[1:]) else int(digits)
    if len(digits) == 3:
        # a two-digit count and a one-digit n ("f125d"), else a count and a two-digit n ("s210d")
        if physical(digits[:2], digits[2:]):
            return int(digits[2:])
        return int(digits[1:]) if physical(digits[:1], digits[1:]) else None
    return None


# The order of the Eissner shell characters. The shell character is case-sensitive, and the lower
# case letters come after "Z".
#
# The ADAS adf04 specification (appxa-04) gives 9=4d, 0=4f, A=5s. The adf04 files from
# AUTOSTRUCTURE give 9=4d, A=4f, B=5s and have no "0". In the OPEN-ADAS file for He-like C
# (cophps][he/dw/ls][c4.dat), each level is 1s nl, so its total L is the l of the outer shell. The
# shells "A" to "F" of that file have L = 3, 0, 1, 2, 3, 4, thus 4f, 5s, 5p, 5d, 5f, 5g. No file
# gives evidence for the shells after "F".
eissner_shell_orders = {
    "AUTOSTRUCTURE": string.digits[1:] + string.ascii_uppercase + string.ascii_lowercase,
    "specification": string.digits[1:] + "0" + string.ascii_uppercase + string.ascii_lowercase,
}
default_eissner_order = "AUTOSTRUCTURE"
# each shell is (n, l), in the order 1s, 2s, 2p, 3s, ...
eissner_shell_by_char: dict[str, dict[str, tuple[int, int]]] = {
    order: dict(zip(chars, ((n, l) for n in range(1, len(chars)) for l in range(n)), strict=False))
    for order, chars in eissner_shell_orders.items()
}

# One Eissner triple is the occupation code (50 + the occupation) and the shell character. A
# string of this form is an Eissner configuration or a defective one. It is not standard notation,
# because standard notation has an orbital letter as its second character.
eissner_like_config_regex = re.compile(r"(?:[56][0-9][0-9A-Za-z])+")
# an f shell holds the largest number of electrons of the shells in use
eissner_largest_occupation = 14


def _with_full_first_triple(config: str) -> str:
    """Return the configuration with a full first triple.

    The specification lets the first shell give the occupation q in place of 50 + q, as in "21522".
    The shell character of the short form must not be an orbital letter in upper case or lower
    case. A label such as "2P", and the compact "3D54P" (3d5 4p), are standard notation.
    """
    if len(config) % 3 == 2 and config[0] in "123456789" and config[1].upper() not in lchars:
        return "5" + config
    return config


def _eissner_shells(config: str, order: str = default_eissner_order) -> list[tuple[int, int, int]] | None:
    """Return the (n, l, occupation) of each shell of an Eissner configuration, or None for a different string."""
    config = _with_full_first_triple(config)
    if eissner_like_config_regex.fullmatch(config) is None:
        return None
    shells = []
    for start in range(0, len(config), 3):
        shell = eissner_shell_by_char[order].get(config[start + 2])
        occupation = int(config[start : start + 2]) - 50
        # A label such as "591" (1s9) is not a configuration.
        if shell is None or not 1 <= occupation <= min(subshell_capacity(shell[1]), eissner_largest_occupation):
            return None
        shells.append((*shell, occupation))
    return shells


def _max_total_l(l: int, occupation: int) -> int:
    """Return the largest total L of a shell. Each m_l value from l down holds two electrons."""
    electrons = min(occupation, subshell_capacity(l) - occupation)
    return sum(l - index // 2 for index in range(electrons))


def eissner_total_l_is_possible(config: str, total_l: int, order: str = default_eissner_order) -> bool:
    """Return False if the shells of an Eissner configuration cannot give the total L of the level.

    A False result for many levels shows a wrong order of the shell characters. The test is exact
    for a maximum of two open shells that each have one electron or one hole. For other levels, it
    is only an upper limit.
    """
    shells = _eissner_shells(config, order)
    if shells is None:
        return True
    open_shells = [(l, occupation) for _n, l, occupation in shells if occupation < subshell_capacity(l)]
    if total_l > sum(starmap(_max_total_l, open_shells)):
        return False
    if len(open_shells) <= 2 and all(occupation in {1, subshell_capacity(l) - 1} for l, occupation in open_shells):
        # Two such shells give each L from |l_a - l_b| to l_a + l_b, and the test above made the upper limit.
        l_a, l_b = [*(l for l, _occupation in open_shells), 0, 0][:2]
        return abs(l_a - l_b) <= total_l
    return True


def is_eissner_config(config: str, order: str = default_eissner_order) -> bool:
    """Return True if the string is an Eissner configuration.

    The string is a sequence of Eissner triples, and the first triple can be short ("21522"). The
    bare "5s2" is standard notation.
    """
    return _eissner_shells(config, order) is not None


def looks_like_eissner_config(config: str) -> bool:
    """Return True if the string has the form of an Eissner configuration, valid or defective."""
    return eissner_like_config_regex.fullmatch(_with_full_first_triple(config)) is not None


# One word of the standard form of the specification: n, the orbital letter, the occupation q.
# n and q use the order 1 to 9, then a=10, b=11, ...
standard_word_regex = re.compile(rf"(?<!\S)([1-9a-z])([{lchars_lower}])([1-9a-z])(?!\S)")


def _expand_standard_word(word: re.Match[str]) -> str:
    n_char, orbital, q_char = word.groups()
    n, l, q = int(n_char, 36), lchars_lower.index(orbital), int(q_char, 36)
    # A word with no digit is a label. A subshell has n > l and holds 2(2l+1) electrons at most,
    # so "4fo" (a term with its parity) and "4ff" are labels too.
    if not (n_char.isdigit() or q_char.isdigit()) or n <= l or q > subshell_capacity(l):
        return word[0]
    return f"{n}{orbital}{q}"


def expand_standard_config(config: str) -> str:
    """Write n and q as decimal numbers in each "nlq" word of the lower-case configuration.

    The word "3da" becomes "3d10". A word in a different form stays as it is, and the whitespace
    between the words stays as it is. The word "4fe" is the 4f14 of the specification. The
    function cannot know if a file uses that word as the label of an even 4F term.
    """
    return standard_word_regex.sub(_expand_standard_word, config)


def convert_eissner_to_standard(eissner_config: str, order: str = default_eissner_order) -> str:
    """Convert an electron configuration from Eissner notation to standard notation.

    The configuration "521522563524565" becomes "1s22s22p63s23p6".

    The function follows a Fortran routine from Leo Mulholland. Appendix A of the ADAS manual
    (https://open.adas.ac.uk/man/appxa-04.pdf, pages 5 to 6) describes the notation. See also
    Eissner, W. (1998), Computer Physics Communications, 114, 295-341, page 323,
    doi:10.1016/S0010-4655(98)00082-4.
    """
    shells = _eissner_shells(eissner_config, order)
    if shells is None:
        msg = f"Not an Eissner configuration: {eissner_config!r}"
        raise ValueError(msg)
    return "".join(f"{n}{lchars_lower[l]}{occupation}" for n, l, occupation in shells)
