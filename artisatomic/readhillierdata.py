#!/usr/bin/env python3
import math
import os
import sys
from collections import defaultdict
from collections import namedtuple
from pathlib import Path
from typing import Any
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from astropy import constants as const
from astropy import units as u

import artisatomic
from artisatomic.manual_matches import hillier_name_replacements

# need to also include collision strengths from e.g., o2col.dat

hillier_rowformat_a = (
    "levelname g energyabovegsinpercm freqtentothe15hz thresholdenergyev lambdaangstrom hillierlevelid arad gam2 gam4"
)
hillier_rowformat_b = (
    "levelname g energyabovegsinpercm freqtentothe15hz thresholdenergyev lambdaangstrom hillierlevelid arad c4 c6"
)
hillier_rowformat_c = "levelname g energyabovegsinpercm freqtentothe15hz lambdaangstrom hillierlevelid"
hillier_rowformat_d = (
    "levelname g energyabovegsinpercm thresholdenergyev freqtentothe15hz lambdaangstrom hillierlevelid"
)

hillier_rowformat = {
    "17-Jun-2014": (
        "levelname g energyabovegsinpercm freqtentothe15hz thresholdenergyev lambdaangstrom hillierlevelid arad c4 c6"
    ),
    "17-Oct-2000": (
        "levelname g energyabovegsinpercm freqtentothe15hz thresholdenergyev lambdaangstrom hillierlevelid arad gam2"
        " gam4"
    ),
    "NOT_SPECIFIED": "levelname g energyabovegsinpercm freqtentothe15hz lambdaangstrom hillierlevelid",
}

# keys are (atomic number, ion stage)
ion_files = namedtuple(
    "ion_files", ["folder", "levelstransitionsfilename", "energylevelrowformat", "photfilenames", "coldatafilename"]
)

ions_data = {
    # H
    (1, 1): ion_files("5dec96", "hi_osc.dat", hillier_rowformat_c, ["hiphot.dat"], "hicol.dat"),
    (1, 2): ion_files("", "", hillier_rowformat_c, [""], ""),
    # He
    # (2, 1): ion_files('11may07', 'heioscdat_a7.dat_old', hillier_rowformat_a, ['heiphot_a7.dat'], 'heicol.dat'),
    # (2, 2): ion_files('5dec96', 'he2_osc.dat', hillier_rowformat_c, ['he2phot.dat'], 'he2col.dat'),
    # C
    (6, 1): ion_files("12dec04", "ci_split_osc", hillier_rowformat_a, ["phot_smooth_50"], "cicol.dat"),
    (6, 2): ion_files("30oct12", "c2osc_rev.dat", hillier_rowformat_b, ["phot_sm_3000.dat"], "c2col.dat"),
    (6, 3): ion_files(
        "23dec04",
        "ciiiosc_st_split_big.dat",
        hillier_rowformat_a,
        ["ciiiphot_sm_a_500.dat", "ciiiphot_sm_b_500.dat"],
        "ciiicol.dat",
    ),
    (6, 4): ion_files("30oct12", "civosc_a12_split.dat", hillier_rowformat_b, ["civphot_a12.dat"], "civcol.dat"),
    # N
    # (7, 1): ion_files('12sep12', 'ni_osc', hillier_rowformat_b, ['niphot_a.dat', 'niphot_b.dat', 'niphot_c.dat', 'niphot_d.dat'], 'ni_col'),
    # (7, 2): ion_files('23jan06', 'fin_osc', hillier_rowformat_b, ['phot_sm_3000'], 'n2col.dat'),
    # (7, 3): ion_files('24mar07', 'niiiosc_rev.dat', hillier_rowformat_a, ['phot_sm_0_A.dat', 'phot_sm_0_B.dat'], 'niiicol.dat'),
    # O
    (8, 1): ion_files("20sep11", "oi_osc_mchf", hillier_rowformat_b, ["phot_nosm_A", "phot_nosm_B"], "oi_col"),
    (8, 2): ion_files("23mar05", "o2osc_fin.dat", hillier_rowformat_a, ["phot_sm_3000.dat"], "o2col.dat"),
    (8, 3): ion_files("15mar08", "oiiiosc", hillier_rowformat_a, ["phot_sm_0"], "col_data_oiii_butler_2012.dat"),
    (8, 4): ion_files("19nov07", "fin_osc", hillier_rowformat_a, ["phot_sm_50_A", "phot_sm_50_B"], "col_oiv"),
    # F
    # (9, 2): ion_files('tst', 'fin_osc', hillier_rowformat_a, ['phot_data_a', 'phot_data_b', 'phot_data_c'], ''),
    # (9, 3): ion_files('tst', 'fin_osc', hillier_rowformat_a, ['phot_data_a', 'phot_data_b', 'phot_data_c', 'phot_data_d'], ''),
    # Ne
    (10, 1): ion_files("9sep11", "fin_osc", hillier_rowformat_b, ["fin_phot"], "col_guess"),
    (10, 2): ion_files("19nov07", "fin_osc", hillier_rowformat_a, ["phot_nosm"], "col_neii"),
    (10, 3): ion_files("19nov07", "fin_osc", hillier_rowformat_a, ["phot_nosm"], "col_neiii"),
    # (10, 4): ion_files('1dec99', 'fin_osc.dat', hillier_rowformat_c, ['phot_sm_3000.dat'], 'col_data.dat'),
    # Na
    # (11, 1): ion_files('5aug97', 'nai_osc_split.dat', hillier_rowformat_c, ['nai_phot_a.dat'], 'col_guess.dat'),
    # (11, 2): ion_files('15feb01', 'na2_osc.dat', hillier_rowformat_a, ['phot_data.dat'], 'col_guess.dat'),
    # (11, 3): ion_files('15feb01', 'naiiiosc_rev.dat', hillier_rowformat_a, ['phot_sm_3000.dat'], 'col_guess.dat'),
    # (11, 4): ion_files('15feb01', 'naivosc_rev.dat', hillier_rowformat_a, ['phot_sm_3000.dat'], 'col_data.dat'),
    # Mg
    (12, 1): ion_files("5aug97", "mgi_osc_split.dat", hillier_rowformat_c, ["mgi_phot_a.dat"], "mgicol.dat"),
    (12, 2): ion_files("30oct12", "mg2_osc_split.dat", hillier_rowformat_b, ["mg2_phot_a.dat"], "mg2col.dat"),
    # (12, 3): ion_files('20jun01', 'mgiiiosc_rev.dat', hillier_rowformat_a, ['phot_sm_3000.dat'], 'col_guess.dat'),
    # (12, 4): ion_files('20jun01', 'mgivosc_rev.dat', hillier_rowformat_a, ['phot_sm_3000.dat'], 'col_guess.dat'),
    # Al
    # (13, 1): ion_files('29jul10', 'fin_osc', hillier_rowformat_a, ['phot_smooth_0'], 'col_data'),
    # (13, 2): ion_files('5aug97', 'al2_osc_split.dat', hillier_rowformat_c, ['al2_phot_a.dat'], 'al2col.dat'),
    # (13, 3): ion_files('30oct12', 'aliii_osc_split.dat', hillier_rowformat_b, ['aliii_phot_a.dat'], 'aliii_col_data.dat'),
    # (13, 4): ion_files('23oct02', 'fin_osc', hillier_rowformat_a, ['phot_sm_3000.dat'], 'col_guess'),
    # Si
    (14, 1): ion_files("23nov11", "SiI_OSC", hillier_rowformat_b, ["SiI_PHOT_DATA"], "col_data"),
    (14, 2): ion_files("30oct12", "si2_osc_nahar", hillier_rowformat_b, ["phot_op.dat"], "si2_col"),
    (14, 3): ion_files("5dec96b", "osc_op_split_rev.dat_1jun12", hillier_rowformat_a, ["phot_op.dat"], "col_data"),
    (14, 4): ion_files("30oct12", "osc_op_split.dat", hillier_rowformat_b, ["phot_op.dat"], "col_data.dat"),
    # P (IV and V are the only ions in CMFGEN)
    # (15, 4): ion_files('15feb01', 'pivosc_rev.dat', hillier_rowformat_a, ['phot_data_a.dat', 'phot_data_b.dat'], 'col_guess.dat'),
    # (15, 5): ion_files('15feb01', 'pvosc_rev.dat', hillier_rowformat_a, ['phot_data.dat'], 'col_guess.dat'),
    # S
    (16, 1): ion_files("24nov11", "SI_OSC", hillier_rowformat_b, ["SI_PHOT_DATA"], "col_data"),
    (16, 2): ion_files("30oct12", "s2_osc", hillier_rowformat_b, ["phot_sm_3000"], "s2_col"),
    (16, 3): ion_files("30oct12", "siiiosc_fin", hillier_rowformat_a, ["phot_nosm"], "col_siii"),
    (16, 4): ion_files("19nov07", "sivosc_fin", hillier_rowformat_a, ["phot_nosm"], "col_siv"),
    # Cl (only ions IV to VII)
    # (17, 4): ion_files('15feb01', 'clivosc_fin.dat', hillier_rowformat_a, ['phot_data.dat'], 'col_data.dat'),
    # Ar
    # (18, 1): ion_files('9sep11', 'fin_osc', hillier_rowformat_b, ['phot_nosm'], 'col_guess'),
    (18, 1): ion_files("9sep11", "fin_osc_ljs", hillier_rowformat_b, ["phot_nosm"], "col_guess"),
    (18, 2): ion_files("9sep11", "fin_osc", hillier_rowformat_b, ["phot_nosm"], "col_data"),
    (18, 3): ion_files("19nov07", "fin_osc", hillier_rowformat_a, ["phot_nosm"], "col_ariii"),
    (18, 4): ion_files("1dec99", "fin_osc.dat", hillier_rowformat_c, ["phot_sm_3000.dat"], "col_data.dat"),
    # K
    # (19, 1): ion_files('4mar12', 'fin_osc', hillier_rowformat_b, ['phot_ki'], 'COL_DATA'),
    # (19, 2): ion_files('4mar12', 'fin_osc', hillier_rowformat_b, ['phot_k2'], 'COL_DATA'),
    # Ca
    (20, 1): ion_files("5aug97", "cai_osc_split.dat", hillier_rowformat_c, ["cai_phot_a.dat"], "caicol.dat"),
    (20, 2): ion_files("30oct12", "ca2_osc_split.dat", hillier_rowformat_c, ["ca2_phot_a.dat"], "ca2col.dat"),
    (20, 3): ion_files("10apr99", "osc_op_sp.dat", hillier_rowformat_c, ["phot_smooth.dat"], "col_guess.dat"),
    (20, 4): ion_files("10apr99", "osc_op_sp.dat", hillier_rowformat_c, ["phot_smooth.dat"], "col_guess.dat"),
    # Sc (only II and III are in CMFGEN)
    # (21, 2): ion_files('01jul13', 'fin_osc', hillier_rowformat_a, ['phot_nosm'], 'col_data'),
    # (21, 3): ion_files('3dec12', 'fin_osc', hillier_rowformat_a, ['phot_nosm'], ''),
    # Ti (only II and III are in CMFGEN, IV has dummy files with a single level)
    # (22, 2): ion_files('18oct00', 'tkii_osc.dat', hillier_rowformat_a, ['phot_data.dat'], 'col_guess.dat'),
    # (22, 3): ion_files('18oct00', 'tkiii_osc.dat', hillier_rowformat_a, ['phot_data.dat'], 'col_guess.dat'),
    # (22, 4): ion_files('18oct00', 'tkiv_osc.dat', hillier_rowformat_a, ['phot_data.dat'], 'col_guess.dat'),
    # V (only V I is in CMFGEN and it has a single level)
    # (23, 1): ion_files('27may10', 'vi_osc', hillier_rowformat_a, ['vi_phot.dat'], 'col_guess.dat'),
    # Cr
    # (24, 1): ion_files('10aug12', 'cri_osc.dat', hillier_rowformat_b, ['phot_data.dat'], 'col_guess.dat'),
    # (24, 2): ion_files('15aug12', 'crii_osc.dat', hillier_rowformat_b, ['phot_data.dat'], 'col_data.dat'),
    # (24, 3): ion_files('18oct00', 'criii_osc.dat', hillier_rowformat_a, ['phot_data.dat'], 'col_guess.dat'),
    # (24, 4): ion_files('18oct00', 'criv_osc.dat', hillier_rowformat_a, ['phot_data.dat'], 'col_guess.dat'),
    # (24, 5): ion_files('18oct00', 'crv_osc.dat', hillier_rowformat_a, ['phot_data.dat'], 'col_guess.dat'),
    # Mn (Mn I is not in CMFGEN)
    # (25, 2): ion_files('18oct00', 'mnii_osc.dat', hillier_rowformat_a, ['phot_data.dat'], 'col_guess.dat'),
    # (25, 3): ion_files('18oct00', 'mniii_osc.dat', hillier_rowformat_a, ['phot_data.dat'], 'col_guess.dat'),
    # (25, 4): ion_files('18oct00', 'mniv_osc.dat', hillier_rowformat_a, ['phot_data.dat'], 'col_guess.dat'),
    # (25, 5): ion_files('18oct00', 'mnv_osc.dat', hillier_rowformat_a, ['phot_data.dat'], 'col_guess.dat'),
    # Fe
    (26, 1): ion_files("07sep16", "fei_osc", hillier_rowformat_b, ["phot_smooth_3000"], "col_data"),
    # (26, 1): ion_files('07sep16', 'fei_osc', hillier_rowformat_b, ['phot_smooth_3000'], 'col_data_ljs'),
    (26, 2): ion_files("10sep16", "fe2_osc", hillier_rowformat_b, ["phot_op.dat"], "fe2_col.dat"),
    (26, 3): ion_files("30oct12", "FeIII_OSC", hillier_rowformat_b, ["phot_sm_3000.dat"], "col_data.dat"),
    (26, 4): ion_files("18oct00", "feiv_osc_rev2.dat", hillier_rowformat_a, ["phot_sm_3000.dat"], "col_data.dat"),
    (26, 5): ion_files("18oct00", "fev_osc.dat", hillier_rowformat_a, ["phot_sm_3000.dat"], "col_guess.dat"),
    # (26, 6): ion_files('18oct00', 'fevi_osc.dat', hillier_rowformat_a, ['phot_sm_3000.dat'], 'col_data.dat'),
    # (26, 7): ion_files('18oct00', 'fevii_osc.dat', hillier_rowformat_a, ['phot_sm_3000.dat'], 'col_guess.dat'),
    # (26, 8): ion_files('8may97', 'feviii_osc_kb_rk.dat', hillier_rowformat_d, ['phot_sm_3000_rev.dat'], 'col_guess.dat'),
    # Co
    (27, 2): ion_files("15nov11", "fin_osc_bound", hillier_rowformat_a, ["phot_nosm"], "Co2_COL_DATA"),
    (27, 3): ion_files("30oct12", "coiii_osc.dat", hillier_rowformat_b, ["phot_nosm"], "col_data.dat"),
    (27, 4): ion_files("4jan12", "coiv_osc.dat", hillier_rowformat_b, ["phot_data"], "col_data.dat"),
    # (27, 5): ion_files('18oct00', 'cov_osc.dat', hillier_rowformat_a, ['phot_data.dat'], 'col_guess.dat'),
    # (27, 6): ion_files('18oct00', 'covi_osc.dat', hillier_rowformat_a, ['phot_data.dat'], 'col_guess.dat'),
    # (27, 7): ion_files('18oct00', 'covii_osc.dat', hillier_rowformat_a, ['phot_data.dat'], 'col_guess.dat'),
    # Ni
    (28, 2): ion_files("30oct12", "nkii_osc.dat", hillier_rowformat_a, ["phot_data"], "col_data_bautista"),
    # (28, 2): ion_files('30oct12', 'nkii_osc.dat', hillier_rowformat_a, ['phot_data_crude'], 'col_data_bautista'),
    (28, 3): ion_files("27aug12", "nkiii_osc.dat", hillier_rowformat_b, ["phot_data.dat"], "col_data.dat"),
    (28, 4): ion_files("18oct00", "nkiv_osc.dat", hillier_rowformat_a, ["phot_data.dat"], "col_guess.dat"),
    (28, 5): ion_files("18oct00", "nkv_osc.dat", hillier_rowformat_a, ["phot_data.dat"], "col_guess.dat"),
    # (28, 6): ion_files('18oct00', 'nkvi_osc.dat', hillier_rowformat_a, ['phot_data.dat'], 'col_guess.dat'),
    # (28, 7): ion_files('18oct00', 'nkvii_osc.dat', hillier_rowformat_a, ['phot_data.dat'], 'col_guess.dat'),
    # Cu, Zn and above are not in CMGFEN?
    # Ba
    # (56, 2): ion_files('', 'fin_osc', hillier_rowformat_a, ['phot_nosm'], 'col_data'),
}

elsymboltohilliercode = {
    "H": "HYD",
    "He": "HE",
    "C": "CARB",
    "N": "NIT",
    "O": "OXY",
    "F": "FLU",
    "Ne": "NEON",
    "Na": "NA",
    "Mg": "MG",
    "Al": "AL",
    "Si": "SIL",
    "P": "PHOS",
    "S": "SUL",
    "Cl": "CHL",
    "Ar": "ARG",
    "K": "POT",
    "Ca": "CA",
    "Sc": "SCAN",
    "Ti": "TIT",
    "V": "VAN",
    "Cr": "CHRO",
    "Mn": "MAN",
    "Fe": "FE",
    "Co": "COB",
    "Ni": "NICK",
    "Ba": "BAR",
}

ryd_to_ev = u.rydberg.to("eV")
hc_in_ev_cm = (const.h * const.c).to("eV cm").value
hc_in_ev_angstrom = (const.h * const.c).to("eV angstrom").value
h_in_ev_seconds = const.h.to("eV s").value
lchars = "SPDFGHIKLMNOPQRSTUVWXYZ"
PYDIR = Path(__file__).parent.absolute()
atomicdata = pd.read_csv(os.path.join(PYDIR, "atomic_properties.txt"), delim_whitespace=True, comment="#")
elsymbols = ["n", *list(atomicdata["symbol"].values)]

# hilliercodetoelsymbol = {v : k for (k,v) in elsymboltohilliercode.items()}
# hilliercodetoatomic_number = {k : elsymbols.index(v) for (k,v) in hilliercodetoelsymbol.items()}

atomic_number_to_hillier_code = {elsymbols.index(k): v for (k, v) in elsymboltohilliercode.items()}

vy95_phixsfitrow = namedtuple("vy95_phixsfitrow", ["n", "l", "E_th_eV", "E_0", "sigma_0", "y_a", "P", "y_w"])

# keys are (n, l), values are energy in Rydberg or cross_section in Megabarns
hyd_phixs_energygrid_ryd: dict[tuple[int, int], float] = {}
hyd_phixs: dict[tuple[int, int], float] = {}

# keys are n quantum number
hyd_gaunt_energygrid_ryd: dict[int, float] = {}
hyd_gaunt_factor: dict[int, float] = {}


def hillier_ion_folder(atomic_number, ion_stage):
    return str(
        (
            artisatomic.PYDIR
            / ".."
            / "atomic-data-hillier"
            / "atomic"
            / atomic_number_to_hillier_code[atomic_number]
            / artisatomic.roman_numerals[ion_stage]
        ).resolve()
    )


def read_levels_and_transitions(
    atomic_number: int, ion_stage: int, flog
) -> tuple[float, list, list, defaultdict[str, int], defaultdict[tuple[int, int, int], list[str]]]:
    hillier_energy_levels: list = [None]
    hillier_levelnamesnoJ_matching_term: defaultdict[tuple[int, int, int], list[str]] = defaultdict(list)
    transition_count_of_level_name: defaultdict[str, int] = defaultdict(int)
    hillier_ionization_energy_ev = 0.0
    transitions: list = []

    if atomic_number == 1 and ion_stage == 2:
        ionization_energy_ev = 0.0
        qub_energy_level_row = namedtuple(  # type: ignore
            "energylevel", "levelname qub_id twosplusone l j energyabovegsinpercm g parity"
        )

        hillier_energy_levels.append(qub_energy_level_row("I", 1, 0, 0, 0, 0.0, 10, 0))
        return (
            hillier_ionization_energy_ev,
            hillier_energy_levels,
            transitions,
            transition_count_of_level_name,
            hillier_levelnamesnoJ_matching_term,
        )

    filename = os.path.join(
        hillier_ion_folder(atomic_number, ion_stage),
        ions_data[(atomic_number, ion_stage)].folder,
        ions_data[(atomic_number, ion_stage)].levelstransitionsfilename,
    )
    artisatomic.log_and_print(flog, "Reading " + filename)
    hillier_transition_row = namedtuple(  # type: ignore
        "transition", "namefrom nameto f A lambdaangstrom i j hilliertransitionid lowerlevel upperlevel coll_str"
    )

    prev_line = ""
    with open(filename) as fhillierosc:
        expected_energy_levels = -1
        expected_transitions = -1
        row_format_energy_level = None
        format_date = "NOT_SPECIFIED"
        for line in fhillierosc:
            row = line.split()
            if line.startswith("**************************") and prev_line:
                headerline = prev_line
                headerline = headerline.replace("ID", "hillierlevelid")
                headerline = headerline.replace("E(cm^-1)", "energyabovegsinpercm")
                headerline = headerline.replace("10^15 Hz", "freqtentothe15hz")
                headerline = headerline.replace("eV", "thresholdenergyev")
                headerline = headerline.replace("Lam(A)", "lambdaangstrom")
                headerline = headerline.replace("ARAD", "arad")
                row_format_energy_level = "levelname " + " ".join(headerline.lower().split())
                print("File contains columns:")
                print(f"  {row_format_energy_level}")
            elif line.rstrip().endswith("!Number of energy levels"):
                expected_energy_levels = int(row[0])
                artisatomic.log_and_print(flog, f"File specifies {expected_energy_levels:d} levels")
            elif line.rstrip().endswith("!Number of transitions"):
                expected_transitions = int(row[0])
                artisatomic.log_and_print(flog, f"File specifies {expected_transitions:d} transitions")
            elif len(row) == 3 and row[1] == "!Format" and row[2] == "date":
                format_date = row[0]
                print(f"Format date: {format_date}")

            if expected_energy_levels >= 0 and not row:
                break
            prev_line = line.strip()

        if not row_format_energy_level:
            assert format_date == "NOT_SPECIFIED"
            row_format_energy_level = ions_data[(atomic_number, ion_stage)].energylevelrowformat

        print("Manually specified columns:")
        print(f"  {ions_data[(atomic_number, ion_stage)].energylevelrowformat}")
        assert row_format_energy_level == ions_data[(atomic_number, ion_stage)].energylevelrowformat
        # assert(row_format_energy_level == hillier_rowformat[format_date])

        hillier_energy_level_row = namedtuple(  # type: ignore
            "energylevel",
            row_format_energy_level + " corestateid twosplusone l parity indexinsymmetry naharconfiguration matchscore",
        )

        for line in fhillierosc:
            row = line.split()
            # check for right number of columns and that are all numbers except first column
            if len(row) == len(row_format_energy_level.split()) and all(map(artisatomic.isfloat, row[1:])):
                hillier_energy_level = hillier_energy_level_row(*row, 0, -1, -1, -1, -1, "", -1)  # type: ignore

                hillierlevelid = int(hillier_energy_level.hillierlevelid.lstrip("-"))  # type: ignore
                levelname = hillier_energy_level.levelname  # type: ignore
                if levelname not in hillier_name_replacements:
                    (twosplusone, l, parity) = artisatomic.get_term_as_tuple(levelname)
                else:
                    (twosplusone, l, parity) = artisatomic.get_term_as_tuple(hillier_name_replacements[levelname])

                hillier_energy_level = hillier_energy_level._replace(  # type: ignore
                    hillierlevelid=hillierlevelid,
                    energyabovegsinpercm=float(hillier_energy_level.energyabovegsinpercm.replace("D", "E")),  # type: ignore
                    g=float(hillier_energy_level.g),  # type: ignore
                    twosplusone=twosplusone,
                    l=l,
                    parity=parity,
                )

                hillier_energy_levels.append(hillier_energy_level)

                if twosplusone == -1 and atomic_number > 1:
                    # -1 indicates that the term could not be interpreted
                    if parity == -1:
                        artisatomic.log_and_print(flog, f"Can't find LS term in Hillier level name '{levelname}'")
                    # else:
                    # artisatomic.log_and_print(flog, "Can't find LS term in Hillier level name '{0:}' (parity is {1:})".format(levelname, parity))
                else:
                    levelnamenoJ = levelname.split("[")[0]
                    if levelnamenoJ not in hillier_levelnamesnoJ_matching_term[(twosplusone, l, parity)]:
                        hillier_levelnamesnoJ_matching_term[(twosplusone, l, parity)].append(levelnamenoJ)

                # if this is the ground state
                if float(hillier_energy_levels[-1].energyabovegsinpercm) < 1.0:  # type: ignore
                    hillier_ionization_energy_ev = hc_in_ev_angstrom / float(hillier_energy_levels[-1].lambdaangstrom)  # type: ignore

                if hillierlevelid != len(hillier_energy_levels) - 1:
                    artisatomic.log_and_print(
                        flog,
                        (
                            f"Hillier levels mismatch: id {len(hillier_energy_levels) - 1:d} found at entry number"
                            f" {hillierlevelid:d}"
                        ),
                    )
                    sys.exit()

            if line.lstrip().startswith("Oscillator strengths") and len(hillier_energy_levels) > 1:
                break

        artisatomic.log_and_print(flog, f"Read {len(hillier_energy_levels[1:]):d} levels")
        assert len(hillier_energy_levels[1:]) == expected_energy_levels

        # defined_transition_ids = []
        for line in fhillierosc:
            if line.startswith("                        Oscillator strengths"):  # only allow one table
                break
            linesplitdash = line.split("-")
            row = (linesplitdash[0] + " " + "-".join(linesplitdash[1:-1]) + " " + linesplitdash[-1]).split()

            if (len(row) == 8 or (len(row) >= 10 and row[-1] == "|")) and all(map(artisatomic.isfloat, row[2:4])):
                try:
                    lambda_value = float(row[4])
                except ValueError:
                    lambda_value = -1

                hilliertransitionid = int(row[7]) if len(row) == 8 else len(transitions) + 1
                # print(row)
                transition = hillier_transition_row(
                    namefrom=row[0],
                    nameto=row[1],
                    f=float(row[2].replace("D", "E")),
                    A=float(row[3].replace("D", "E")),
                    lambdaangstrom=lambda_value,
                    i=int(row[5].rstrip("-")),
                    j=int(row[6]),
                    hilliertransitionid=hilliertransitionid,
                    lowerlevel=-1,
                    upperlevel=-1,
                    coll_str=-99,
                )

                if (
                    True
                ):  # or int(transition.hilliertransitionid) not in defined_transition_ids: #checking for duplicates massively slows down the code
                    #                    defined_transition_ids.append(int(transition.hilliertransitionid))
                    transitions.append(transition)
                    transition_count_of_level_name[transition.namefrom] += 1
                    transition_count_of_level_name[transition.nameto] += 1

                    if int(transition.hilliertransitionid) != len(transitions):
                        print(
                            f"{filename} WARNING: Transition id {int(transition.hilliertransitionid):d} found at entry"
                            f" number {len(transitions):d}"
                        )
                        # sys.exit()
                else:
                    artisatomic.log_and_print(
                        flog, f"FATAL: multiply-defined Hillier transition: {transition.namefrom} {transition.nameto}"
                    )
                    sys.exit()

    artisatomic.log_and_print(flog, f"Read {len(transitions):d} transitions")
    assert len(transitions) == expected_transitions

    # filter out levels with no transitions
    hillier_energy_levels = [
        hillier_energy_levels[0],
        *[level for level in hillier_energy_levels[1:] if transition_count_of_level_name[level.levelname] > 0],
    ]

    return (
        hillier_ionization_energy_ev,
        hillier_energy_levels,
        transitions,
        transition_count_of_level_name,
        hillier_levelnamesnoJ_matching_term,
    )


# cross section types
phixs_type_labels = {
    0: "Constant (always zero?) [constant]",
    1: "Seaton formula fit [sigma_o, alpha, beta]",
    2: "Hydrogenic split l (z states, n > 11) [n, l_start, l_end]",
    3: "Hydrogenic pure n level (all l, n >= 13) [scale, n]",
    4: "Used for CIV rates from Leobowitz (JQSRT 1972,12,299) (6 numbers)",
    5: "Opacity project fits (from Peach, Sraph, and Seaton (1988) (5 numbers)",
    6: "Hummer fits to the opacity cross-sections for HeI",
    7: "Modified Seaton formula fit (cross section zero until offset edge)",
    8: "Modified hydrogenic split l (cross-section zero until offset edge) [n,l_start,l_end,nu_o]",
    9: "Verner & Yakolev 1995 ground state fits (multiple shells)",
    20: "Opacity Project: smoothed [number of data points]",
    21: "Opacity Project: scaled, smoothed [number of data poits]",
    22: "energy is in units of threshold, cross section in Megabarns? [number of data points]",
}


def read_phixs_tables(atomic_number, ion_stage, energy_levels, args, flog):
    # this gets partially overwritten anyway
    photoionization_crosssections = np.zeros((len(energy_levels), args.nphixspoints))
    photoionization_thresholds_ev = np.zeros(len(energy_levels))
    photoionization_targetconfig_fractions = [[] for _ in energy_levels]
    # return np.zeros((len(energy_levels), args.nphixspoints)),
    # photoionization_targetfractions  # TODO: replace with real data

    n_eff = ion_stage - 1  # effective nuclear charge (with be replaced with value in file if available)
    photfilenames = ions_data[(atomic_number, ion_stage)].photfilenames
    phixstables = [{} for _ in photfilenames]
    phixstargets = ["" for _ in photfilenames]
    reduced_phixs_dict = {}
    phixs_targetconfigfactors_of_levelname = defaultdict(list)

    j_splitting_on = False  # hopefully this is either on or off for all photoion files asssociated with a given ion

    phixs_type_levels = defaultdict(list)
    unknown_phixs_types = []
    for filenum, photfilename in enumerate(photfilenames):
        if photfilename == "":
            continue
        filename = os.path.join(
            hillier_ion_folder(atomic_number, ion_stage), ions_data[(atomic_number, ion_stage)].folder, photfilename
        )
        artisatomic.log_and_print(flog, "Reading " + filename)
        with open(filename) as fhillierphot:
            lowerlevelid = -1
            lowerlevelname = ""
            # upperlevelname = ''
            numpointsexpected = 0
            crosssectiontype = -1
            fitcoefficients = []

            for line in fhillierphot:
                row = line.split()

                if len(row) >= 2 and " ".join(row[-4:]) == "!Final state in ion":
                    # this is not used because the upper ion's levels are not known at this time
                    targetlevelname = row[0]
                    artisatomic.log_and_print(flog, "Photoionisation target: " + targetlevelname)
                    if "[" in targetlevelname:
                        print("STOP! target level contains a bracket (is J-split?)")
                        sys.exit()
                    if targetlevelname in phixstargets:
                        print("STOP! Multiple phixs files for the same target configuration")
                        sys.exit()
                    phixstargets[filenum] = targetlevelname

                if len(row) >= 2 and " ".join(row[-3:]) == "!Split J levels":
                    if row[0].lower() == "true":
                        j_splitting_on = True
                        artisatomic.log_and_print(flog, "File specifies J-splitting enabled")
                    elif row[0].lower() == "false":
                        if j_splitting_on:
                            print("STOP! J-splitting disabled here, but was previously enabled for this ion")
                            sys.exit()
                        j_splitting_on = False
                    else:
                        print(f'STOP! J-splitting not true or false: "{row[0]}"')
                        sys.exit()

                if (
                    len(row) >= 2
                    and " ".join(row[-2:]) == "!Configuration name"
                    or " ".join(row[-3:]) == "!Configuration name [*]"
                ):
                    lowerlevelname = row[0]
                    if "[" in lowerlevelname:
                        lowerlevelname.split("[")[0]
                    fitcoefficients = []
                    numpointsexpected = 0
                    lowerlevelid = 1
                    # find the first matching level (several matches by differ by J values)
                    for levelid, energy_level in enumerate(energy_levels[1:], 1):
                        this_levelnamenoj = energy_level.levelname.split("[")[0]
                        if this_levelnamenoj == lowerlevelname:
                            lowerlevelid = levelid
                            break
                    if targetlevelname == "":
                        print("ERROR: no upper level name")
                        sys.exit()
                    # print(f"Reading level {lowerlevelid} '{lowerlevelname}'")

                if len(row) >= 2 and " ".join(row[-3:]) == "!Screened nuclear charge":
                    # 'Screened nuclear charge' appears mislabelled in the CMFGEN database
                    # it is really an ionisation stage
                    n_eff = int(float(row[0])) - 1

                if len(row) >= 2 and " ".join(row[1:]) == "!Number of cross-section points":
                    numpointsexpected = int(row[0])
                    pointnumber = 0

                if len(row) >= 2 and " ".join(row[1:]) == "!Cross-section unit" and row[0] != "Megabarns":
                    print(f"Wrong cross-section unit: {row[0]}")
                    sys.exit()

                row_is_all_floats = all(map(artisatomic.isfloat, row))
                if crosssectiontype == 0:
                    if len(row) == 1 and row_is_all_floats and numpointsexpected > 0:
                        fitcoefficients.append(float(row[0].replace("D", "E")))

                        if fitcoefficients[-1] != 0.0:
                            print("ERROR: Cross section type 0 has non-zero number after it")
                            sys.exit()

                        # phixstables[filenum][lowerlevelname] = np.zeros((numpointsexpected, 2))

                elif crosssectiontype == 1:
                    if len(row) == 1 and row_is_all_floats and numpointsexpected > 0:
                        fitcoefficients.append(float(row[0].replace("D", "E")))
                        if len(fitcoefficients) == 3:
                            lambda_angstrom = abs(float(energy_levels[lowerlevelid].lambdaangstrom))
                            phixstables[filenum][lowerlevelname] = get_seaton_phixstable(  # type: ignore
                                lambda_angstrom, *fitcoefficients
                            )
                            numpointsexpected = len(phixstables[filenum][lowerlevelname])
                            # artisatomic.log_and_print(flog, 'Using Seaton formula values for level {0}'.format(lowerlevelname))

                elif crosssectiontype == 2:
                    if len(row) == 1 and row_is_all_floats and numpointsexpected > 0:
                        fitcoefficients.append(int(float(row[0])))
                        if len(fitcoefficients) == 3:
                            n, l_start, l_end = fitcoefficients
                            if l_end > n - 1:
                                artisatomic.log_and_print(
                                    flog, "ERROR: can't have l_end = {} > n - 1 = {}".format(l_end, n - 1)
                                )
                            else:
                                lambda_angstrom = abs(float(energy_levels[lowerlevelid].lambdaangstrom))
                                phixstables[filenum][lowerlevelname] = get_hydrogenic_nl_phixstable(
                                    lambda_angstrom, n, l_start, l_end
                                )
                                numpointsexpected = len(phixstables[filenum][lowerlevelname])
                            # artisatomic.log_and_print(flog, 'Using Hydrogenic split l formula values for level {0}'.format(lowerlevelname))

                elif crosssectiontype == 3:
                    if len(row) == 1 and row_is_all_floats and numpointsexpected > 0:
                        fitcoefficients.append(float(row[0]))
                        if len(fitcoefficients) == 2:
                            scale, n = fitcoefficients
                            n = int(n)
                            lambda_angstrom = abs(float(energy_levels[lowerlevelid].lambdaangstrom))
                            phixstables[filenum][lowerlevelname] = scale * get_hydrogenic_n_phixstable(
                                lambda_angstrom, n
                            )

                            numpointsexpected = len(phixstables[filenum][lowerlevelname])
                            # artisatomic.log_and_print(flog, 'Using Hydrogenic pure n formula values for level {0}'.format(lowerlevelname))
                            # print(lowerlevelname)
                            # print(phixstables[filenum][lowerlevelname][:10])
                            # print(get_hydrogenic_nl_phixstable(lambda_angstrom, n, 0, n - 1)[:10])

                elif crosssectiontype == 5:
                    if len(row) == 1 and row_is_all_floats and numpointsexpected > 0:
                        fitcoefficients.append(float(row[0].replace("D", "E")))
                        if len(fitcoefficients) == 5:
                            lambda_angstrom = abs(float(energy_levels[lowerlevelid].lambdaangstrom))
                            phixstables[filenum][lowerlevelname] = get_opproject_phixstable(
                                lambda_angstrom, *fitcoefficients
                            )
                            numpointsexpected = len(phixstables[filenum][lowerlevelname])

                            # artisatomic.log_and_print(flog, 'Using OP project formula values for level {0}'.format(lowerlevelname))

                elif crosssectiontype == 6:
                    if len(row) == 1 and row_is_all_floats and numpointsexpected > 0:
                        fitcoefficients.append(float(row[0].replace("D", "E")))
                        if len(fitcoefficients) == 8:
                            lambda_angstrom = abs(float(energy_levels[lowerlevelid].lambdaangstrom))
                            phixstables[filenum][lowerlevelname] = get_hummer_phixstable(
                                lambda_angstrom, *fitcoefficients
                            )
                            numpointsexpected = len(phixstables[filenum][lowerlevelname])

                            # print(lowerlevelname, "HUMMER")
                            # print(fitcoefficients)
                            # print(phixstables[lowerlevelname][::5])
                            # print(phixstables[lowerlevelname][-10:])

                            # artisatomic.log_and_print(flog, 'Using Hummer formula values for level {0}'.format(lowerlevelname))

                elif crosssectiontype == 7:
                    if len(row) == 1 and row_is_all_floats and numpointsexpected > 0:
                        fitcoefficients.append(float(row[0].replace("D", "E")))
                        if len(fitcoefficients) == 4:
                            lambda_angstrom = abs(float(energy_levels[lowerlevelid].lambdaangstrom))
                            phixstables[filenum][lowerlevelname] = get_seaton_phixstable(
                                lambda_angstrom, *fitcoefficients
                            )
                            numpointsexpected = len(phixstables[filenum][lowerlevelname])
                            # log_and_print(flog, 'Using modified Seaton formula values for level {0}'.format(lowerlevelname))

                elif crosssectiontype == 8:
                    if len(row) == 1 and row_is_all_floats and numpointsexpected > 0:
                        if len(fitcoefficients) <= 2:
                            # first three params should be integers
                            fitcoefficients.append(int(float(row[0].replace("D", "E"))))
                        else:
                            # fourth param is a float
                            fitcoefficients.append(float(row[0].replace("D", "E")))

                        if len(fitcoefficients) == 4:
                            n, l_start, l_end, nu_o = fitcoefficients
                            if l_end > n - 1:
                                artisatomic.log_and_print(flog, f"ERROR: can't have l_end = {l_end} > n - 1 = {n - 1}")
                            else:
                                lambda_angstrom = abs(float(energy_levels[lowerlevelid].lambdaangstrom))
                                phixstables[filenum][lowerlevelname] = get_hydrogenic_nl_phixstable(
                                    lambda_angstrom, n, l_start, l_end, nu_o=nu_o
                                )
                                # log_and_print(flog, 'Using offset Hydrogenic split l formula values for level {0}'.format(lowerlevelname))
                                numpointsexpected = len(phixstables[filenum][lowerlevelname])

                elif crosssectiontype == 9:
                    if len(row) == 8 and numpointsexpected > 0:
                        fitcoefficients.append(
                            vy95_phixsfitrow(int(row[0]), int(row[1]), *[float(x.replace("D", "E")) for x in row[2:]])
                        )

                        if len(fitcoefficients) * 8 == numpointsexpected:
                            lambda_angstrom = abs(float(energy_levels[lowerlevelid].lambdaangstrom))
                            phixstables[filenum][lowerlevelname] = get_vy95_phixstable(lambda_angstrom, fitcoefficients)
                            numpointsexpected = len(phixstables[filenum][lowerlevelname])
                            # artisatomic.log_and_print(flog, 'Using Verner & Yakolev 1995 formula values for level {0}'.format(lowerlevelname))

                elif crosssectiontype in [20, 21, 22]:  # sampled data points
                    if len(row) == 2 and row_is_all_floats and lowerlevelname != "":
                        if lowerlevelname not in phixstables[filenum]:
                            phixstables[filenum][lowerlevelname] = np.zeros((numpointsexpected, 2))

                        lambda_angstrom = abs(float(energy_levels[lowerlevelid].lambdaangstrom))
                        thresholdenergyryd = hc_in_ev_angstrom / lambda_angstrom / ryd_to_ev
                        enryd = float(row[0].replace("D", "E"))

                        if crosssectiontype in [
                            20,
                            21,
                            22,
                        ]:  # the x value is actually a fraction of the threshold, not an energy
                            if pointnumber == 0 and abs(enryd - 1.0) > 0.5:
                                print(
                                    f"{lowerlevelname} cross section type:{crosssectiontype}, {enryd:.3f} is not near"
                                    f" one? might be energy instead? E_threshold = {thresholdenergyryd:.3f} Ry"
                                )
                            enryd *= thresholdenergyryd
                        # elif pointnumber == 0 and abs(enryd - 1.0) < 0.2:
                        #     print(f'{lowerlevelname} cross section type:{crosssectiontype}, {enryd:.3f} is near one? might be energy instead? E_threshold = {thresholdenergyryd:.3f} Ry')

                        xspoint = enryd, float(row[1].replace("D", "E"))
                        phixstables[filenum][lowerlevelname][pointnumber] = xspoint

                        if pointnumber > 0:
                            curenergy = phixstables[filenum][lowerlevelname][pointnumber][0]
                            prevenergy = phixstables[filenum][lowerlevelname][pointnumber - 1][0]
                            if curenergy == prevenergy:
                                print(
                                    f"WARNING: photoionization table for {lowerlevelname} first column duplicated "
                                    f"energy value of {prevenergy}"
                                )
                            elif curenergy < prevenergy:
                                print(
                                    f"ERROR: photoionization table for {lowerlevelname} first column decreases "
                                    f"with energy {prevenergy} followed by {curenergy}"
                                )
                                print(phixstables[filenum][lowerlevelname])
                                sys.exit()
                        pointnumber += 1

                elif crosssectiontype != -1:
                    if crosssectiontype not in unknown_phixs_types:
                        unknown_phixs_types.append(crosssectiontype)
                    fitcoefficients = []
                    lowerlevelname = ""
                    numpointsexpected = 0

                if len(row) >= 2 and " ".join(row[1:]) == "!Type of cross-section":
                    crosssectiontype = int(row[0])
                    if lowerlevelname not in phixs_type_levels[crosssectiontype]:
                        phixs_type_levels[crosssectiontype].append(lowerlevelname)

                if len(row) == 0:
                    if (
                        lowerlevelname != ""
                        and lowerlevelname in phixstables
                        and targetlevelname in phixstables[lowerlevelname]
                        and numpointsexpected != len(phixstables[filenum][lowerlevelname])
                    ):
                        print(
                            f"photoionization_crosssections mismatch: expecting {numpointsexpected:d} rows but found"
                            f" {len(phixstables[filenum][lowerlevelname]):d}"
                        )
                        print(
                            f"A={atomic_number}, ion_stage={ion_stage}, lowerlevel={lowerlevelname},"
                            f" crosssectiontype={crosssectiontype}"
                        )
                        sys.exit()
                    lowerlevelname = ""
                    crosssectiontype = -1
                    numpointsexpected = 0

        for crosssectiontype in sorted(phixs_type_levels.keys()):
            if crosssectiontype in unknown_phixs_types:
                artisatomic.log_and_print(
                    flog,
                    (
                        f"WARNING {len(phixs_type_levels[crosssectiontype])} levels with UNKNOWN cross-section type"
                        f" {crosssectiontype}: {phixs_type_labels[crosssectiontype]}"
                    ),
                )
            else:
                artisatomic.log_and_print(
                    flog,
                    (
                        f"{len(phixs_type_levels[crosssectiontype])} levels with cross-section type {crosssectiontype}:"
                        f" {phixs_type_labels[crosssectiontype]}"
                    ),
                )

        # testing
        # print('3d8(3F)4s_4Fe')
        # print(phixstables[filenum]['3d8(3F)4s_4Fe'])
        reduced_phixstables_onetarget = artisatomic.reduce_phixs_tables(
            phixstables[filenum], args.optimaltemperature, args.nphixspoints, args.phixsnuincrement
        )
        # print(reduced_phixstables_onetarget['3d8(3F)4s_4Fe'])

        # if atomic_number == 26 and ion_stage == 1:
        if args.plotphixs:
            plot_phixs(
                atomic_number, ion_stage, energy_levels, phixstables[filenum], reduced_phixstables_onetarget, args
            )

        for lowerlevelname, reduced_phixstable in reduced_phixstables_onetarget.items():
            try:
                phixs_at_threshold = reduced_phixstable[np.nonzero(reduced_phixstable)][0]
                phixs_targetconfigfactors_of_levelname[lowerlevelname].append(
                    (phixstargets[filenum], phixs_at_threshold)
                )

                # add the new phixs table, or replace the
                # existing one if this target has a larger threshold cross section
                # if lowerlevelname not in reduced_phixs_dict or \
                #         phixs_at_threshold > reduced_phixs_dict[lowerlevelname][0]:
                #     reduced_phixs_dict[lowerlevelname] = reduced_phixstables_onetarget[lowerlevelname]
                if lowerlevelname not in reduced_phixs_dict:
                    reduced_phixs_dict[lowerlevelname] = reduced_phixstables_onetarget[lowerlevelname]
                else:
                    print(f"ERROR: DUPLICATE CROSS SECTION TABLE FOR {lowerlevelname}")
                    # sys.exit()
            except IndexError:
                print(f"WARNING: No non-zero cross section points for {lowerlevelname}")

    # normalise the target factors and scale the phixs table
    phixs_targetconfigfractions_of_levelname = defaultdict(list)
    for lowerlevelname, reduced_phixstable in reduced_phixs_dict.items():
        target_configfactors_nofilter = phixs_targetconfigfactors_of_levelname[lowerlevelname]
        # the factors are arbitary and need to be normalised into fractions

        # filter out low fraction targets
        factor_sum_nofilter = sum([x[1] for x in target_configfactors_nofilter])

        if factor_sum_nofilter > 0.0:
            # if these are false, it's probably all zeros, so leave it and "send" it to the ground state
            target_configfactors = [x for x in target_configfactors_nofilter if (x[1] / factor_sum_nofilter > 0.01)]

            if len(target_configfactors) == 0:
                print("ERRORHERE", lowerlevelname, target_configfactors_nofilter)
                print(reduced_phixstable)
            max_factor = max([x[1] for x in target_configfactors])
            factor_sum = sum([x[1] for x in target_configfactors])

            for target_config, target_factor in target_configfactors:
                target_fraction = target_factor / factor_sum
                phixs_targetconfigfractions_of_levelname[lowerlevelname].append((target_config, target_fraction))

            # e.g. if the target (non-J-split) with the highest fraction has 50%, the cross sections need to be multiplied by two
            # reduced_phixs_dict[lowerlevelname] = reduced_phixstable / (max_factor / factor_sum)
            reduced_phixs_dict[lowerlevelname] = reduced_phixstable  # / (max_factor / factor_sum)

    # now the non-J-split cross sections are mapped onto J-split levels
    for lowerlevelname_a, phixstable in reduced_phixs_dict.items():
        for levelid, energy_level in enumerate(energy_levels[1:], 1):
            levelname_b = energy_level.levelname if j_splitting_on else energy_level.levelname.split("[")[0]

            if levelname_b == lowerlevelname_a:  # due to J splitting, we may match multiple levels here
                photoionization_crosssections[levelid] = phixstable
                photoionization_targetconfig_fractions[levelid] = phixs_targetconfigfractions_of_levelname[levelname_b]

                # photoionization_thresholds_ev[levelid] = energy_level.thresholdenergyev
                photoionization_thresholds_ev[levelid] = hc_in_ev_angstrom / float(energy_level.lambdaangstrom)

    return photoionization_crosssections, photoionization_targetconfig_fractions, photoionization_thresholds_ev


def get_seaton_phixstable(lambda_angstrom, sigmat, beta, s, nu_o=None):
    energygrid = np.arange(0, 1.0, 0.001)
    phixstable = np.empty((len(energygrid), 2))

    thresholdenergyryd = hc_in_ev_angstrom / lambda_angstrom / ryd_to_ev

    for index, c in enumerate(energygrid):
        energy_div_threshold = 1 + 20 * (c**2)

        if nu_o is None:
            threshold_div_energy = energy_div_threshold**-1
            crosssection = sigmat * (beta + (1 - beta) * threshold_div_energy) * (threshold_div_energy**s)
        else:
            # type 7
            # include Christian Vogl's python adaption of CMFGEN sub_phot_gen.f:
            # Altered 07-Oct-2015 : Bug fix for Type 7 (modified Seaton formula).
            #                       Offset was beeing added to the current frequency instead
            #                       of the ionization edge.

            threshold_energy_ev = hc_in_ev_angstrom / lambda_angstrom
            offset_threshold_div_energy = (energy_div_threshold**-1) * (
                1 + (nu_o * 1e15 * h_in_ev_seconds) / threshold_energy_ev
            )

            crosssection = (
                sigmat * (beta + (1 - beta) * offset_threshold_div_energy) * offset_threshold_div_energy**s
                if offset_threshold_div_energy < 1.0
                else 0.0
            )

        phixstable[index] = energy_div_threshold * thresholdenergyryd, crosssection
    return phixstable


# test: for n = 5, l_start = 4, l_end = 4 (2s2_5g_2Ge level of C II)
# 2.18 eV threshold cross section is near 4.37072813 Mb, great!
def get_hydrogenic_nl_phixstable(lambda_angstrom, n, l_start, l_end, nu_o=None):
    assert l_start >= 0
    assert l_end <= n - 1
    energygrid = hyd_phixs_energygrid_ryd[(n, l_start)]
    phixstable = np.empty((len(energygrid), 2))

    thresholdenergyev = hc_in_ev_angstrom / lambda_angstrom
    thresholdenergyryd = thresholdenergyev / ryd_to_ev

    scale_factor = 1 / thresholdenergyryd / (n**2) / ((l_end - l_start + 1) * (l_end + l_start + 1))
    # scale_factor = 1.0

    for index, energy_ryd in enumerate(energygrid):
        energydivthreshold = energy_ryd / energygrid[0]
        if nu_o is None:
            U = energydivthreshold
        else:
            E_o = nu_o * 1e15 * h_in_ev_seconds
            U = thresholdenergyev * energydivthreshold / (E_o + thresholdenergyev)  # energy / (E_0 + E_threshold)
        if U > 0:
            crosssection = 0.0
            for l in range(l_start, l_end + 1):
                if not np.array_equal(hyd_phixs_energygrid_ryd[(n, l)], energygrid):
                    print("TABLE MISMATCH")
                    sys.exit()
                crosssection += (2 * l + 1) * hyd_phixs[(n, l)][index]
            crosssection = crosssection * scale_factor
        else:
            crosssection = 0.0
        phixstable[index][0] = energydivthreshold * thresholdenergyryd  # / ryd_to_ev
        phixstable[index][1] = crosssection

    return phixstable


# test: hydrogen n = 1: 13.606 eV threshold cross section is near 6.3029 Mb
# test: hydrogen n = 5: 2.72 eV threshold cross section is near 37.0 Mb?? can't find a source for this
# give the same results as get_hydrogenic_nl_phixstable(lambda_angstrom, n, 0, n - 1)
def get_hydrogenic_n_phixstable(lambda_angstrom, n):
    energygrid = hyd_gaunt_energygrid_ryd[n]
    phixstable = np.empty((len(energygrid), 2))

    thresholdenergyev = hc_in_ev_angstrom / lambda_angstrom
    thresholdenergyryd = thresholdenergyev / ryd_to_ev

    scale_factor = 7.91 / thresholdenergyryd / n

    for index, energy_ryd in enumerate(energygrid):
        energydivthreshold = energy_ryd / energygrid[0]

        crosssection = (
            scale_factor * hyd_gaunt_factor[n][index] / energydivthreshold**3 if energydivthreshold > 0 else 0.0
        )

        phixstable[index][0] = energydivthreshold * thresholdenergyryd  # / ryd_to_ev
        phixstable[index][1] = crosssection

    return phixstable


# Peach, Sraph, and Seaton (1988)
def get_opproject_phixstable(lambda_angstrom, a, b, c, d, e):
    energygrid = np.arange(0, 1.0, 0.001)
    phixstable = np.empty((len(energygrid), 2))

    thresholdenergyryd = hc_in_ev_angstrom / lambda_angstrom / ryd_to_ev

    for index, cb in enumerate(energygrid):
        energydivthreshold = 1 + 20 * (cb**2)
        u = energydivthreshold

        x = math.log10(min(u, e))

        crosssection = 10 ** (a + x * (b + x * (c + x * d)))
        if u > e:
            crosssection *= (e / u) ** 2

        phixstable[index] = energydivthreshold * thresholdenergyryd, crosssection

    return phixstable


# only applies to helium
# the threshold cross sections seems ok, but energy dependence could be slightly wrong
# what is the h parameter that is not used??
def get_hummer_phixstable(lambda_angstrom, a, b, c, d, e, f, g, h):
    energygrid = np.arange(0, 1.0, 0.001)
    phixstable = np.empty((len(energygrid), 2))

    thresholdenergyryd = hc_in_ev_angstrom / lambda_angstrom / ryd_to_ev

    for index, c_en in enumerate(energygrid):
        energydivthreshold = 1 + 20 * (c_en**2)

        x = math.log10(energydivthreshold)

        crosssection = 10 ** (((d * x + c) * x + b) * x + a) if x < e else 10 ** (f + g * x)

        phixstable[index] = energydivthreshold * thresholdenergyryd, crosssection

    return phixstable


def get_vy95_phixstable(lambda_angstrom, fitcoefficients):
    energygrid = np.arange(0, 1.0, 0.001)
    phixstable = np.empty((len(energygrid), 2))
    thresholdenergyryd = hc_in_ev_angstrom / lambda_angstrom / ryd_to_ev

    for index, c in enumerate(energygrid):
        energydivthreshold = 1 + 20 * (c**2)

        crosssection = 0.0
        for params in fitcoefficients:
            y = energydivthreshold * params.E_th_eV / params.E_0  # E / E_0
            P = params.P
            Q = 5.5 + params.l - 0.5 * params.P
            y_a = params.y_a
            y_w = params.y_w
            crosssection += params.sigma_0 * ((y - 1) ** 2 + y_w**2) * (y**-Q) * ((1 + math.sqrt(y / y_a)) ** -P)

        phixstable[index] = energydivthreshold * thresholdenergyryd, crosssection
    return phixstable


def read_coldata(atomic_number, ion_stage, energy_levels, flog, args):
    t_scale_factor = 1e4  # Hiller temperatures are given as T_4
    upsilondict = {}
    coldatafilename = ions_data[(atomic_number, ion_stage)].coldatafilename
    if coldatafilename == "":
        artisatomic.log_and_print(flog, "No collisional data file specified")
        return upsilondict

    found_nonjsplit_transition = False
    level_ids_of_level_name = {}
    for levelid in range(1, len(energy_levels)):
        if hasattr(energy_levels[levelid], "levelname"):
            levelname = energy_levels[levelid].levelname

            levelnamenoJ = levelname.split("[")[0]
            if levelname != levelnamenoJ:  # levels are J split
                level_ids_of_level_name[levelname] = [levelid]
            elif not found_nonjsplit_transition:
                artisatomic.log_and_print(flog, "Found at least one transition specifying level name with no J value")
                found_nonjsplit_transition = True

            # keep track of the level ids of states that differ by J only
            # in case the collisional data level names are not J split
            try:
                level_ids_of_level_name[levelnamenoJ].append(levelid)
            except KeyError:
                level_ids_of_level_name[levelnamenoJ] = [levelid]

    filename = os.path.join(
        hillier_ion_folder(atomic_number, ion_stage), ions_data[(atomic_number, ion_stage)].folder, coldatafilename
    )
    artisatomic.log_and_print(flog, "Reading " + filename)
    coll_lines_in = 0
    with open(filename) as fcoldata:
        header_row = []
        temperature_index = -1
        num_expected_t_values = -1
        for line in fcoldata:
            row = line.split()
            if len(line.strip()) == 0:
                continue  # skip blank lines

            if line.startswith(("dln_OMEGA_dlnT = T/OMEGA* dOMEGAdt for HE2", "Johnson values")):  # found in col_ariii
                break

            if line.lstrip().startswith(r"Transition\T"):  # found the header row
                header_row = row
                if len(header_row) != num_expected_t_values + 1:
                    artisatomic.log_and_print(
                        flog,
                        (
                            f"WARNING: Expected {num_expected_t_values:d} temperature values, but header has"
                            f" {len(header_row):d} columns"
                        ),
                    )
                    num_expected_t_values = len(header_row) - 1
                    artisatomic.log_and_print(
                        flog,
                        f"Assuming header is incorrect and setting num_expected_t_values={num_expected_t_values:d}",
                    )

                temperatures = row[-num_expected_t_values:]
                artisatomic.log_and_print(
                    flog,
                    (
                        "Temperatures available for effective collision strengths (units of"
                        f" {t_scale_factor:.1e} K):\n{', '.join(temperatures)}"
                    ),
                )
                match_sorted_temperatures = sorted(
                    temperatures,
                    key=lambda t: abs(float(t.replace("D", "E")) * t_scale_factor - args.electrontemperature),
                )
                best_temperature = match_sorted_temperatures[0]
                temperature_index = temperatures.index(best_temperature)
                artisatomic.log_and_print(
                    flog, f"Selecting {float(temperatures[temperature_index].replace('D', 'E')) * t_scale_factor:.3f} K"
                )
                continue

            if len(row) >= 2:
                row_two_to_end = " ".join(row[1:])

                if row_two_to_end == "!Number of transitions":
                    number_expected_transitions = int(row[0])
                elif row_two_to_end.startswith("!Number of T values OMEGA tabulated at"):
                    num_expected_t_values = int(row[0])
                elif row_two_to_end == "!Scaling factor for OMEGA (non-file values)" and float(row[0]) != 1.0:
                    artisatomic.log_and_print(flog, "ERROR: non-zero scaling factor for OMEGA. what does this mean?")
                    sys.exit()

            if header_row != []:
                namefromnameto = "".join(row[:-num_expected_t_values])
                upsilonvalues = row[-num_expected_t_values:]

                namefrom, nameto = map(str.strip, namefromnameto.split("-"))
                upsilon = float(upsilonvalues[temperature_index].replace("D", "E"))
                coll_lines_in += 1

                try:
                    if level_ids_of_level_name[namefrom][0] > level_ids_of_level_name[nameto][0]:
                        artisatomic.log_and_print(
                            flog,
                            (
                                f"WARNING: Swapping transition levels {namefrom} {level_ids_of_level_name[namefrom]} "
                                f"-> {nameto} {level_ids_of_level_name[nameto]}."
                            ),
                        )
                        namefrom, nameto = nameto, namefrom

                    # add forbidden collisions between states within lower and upper terms if
                    # the upper and lower levels have no J specified
                    for id_lower in level_ids_of_level_name[namefrom]:
                        for id_lower2 in level_ids_of_level_name[namefrom]:
                            if id_lower < id_lower2 and (id_lower, id_lower2) not in upsilondict:
                                upsilondict[(id_lower, id_lower2)] = -2.0

                    for id_upper in level_ids_of_level_name[nameto]:
                        for id_upper2 in level_ids_of_level_name[nameto]:
                            if id_upper < id_upper2 and (id_upper, id_upper2) not in upsilondict:
                                upsilondict[(id_upper, id_upper2)] = -2.0

                    for id_lower in level_ids_of_level_name[namefrom]:
                        id_upper_list = [levelid for levelid in level_ids_of_level_name[nameto] if levelid > id_lower]
                        upper_g_sum = sum([energy_levels[id_upper].g for id_upper in id_upper_list])

                        for id_upper in id_upper_list:
                            # print(f'Transition {namefrom} (level {id_lower:d} in {level_ids_of_level_name[namefrom]}) -> {nameto} (level {id_upper:d} in {level_ids_of_level_name[nameto]})')
                            upsilonscaled = upsilon * energy_levels[id_upper].g / upper_g_sum
                            if (id_lower, id_upper) in upsilondict and upsilondict[(id_lower, id_upper)] >= 0.0:
                                artisatomic.log_and_print(
                                    flog,
                                    (
                                        f"ERROR: Duplicate collisional transition from {namefrom} <->"
                                        f" {nameto} ({id_lower} -> {id_upper}). Keeping existing collision strength of"
                                        f" {upsilondict[(id_lower, id_upper)]:.2e} instead of new value of"
                                        f" {upsilonscaled:.2e}."
                                    ),
                                )
                            else:
                                upsilondict[(id_lower, id_upper)] = upsilonscaled

                    # print(namefrom, nameto, upsilon)
                except KeyError:
                    unlisted_from_message = " (unlisted)" if namefrom not in level_ids_of_level_name else ""
                    unlisted_to_message = " (unlisted)" if nameto not in level_ids_of_level_name else ""
                    artisatomic.log_and_print(
                        flog,
                        (
                            f"Discarding upsilon={upsilon:.3f} for {namefrom}{unlisted_from_message} ->"
                            f" {nameto}{unlisted_to_message}"
                        ),
                    )

    if coll_lines_in < number_expected_transitions:
        print(
            f"ERROR: file specified {number_expected_transitions:d} transitions, but only {coll_lines_in:d} were found"
        )
        sys.exit()
    elif coll_lines_in > number_expected_transitions:
        artisatomic.log_and_print(
            flog,
            f"WARNING: file specified {number_expected_transitions:d} transitions, but {coll_lines_in:d} were found",
        )
    else:
        artisatomic.log_and_print(flog, f"Read {coll_lines_in} effective collision strengths ")
        artisatomic.log_and_print(flog, f"Output {len(upsilondict)} effective collision strengths ")

    return upsilondict


def get_photoiontargetfractions(energy_levels, energy_levels_upperion, hillier_photoion_targetconfigs, flog):
    targetlist = [[] for _ in energy_levels]
    targetlist_of_targetconfig = defaultdict(list)

    for lowerlevelid, energy_level in enumerate(energy_levels[1:], 1):
        if hillier_photoion_targetconfigs is None:
            continue
        if lowerlevelid in hillier_photoion_targetconfigs and hillier_photoion_targetconfigs[lowerlevelid] is None:
            continue  # photoionisation flagged as not available

        for targetconfig, targetconfig_fraction in hillier_photoion_targetconfigs[lowerlevelid]:
            if targetconfig not in targetlist_of_targetconfig:
                # sometimes the target has a slash, e.g. '3d7_4Fe/3d7_a4Fe'
                # so split on the slash and match all parts
                targetconfiglist = targetconfig.split("/")
                upperionlevelids = []
                for upperlevelid, upper_energy_level in enumerate(energy_levels_upperion[1:], 1):
                    upperlevelnamenoj = upper_energy_level.levelname.split("[")[0]
                    if upperlevelnamenoj in targetconfiglist:
                        upperionlevelids.append(upperlevelid)
                if not upperionlevelids:
                    upperionlevelids = [1]
                targetlist_of_targetconfig[targetconfig] = []

                summed_statistical_weights = sum([float(energy_levels_upperion[index].g) for index in upperionlevelids])
                for upperionlevelid in sorted(upperionlevelids):
                    statweight_fraction = energy_levels_upperion[upperionlevelid].g / summed_statistical_weights
                    targetlist_of_targetconfig[targetconfig].append((upperionlevelid, statweight_fraction))

            for upperlevelid, statweight_fraction in targetlist_of_targetconfig[targetconfig]:
                targetlist[lowerlevelid].append((upperlevelid, targetconfig_fraction * statweight_fraction))

        if len(targetlist[lowerlevelid]) == 0:
            targetlist[lowerlevelid].append((1, 1.0))

    return targetlist


def read_hyd_phixsdata():
    (
        hillier_ionization_energy_ev,
        hillier_energy_levels,
        transitions,
        transition_count_of_level_name,
        hillier_level_ids_matching_term,
    ) = read_levels_and_transitions(1, 1, open("/dev/null", "w"))

    hyd_filename = hillier_ion_folder(1, 1) + "/5dec96/hyd_l_data.dat"
    print(f"Reading hydrogen photoionization cross sections from {hyd_filename}")
    max_n = -1
    l_start_u = 0.0
    with open(hyd_filename) as fhyd:
        for line in fhyd:
            row = line.split()
            if " ".join(row[1:]) == "!Maximum principal quantum number":
                max_n = int(row[0])

            if " ".join(row[1:]) == "!L_ST_U":
                l_start_u = float(row[0].replace("D", "E"))

            if " ".join(row[1:]) == "!L_DEL_U":
                l_del_u = float(row[0].replace("D", "E"))

            if max_n >= 0 and line.strip() == "":
                break

        for line in fhyd:
            if line.strip() == "":
                continue

            n, l, num_points = (int(x) for x in line.split())
            e_threshold_ev = hc_in_ev_angstrom / float(hillier_energy_levels[n].lambdaangstrom)

            xs_values = []
            for line in fhyd:
                values_thisline = [float(x) for x in line.split()]
                xs_values = xs_values + values_thisline
                if len(xs_values) == num_points:
                    break
                elif len(xs_values) > num_points:
                    print(
                        f"ERROR: too many datapoints for (n,l)=({n},{l}), expected {num_points} but found"
                        f" {len(xs_values)}"
                    )
                    sys.exit()

            hyd_phixs_energygrid_ryd[(n, l)] = [
                e_threshold_ev / ryd_to_ev * 10 ** (l_start_u + l_del_u * index) for index in range(num_points)
            ]
            hyd_phixs[(n, l)] = [10 ** (8 + logxs) for logxs in xs_values]  # cross sections in Megabarns
            # hyd_phixs_f = interpolate.interp1d(hyd_energydivthreholdgrid[(n, l)], hyd_phixs[(n, l)], kind='linear', assume_sorted=True)

    hyd_filename = hillier_ion_folder(1, 1) + "/5dec96/gbf_n_data.dat"
    print(f"Reading hydrogen Gaunt factors from {hyd_filename}")
    max_n = -1
    l_start_u = 0.0
    with open(hyd_filename) as fhyd:
        for line in fhyd:
            row = line.split()
            if " ".join(row[1:]) == "!Maximum principal quantum number":
                max_n = int(row[0])

            if len(row) > 1:
                if row[1] == "!N_ST_U":
                    n_start_u = float(row[0].replace("D", "E"))
                elif row[1] == "!N_DEL_U":
                    n_del_u = float(row[0].replace("D", "E"))

            if max_n >= 0 and line.strip() == "":
                break

        for line in fhyd:
            if line.strip() == "":
                continue

            n, num_points = (int(x) for x in line.split())
            e_threshold_ev = hc_in_ev_angstrom / float(hillier_energy_levels[n].lambdaangstrom)

            gaunt_values = []
            for line in fhyd:
                values_thisline = [float(x) for x in line.split()]
                gaunt_values = gaunt_values + values_thisline
                if len(gaunt_values) == num_points:
                    break
                elif len(gaunt_values) > num_points:
                    print(f"ERROR: too many datapoints for n={n}, expected {num_points} but found {len(gaunt_values)}")
                    sys.exit()

            hyd_gaunt_energygrid_ryd[n] = [
                e_threshold_ev / ryd_to_ev * 10 ** (n_start_u + n_del_u * index) for index in range(num_points)
            ]
            hyd_gaunt_factor[n] = gaunt_values  # cross sections in Megabarns


def extend_ion_list(
    listelements: list[tuple[int, list[Union[int, tuple[int, str]]]]], maxionstage: Optional[int] = None
):
    for atomic_number, ion_stage in ions_data:
        if atomic_number == 1 or (maxionstage is not None and ion_stage > maxionstage):
            continue  # skip
        found_element = False
        for tmp_atomic_number, list_ions_handlers in listelements:
            if tmp_atomic_number == atomic_number:
                if ion_stage not in [x[0] if hasattr(x, "__getitem__") else x for x in list_ions_handlers]:
                    list_ions_handlers.append((ion_stage, "hillier"))
                    list_ions_handlers.sort(key=lambda x: x[0] if hasattr(x, "__getitem__") else x)
                found_element = True
        if not found_element:
            listelements.append(
                (
                    atomic_number,
                    [(ion_stage, "hillier")],
                )
            )
    listelements.sort(key=lambda x: x[0])
    return listelements


def plot_phixs(atomic_number, ion_stage, energy_levels, phixstables, reduced_phixstables, args):
    import matplotlib.pyplot as plt
    from pathlib import Path

    print("STARTING PHIXS PLOT")

    xgrid = np.linspace(
        1.0, 1.0 + args.phixsnuincrement * (args.nphixspoints + 1), num=args.nphixspoints + 1, endpoint=False
    )

    nrows = 25
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        sharey=False,
        figsize=(8, 6 * (0.25 + nrows * 0.4)),
        tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0},
    )

    ionstr = f"{artisatomic.elsymbols[atomic_number]} {artisatomic.roman_numerals[ion_stage]}"

    for levelnum, ax in enumerate(axes, 1):
        levelname = energy_levels[levelnum].levelname
        levelnamenoj = levelname.split("[")[0]
        if levelnamenoj not in phixstables:
            continue
        threshold = phixstables[levelnamenoj][0][0]

        levellabel = ionstr + " " + f"{levelnum} {levelnamenoj}"

        ax.plot(phixstables[levelnamenoj][:, 0], phixstables[levelnamenoj][:, 1], label=f"Hillier {levellabel}")

        ax.step(xgrid[:-1] * threshold, reduced_phixstables[levelnamenoj], where="mid", label=f"ARTIS {levellabel}")

        ax.legend(loc="best", handlelength=1, frameon=False, numpoints=1)
        ax.set_xlim(xmin=threshold * 0.97, xmax=threshold * 5)

    filenameout = f"phixs {ionstr}.pdf"
    fig.savefig(Path(filenameout).open("wb"), format="pdf")
    # plt.show()
    print(f"Saved {filenameout}")
    plt.close()
