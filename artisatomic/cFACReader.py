#!/usr/bin/env python3
import linecache
import re
from datetime import date

import numpy as np
import pandas as pd
import scipy.constants as const
from tabulate import tabulate


# Constants
me = 9.10938e-28  # grams
PI = const.pi
NA = 6.0221409e23  # mol^-1
cspeed = 29979245800  # cm/s
kB = 0.6950356  # cm-1 K
echarge = 4.8e-10  # statC
hc = 4.1357e-15 * cspeed


def Convert_ev_cm(energyev):
    return energyev / hc


def GetLevels_FAC(filename):
    widths = [(0, 7), (7, 14), (14, 30), (30, 31), (32, 38), (38, 43), (44, 76), (76, 125), (127, 200)]
    names = ["Ilev", "Ibase", "Energy", "P", "VNL", "2J", "Configs_no", "Configs", "Config rel"]
    levels_FAC = pd.read_fwf(filename, header=10, index_col=False, colspecs=widths, names=names)
    levels_FAC["Config"] = levels_FAC["Configs"].apply(lambda x: " ".join(x.split(".")))
    levels_FAC["Config rel"] = levels_FAC["Config rel"].apply(lambda x: x.replace(".", " "))
    levels_FAC["g"] = levels_FAC["2J"].apply(lambda x: x + 1)
    levels_FAC = levels_FAC[["Config", "Config rel", "P", "2J", "g", "Energy"]].copy()
    levels_FAC["Config"] = levels_FAC["Config"].apply(lambda s: s.replace("1", ""))
    levels_FAC["Energy[cm^-1]"] = [Convert_ev_cm(e) for e in levels_FAC["Energy"]]
    levels_FAC = levels_FAC.rename(columns={"Energy": "Energy[eV]"})
    return levels_FAC


def GetLevels_cFAC(filename):
    widths = [(0, 7), (7, 14), (14, 30), (30, 31), (32, 38), (38, 43), (43, 150)]
    names = ["Ilev", "Ibase", "Energy", "P", "VNL", "2J", "Configs"]
    levels_cFAC = pd.read_fwf(filename, header=10, index_col=False, colspecs=widths, names=names)
    levels_cFAC["Config"] = levels_cFAC["Configs"].apply(lambda x: re.split(r"\s{2,}", x)[0])
    levels_cFAC["Config rel"] = levels_cFAC["Configs"].apply(lambda x: re.split(r"\s{2,}", x)[1])
    with open(filename) as file:
        for i, line in enumerate(file):
            if i == 9:
                nele = int(line.split("=")[1])
    for i in range(len(levels_cFAC["Config"])):
        c = levels_cFAC["Config"][i].split()
        n_temp = 0
        k = 0
        for s in c:
            if "*" in s:
                arr = np.array(s.split("*"), dtype=int)
                n_temp += arr[1]
                k += 1
            # if n_temp == nele:
            #     # print(c)
            #     # print(levels_cFAC['Config'][i])
            #     levels_cFAC['Config'][i] = ' '.join(c[k:])
            #     break
    levels_cFAC = levels_cFAC[["Config", "Config rel", "P", "2J", "Energy"]].copy()
    levels_cFAC["g"] = levels_cFAC["2J"].apply(lambda x: x + 1)
    levels_cFAC["Config"] = levels_cFAC["Config"].apply(lambda s: s.replace("1", ""))
    levels_cFAC["Energy[cm^-1]"] = [Convert_ev_cm(e) for e in levels_cFAC["Energy"]]
    levels_cFAC = levels_cFAC.rename(columns={"Energy": "Energy[eV]"})
    levels_cFAC = levels_cFAC[["Config", "Config rel", "P", "g", "Energy[eV]", "Energy[cm^-1]"]]
    return levels_cFAC


# In[5]:


def GetLevels(filename, Z, ion_name=0, date=date.today(), Get_csv=True, Get_dat=True):
    """Returns a dataframe of the energy levels extracted from ascii level output of cFAC and csv and dat files of the data.

    Parameters
    ----------
    data : str
        Filename of cFAC ascii output for the energy levels

    Z: int
        Ion atomic number

    ion_name : str, optional
        If omited, it will extract the ion name from the filename of the cFAC output file. Otherwise must be a str specifiying the ion name (e.g. 'NdII')

    date: str, optional
        Date of the last update of the calculations, by default today.

    Get_csv: bool, optional
        Flag to output a csv file with the data, by default True.

    Get_dat: bool, optional
        Flag to output a dat file with the data, by default True.
    """

    LastUpdated = date

    if ion_name == 0:
        ion_name = filename.split(".")[0]
    elif type(ion_name) != str:
        raise Exception("Input of ion_name must be a string")
    else:
        ...

    name = "Levels{}".format(ion_name)
    GState = linecache.getline(filename, 8)[8:]
    IonStage = Z - int(float(linecache.getline(filename, 6)[6:].strip()))
    version_FAC = linecache.getline(filename, 1).split(" ")[0]

    if version_FAC == "FAC":
        levels = GetLevels_FAC(filename)
    elif version_FAC == "cFAC":
        levels = GetLevels_cFAC(filename)
    else:
        raise Exception("No FAC-like code detected on output file")

    if Get_csv:
        with open("{}.csv".format(name), "a") as f:
            f.write("                {}                 \n".format(name))
            f.write("***********************************************************************\n")
            f.write("Last updated: {} \n".format(LastUpdated))
            f.write("Z: {}\n".format(Z))
            f.write("Ionization Stage: {}\n".format(2))
            f.write("Ground State Energy[eV]: {}\n".format(GState))
            f.write("Number Levels: {}\n".format(len(levels)))
            f.write("***********************************************************************\n")
            levels.to_csv(f, header=True, index=True)

    if Get_dat:
        T = tabulate(levels, showindex=False, headers="keys", tablefmt="plain", floatfmt=".5f")
        print(len(T.split("\n")[0]))
        with open("{}.dat".format(name), "a") as f:
            f.write("                {}                 \n".format(name))
            # f.write('\n')
            f.write("*" * len(T.split("\n")[0]) + "\n")
            f.write("Last updated: {} \n".format(LastUpdated))
            f.write("Z: {}\n".format(Z))
            f.write("Ionization Stage: {}\n".format(2))
            f.write("Ground State Energy[eV]: {}\n".format(GState))
            f.write("Number Levels: {}\n".format(len(levels)))
            f.write("*" * len(T.split("\n")[0]) + "\n")
            f.write(T)

    return levels


# In[7]:


def GetLines_FAC(filename):
    names = ["Upper", "2J1", "Lower", "2J2", "DeltaE[eV]", "gf", "TR_rate[1/s]", "Monopole"]

    widths = [(0, 7), (7, 11), (11, 17), (17, 21), (21, 35), (35, 49), (49, 63), (63, 77)]
    trans_FAC = pd.read_fwf(filename, header=11, index_col=False, colspecs=widths, names=names)
    trans_FAC["Wavelength[Ang]"] = trans_FAC["DeltaE[eV]"].apply(lambda e: (1 / Convert_ev_cm(e)) * 1e8)
    trans_FAC["DeltaE[cm^-1]"] = trans_FAC["DeltaE[eV]"].apply(lambda e: Convert_ev_cm(e))
    trans_FAC["TR_rate[1/s]"] = trans_FAC["TR_rate[1/s]"].apply(lambda tr: float(tr.rstrip(" -")))
    trans_FAC = trans_FAC[["Upper", "Lower", "DeltaE[eV]", "DeltaE[cm^-1]", "Wavelength[Ang]", "gf", "TR_rate[1/s]"]]
    return trans_FAC


def GetLines_cFAC(filename):
    names = ["Upper", "2J1", "Lower", "2J2", "DeltaE[eV]", "UTAdiff", "gf", "TR_rate[1/s]", "Monopole"]

    widths = [(0, 6), (6, 10), (10, 16), (16, 21), (21, 35), (35, 47), (47, 61), (61, 75), (75, 89)]
    trans_cFAC = pd.read_fwf(filename, header=11, index_col=False, colspecs=widths, names=names)
    trans_cFAC["Wavelength[Ang]"] = trans_cFAC["DeltaE[eV]"].apply(lambda e: (1 / Convert_ev_cm(e)) * 1e8)
    trans_cFAC["DeltaE[cm^-1]"] = trans_cFAC["DeltaE[eV]"].apply(lambda e: Convert_ev_cm(e))
    trans_cFAC = trans_cFAC[["Upper", "Lower", "DeltaE[eV]", "DeltaE[cm^-1]", "Wavelength[Ang]", "gf", "TR_rate[1/s]"]]
    trans_cFAC = trans_cFAC.astype({"Upper": "int64", "Lower": "int64"})
    return trans_cFAC


def GetLines(filename, Z, ion_name=0, date=date.today(), Get_csv=True, Get_dat=True):
    """Returns a dataframe of the transitions extracted from ascii level output of cFAC and csv and dat files of the data.

    Parameters
    ----------
    data : str
        Filename of cFAC ascii output for the transitions

    Z: int
        Ion atomic number

    ion_name : str, optional
        If omited, it will extract the ion name from the filename of the cFAC output file. Otherwise must be a str specifiying the ion name (e.g. 'NdII')

    date: str, optional
        Date of the last update of the calculations, by default today.

    Get_csv: bool, optional
        Flag to output a csv file with the data, by default True.

    Get_dat: bool, optional
        Flag to output a dat file with the data, by default True.
    """

    LastUpdated = date

    if ion_name == 0:
        ion_name = filename.split(".")[0]
    elif type(ion_name) != str:
        raise Exception("Input of ion_name must be a string")
    else:
        ...

    name = "Lines{}".format(ion_name)
    GState = linecache.getline(filename, 8)[8:]
    Multi = linecache.getline(filename, 11)[9:]
    IonStage = Z - int(float(linecache.getline(filename, 6)[6:].strip()))
    version_FAC = linecache.getline(filename, 1).split(" ")[0]

    if version_FAC == "FAC":
        lines = GetLines_FAC(filename)
    elif version_FAC == "cFAC":
        lines = GetLines_cFAC(filename)
    else:
        raise Exception("No FAC-like code detected on output file")

    if Get_csv:
        with open("{}.csv".format(filename), "a") as f:
            f.write("                {}                 \n".format(name))
            f.write("***********************************************************************\n")
            f.write("Last updated: {} \n".format(LastUpdated))
            f.write(
                "Multipole:{}  #negative values for electric transitions, positive for magnetic, 1 - dipole, 2 -"
                " quadrupole ...\n".format(Multi)
            )
            f.write("Z: {}\n".format(Z))
            f.write("Ionization Stage: {}\n".format(2))
            f.write("Number transitions: {}\n".format(len(lines)))
            f.write("***********************************************************************\n")
            lines.to_csv(f, header=True, index=True)

    if Get_dat:
        T = tabulate(
            lines,
            showindex=False,
            headers="keys",
            tablefmt="plain",
            floatfmt=(".0f", ".0f", ".5f", ".5f", ".5f", ".5e", ".5e"),
        )
        print(len(T.split("\n")[0]))
        with open("{}.dat".format(filename), "a") as f:
            f.write("                {}                 \n".format(name))
            f.write("*" * len(T.split("\n")[0]) + "\n")
            f.write("Last updated: {} \n".format(LastUpdated))
            f.write(
                "Multipole:{}  #negative values for electric transitions, positive for magnetic, 1 - dipole, 2 -"
                " quadrupole ...\n".format(Multi)
            )
            f.write("Z: {}\n".format(Z))
            f.write("Ionization Stage: {}\n".format(2))
            f.write("Number transitions: {}\n".format(len(lines)))
            f.write("*" * len(T.split("\n")[0]) + "\n")
            f.write(T)

    return lines


def GetOpacities(
    dfE,
    dfT,
    M,
    ion_name,
    T=10000,
    z=1,
    rho=1e-13,
    t=86400,
    wbins=10,
    minbin=0,
    maxbin=25001,
    ToFeather=False,
    ToCSV=False,
):
    """Returns a dataframe containing the expansion opacities in LTE and the 1-exp(-tau) factor, calculated for a specified wavelength bin.

    Parameters
    ----------
    dfE : DataFrame
        Data frame of the energy levels

    dfT : DataFrame
        Data frame of the transitions

    M: float
        Atomic mass of the ion, in g/mol

    ion_name : str
        String specifiying the ion name (e.g. 'NdII')

    T: float, optional
        Temperature at which the opacities are calculated, in Kelvin, default 10000

    z: float, optional
        Ionic population fraction, default 1

    rho: float, optional
        Density, in g/cm^3, default 1e-13

    t: int, optional
        Time after the merger, in seconds, default 86400 (1 day)

    wbins: int, optional
        Width of the wavelenght bin, in angstroms, default 10

    minbin: int,optional
        Minumum wavelenght considered when binning in angstroms, deafult 0

    maxbin: int optional
        Maximum wavelenght considered when binning in angstroms, default 25001

    ToFeather: bool, optional
    Flag to output a feather type file with the data, by default False.

    ToCSV: bool, optional
        Flag to output a csv file with the data, by default False.
    """

    # Constants

    g0 = dfE["g"]
    n = (z * NA / M) * rho
    C_tau = (PI * echarge**2) / (me * cspeed)
    C_op = 1 / (rho * cspeed * t)

    # Calculations
    dfT["EnergyLower"] = dfT["Lower"].apply(lambda x: dfE["Energy[cm^-1]"][x])

    boltz = np.exp(-dfT["EnergyLower"] / (kB * T))
    lbda = dfT["Wavelength[Ang]"] * 10 ** (-8)
    gf = dfT["gf"]
    tau = (1 / (4 * PI**2)) * C_tau * (n * lbda * t / g0) * gf * boltz

    # Expanding DF
    dfT["tau"] = tau
    dfT["1_exptau"] = 1 - np.exp(-dfT["tau"])
    dfT["Opacity"] = C_op * dfT["Wavelength[Ang]"] * dfT["1_exptau"] / wbins

    # Binning
    bins = np.arange(minbin, maxbin, wbins)
    wavemid = [b + wbins / 2 for b in bins[:-1]]
    dfT["Bins"] = pd.cut(dfT["Wavelength[Ang]"], bins)
    dfT_G = dfT[["Wavelength[Ang]", "1_exptau", "Opacity"]].groupby(dfT["Bins"]).sum()
    dfT_G["Wavemid"] = wavemid

    if ToFeather:
        dfT_G = dfT_G.reset_index()
        dfT_G.to_feather("{}_Opacity.feather".format(ion_name))
    if ToCSV:
        dfT_G.to_csv("{}_Opacity.csv".format(ion_name))

    print(dfT_G.head())
    return dfT_G
