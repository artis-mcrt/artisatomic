import argparse
import math
import re
from fractions import Fraction
from pathlib import Path

import artistools as at
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

rydberg_energy_eV = 13.6


def get_litvinov_ionisation_energy_data(
    litvinov_file_path: Path | str,
) -> np.ndarray:
    litvinov_file_path = Path(litvinov_file_path)

    with litvinov_file_path.open() as f:
        litvinov_cols = f.readline().lstrip("#").split()

    litvinov_data = pd.read_csv(
        litvinov_file_path,
        sep=r"\s+",
        skiprows=1,
        names=litvinov_cols,
    )

    E_ion_data = np.zeros((101, 101), dtype=float)
    E_ion_data[1, 0] = rydberg_energy_eV

    for Z in range(2, 101):
        row = litvinov_data.loc[litvinov_data["Z"] == Z]

        if row.empty:
            raise ValueError(f"No Litvinov data available for Z={Z}")

        row = row.iloc[0]
        for ion_stage in range(Z):
            if ion_stage == Z - 1:
                E_ion_data[Z, ion_stage] = row["1e-io"]

            else:
                higher = row[f"{Z - ion_stage}e-io"]
                lower = row[f"{Z - ion_stage - 1}e-io"]

                E_ion_data[Z, ion_stage] = higher - lower

    return E_ion_data


def parse_nist_ionization_table(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    data = []

    for tr in soup.find_all("tr", class_="bsl"):
        tds = tr.find_all("td")
        if len(tds) < 9:
            continue

        ion_charge_text = tds[2].get_text(strip=True)
        ion_charge = int(ion_charge_text.replace("+", "").replace("-", "0").strip())

        gs_config = "".join(tds[5].stripped_strings)
        gs_config = parse_econf(gs_config)

        ground_level = "".join(tds[6].stripped_strings)

        ionization_energy_text = "".join(tds[8].stripped_strings)
        ionization_energy_text = re.sub(r"[^\d\.]", "", ionization_energy_text)
        ionization_energy = float(ionization_energy_text) if ionization_energy_text else None

        if not ground_level or ground_level in ["°", "&deg;"]:
            ground_level = None
            g = None
        else:
            ground_level_clean = ground_level.replace("°", "").replace("&deg;", "").strip()

            j_value = None
            bracket_match = re.search(r"\)([0-9/]+)", ground_level_clean)
            if bracket_match:
                j_text = bracket_match.group(1)
                j_value = float(Fraction(j_text))
            else:
                letter_match = re.search(r"[A-Z](?:_|\^)?([0-9/]+)", ground_level_clean)
                if letter_match:
                    j_text = letter_match.group(1)
                    j_value = float(Fraction(j_text))

            g = 2 * j_value + 1 if j_value is not None else None
            ground_level = ground_level_clean

        data.append(
            {
                "ion_charge": ion_charge,
                "gs_config": gs_config,
                "ground_level": ground_level,
                "g_statistical": g,
                "ionization_energy_eV": ionization_energy,
            }
        )

    df = pd.DataFrame(data)
    return df


def parse_econf(conf_str):
    conf_str = conf_str.split("]")[-1]
    blocks = []
    pattern = re.compile(r"\d*[a-zA-Z]\d{0,3}")

    pos = 0
    while pos < len(conf_str):
        m = pattern.match(conf_str, pos)
        if not m:
            break
        block = m.group()
        pos = m.end()

        if blocks:
            prev_block = blocks.pop()
            # digits_between = re.findall(r"\d+", prev_block + block)

            combined = prev_block + block
            match = re.match(r"(\d*[a-zA-Z]\d{1,3})(.*)", combined)
            if match:
                first_part = match.group(1)
                rest = match.group(2)
                m2 = re.match(r"(\d*[a-zA-Z])(\d{1,3})", first_part)

                if m2 is None:
                    raise ValueError(f"Could not parse: {first_part}")
                letter = m2.group(1)
                digits = m2.group(2)
                if len(digits) == 1:
                    split_index = 0
                elif len(digits) == 2:
                    split_index = 1
                else:
                    split_index = 2
                blocks.append(letter + digits[:split_index])
                blocks.append(digits[split_index:] + rest)
            else:
                blocks.append(combined)
        else:
            blocks.append(block)

    return blocks


def orbital_value(orb):
    max_electrons = {"s": 2, "p": 6, "d": 10, "f": 14}
    result = 0

    orb_type = orb[1]
    max_e = max_electrons[orb_type]

    n_e = 1 if len(orb) == 2 else int(orb[2:])

    if n_e < max_e:
        holes = max_e - n_e
        result += min(n_e, holes)

    return result


def create_statopacdata(litvinov_file_path: Path | str):
    litvinov_ionisation_energy_data = get_litvinov_ionisation_energy_data(litvinov_file_path)

    artis_files_path = Path(__file__).parent.parent.absolute() / "artis_files"
    with open(artis_files_path / "stat_cc_opac_data.txt", mode="w", encoding="utf-8") as statopacity_filestream:
        statopacity_filestream.write(
            "Z ion_stage E_ion[eV] g_0 spin_cutoff omega_0 beta numb_active_electrons numb_lines\n"
        )
        for atomic_number in range(20, 101):
            nist_data = parse_nist_ionization_table(
                f"https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra={at.get_elsymbol(atomic_number)}&submit=Retrieve+Data&units=1&format=0&order=0&at_num_out=on&sp_name_out=on&ion_charge_out=on&el_name_out=on&seq_out=on&shells_out=on&level_out=on&ion_conf_out=on&e_out=0&unc_out=on&biblio=on"
            )
            for ionisation_stage in range(atomic_number + 1):
                numb_electrons = atomic_number - ionisation_stage
                ground_state_stat_weight = nist_data.loc[nist_data["ion_charge"] == ionisation_stage][
                    "g_statistical"
                ].item()
                # replace lowest ionisation stages by experimental NIST values
                if ionisation_stage < 3:
                    ionisation_energy_eV = nist_data.loc[nist_data["ion_charge"] == ionisation_stage][
                        "ionization_energy_eV"
                    ].item()
                else:
                    ionisation_energy_eV = litvinov_ionisation_energy_data[atomic_number - 1][ionisation_stage]
                electron_configs = nist_data.loc[nist_data["ion_charge"] == ionisation_stage]["gs_config"].item()

                eff_charge = atomic_number - 0.425 * numb_electrons
                beta_pereV = numb_electrons**1.25 / (rydberg_energy_eV * eff_charge)
                combinatorical_sum = 0
                for electron_config in electron_configs:
                    combinatorical_sum += orbital_value(electron_config)
                spin_cutoff_prefactor = 0.6 - 0.4 * np.exp(-0.4 * combinatorical_sum)
                spin_cutoff = spin_cutoff_prefactor * np.sqrt(numb_electrons)
                numb_levels = 10 ** (0.5 * combinatorical_sum)
                initial_state_density = (
                    spin_cutoff_prefactor
                    * np.sqrt(2 * np.pi * numb_electrons)
                    * numb_levels
                    / (np.exp(beta_pereV * ionisation_energy_eV) - 1)
                )
                N_lines = max(
                    1,
                    15
                    * initial_state_density**2
                    / (16 * np.pi**3 * spin_cutoff)
                    * (np.exp(2 * beta_pereV * ionisation_energy_eV) - 1)
                    * (np.exp(beta_pereV * ionisation_energy_eV) - 1),
                )
                print(f"    has {N_lines} lines")
                if math.isnan(ground_state_stat_weight):
                    ground_state_stat_weight = spin_cutoff
                statopacity_filestream.write(
                    f"{atomic_number} {ionisation_stage} {ionisation_energy_eV} {ground_state_stat_weight} {spin_cutoff} {initial_state_density} {beta_pereV} {combinatorical_sum} {N_lines}\n"
                )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        type=Path,
        required=True,
        help="Path to the ionisation data file by Y. Litvinov (not public)",
    )

    args = parser.parse_args()

    file_path = args.file

    if not file_path.exists():
        raise FileNotFoundError(f"Litvinov file not found: {file_path}")

    create_statopacdata(file_path)


if __name__ == "__main__":
    main()
