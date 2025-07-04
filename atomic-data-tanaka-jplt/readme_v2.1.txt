# Japan-Lithuania Opacity Database for Kilonova (version 1.1)
# HULLAC
# M. Tanaka, D. Kato, G. Gaigalas, K. Kawaguchi, "Systematic opacity calculations for kilonovae" Monthly Notices of the Royal Astronomical Society 496 (2020) 1369-1392.

# Japan-Lithuania Opacity Database for Kilonova (version 2.0)
# HULLAC
# D. Kato, M. Tanaka, G. Gaigalas, L. Kitovienė, P. Rynkun, "Systematic opacity calculations for kilonovae – II. Improved atomic data for singly ionized lanthanides" Monthly Notices of the Royal Astronomical Society 535 (2024) 2670-2686.
# GRASP
# G. Gaigalas, D. Kato, P. Rynkun, L. Radžiūtė, M. Tanaka, "Extended Calculations of Energy Levels and Transition Rates of Nd ii-iv Ions for Application to Neutron Star Mergers" The Astrophysical Journal Supplement Series 240 (2019) 29.
# L. Radžiūtė, G. Gaigalas, D. Kato, P. Rynkun, M. Tanaka, "Extended Calculations of Energy Levels and Transition Rates for Singly Ionized Lanthanide Elements. I. Pr–Gd" The Astrophysical Journal Supplement Series 248 (2020) 17.
# L. Radžiūtė, G. Gaigalas, D. Kato, P. Rynkun, M. Tanaka, "Extended Calculations of Energy Levels and Transition Rates for Singly Ionized Lanthanide Elements. II. Tb−Yb" The Astrophysical Journal Supplement Series 257 (2021) 29.

# Japan-Lithuania Opacity Database for Kilonova (version 2.1)
# GRASP
L. Kitovienė, G. Gaigalas, P. Rynkun, M. Tanaka, D. Kato, "Theoretical Investigation of the Ge Isoelectronic Sequence" Journal of Physical and Chemical Reference Data 53 (2024) 033101.
G. Gaigalas, P. Rynkun, N. Domoto, M. Tanaka, D. Kato, L. Kitovienė, "Theoretical investigation of energy levels and transitions for Ce III with applications to kilonova spectra" Monthly Notices of the Royal Astronomical Society 530 (2024) 5220.
L. Radžiūtė, G. Gaigalas, "Theoretical investigation of Sb-like sequence: Sb I, Te II, I III, Xe IV, and Cs V" Atomic Data and Nuclear Data Tables 152 (2023) 101585.
L. Radžiūtė, G. Gaigalas, "Energy levels and transition properties for As-like ions Se II, Br III, Kr IV, Rb V, and Sr VI" Atomic Data and Nuclear Data Tables 147 (2022) 101515.
P. Rynkun, S. Banerjee, G. Gaigalas, M. Tanaka, L. Radžiūtė, D. Kato, "Theoretical investigation of energy levels and transition for Ce IV" Astronomy & Astrophysics 658 (2022) A82.
G. Gaigalas, P. Rynkun, S. Banerjee, M. Tanaka, D. Kato, L. Radžiūtė, "Theoretical investigation of energy levels and transitions for Pr iv" Monthly Notices of the Royal Astronomical Society 517 (2022) 281.



##################
# Atomic data table
##################

- Atomic number = 26 - 88
- Charge state = from neutral to triply charged ions

# File format
X = Atomic number
Y = Spectroscopic notation of ionization state (1 = I, 2 = II, 3 = III, 4 = IV, 5 = V)

- XX_Y.txt
Spectrum name
XX, Y
Number of Energy levels, Number of Transitions
Closed shells
Ionization potential:IP (eV) from NIST Atomic Spectra Database (version 5.9)
Energy levels below IP:
#num / 2*J+1 / Parity / E (eV) / Configuration
Electric dipole transitions:
Upper #num / Lower #num / Wavelength (nm) / g_up*A (s^-1) / log10(g_low*f)
