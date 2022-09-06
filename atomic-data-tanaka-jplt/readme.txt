# Japan-Lithuania Opacity Database for Kilonova (version 1.0)
# M. Tanaka, D. Kato, G. Gaigalas, K. Kawaguchi, "Systematic opacity calculations for kilonovae" Monthly Notices of the Royal Astronomical Society 496 (2020) 1369-1392.

##################
# Atomic data table
##################

- Atomic number = 26 - 88
- Charge state = from neutral to triply charged ions

# File format
X = Atomic number
Y = Spectroscopic notation of ionization state (1 = I, 2 = II, 3 = III, and 4 = IV)

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
