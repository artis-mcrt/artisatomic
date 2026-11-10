A table of the ground states of some ions, from the NIST Atomic Spectra Database (ASD).
Reference: Kramida, A., Ralchenko, Yu., Reader, J. and NIST ASD Team, NIST Atomic Spectra
Database, https://physics.nist.gov/asd, doi:10.18434/T4W30F.

The "gsnist" handler in groundstatesonlynist.py reads groundstates.dat. It gives each ion one
level, the ground level, and no transitions. Use it for the top ion stage of an element when no
other source has data for that ion stage. No built-in ion selection uses the handler. Name it in
artisatomicionhandlers.json to use it. The jplt checksum set in tests/ uses it for Se V.

groundstates.dat is a table with a header line. A tab character separates the columns:

- Z: the atomic number;
- ion: the ion stage, where 1 is the neutral atom;
- IonizationEnergy: the ionisation energy in eV;
- g: the statistical weight of the ground level;
- config: the configuration of the ground level.

The table holds 51 ions. All of them have ion stage 5. The table came into the repository in
July 2024.
