# AGENTS.md

Guidance for AI coding agents that work in this repository.

## Language style

Write all English in ASD-STE100 (Simplified Technical English). This rule applies to docstrings, comments, commit messages, documentation, and log or error messages. Follow these STE rules:

- Use the active voice.
- Use simple verb tenses (past, present, future).
- Write short sentences. Use a maximum of 20 words in an instruction. Use a maximum of 25 words in a description.
- Give only one instruction in each sentence.
- Use one approved word for each meaning. Do not use a different word for the same thing.
- Use a noun with an article ("the file", "a level") where possible.
- Do not use slang or idioms.

Use these terms and spellings in the prose. Identifiers, output-file formats and quoted file text are exempt.

- British spellings: ionisation, photoionisation, normalise, behaviour.
- Two words: "cross section", "data set".
- "level id": the zero-based number of a level in memory.
- "file index": the number of a level in a source file.

## What this project is

artisatomic converts published atomic data (for example CMFGEN, NORAD, Kurucz, JPLT, DREAM, Floers+25, QUB, MONS) into the ARTIS atomic database format. The output files are adata.txt, compositiondata.txt, transitiondata.txt, and phixsdata_v2.txt. The command `makeartisatomicfiles` starts the conversion. The tool is not user friendly by design. To change ions or data sources, edit the Python code or supply an ion handlers JSON file.

The command-line scripts (`makeartisatomicfiles`, `makerecombratefile`, and `makechargetransferfile`) are the only callers of the package, and this repository holds all of them. Change a function signature or a module layout when you must.

## Setup

The project requires Python >= 3.13 and uses [uv](https://docs.astral.sh/uv/):

```sh
uv sync --frozen
```

This installs the package in editable mode with the `dev` dependency group.

Some readers require large external data sets. The `atomic-data-*` directories contain download scripts (for example `atomic-data-hillier/setup_cmfgen_data.sh`) but not the data itself. Run the applicable script before you run tests that read that data. The Kurucz, QUB, MONS and Floers+25 tests read committed samples, so the full test suite needs only the CMFGEN download. The charge transfer source files in `atomic-data-chargetransfer` are small and tracked.

## Commands

Run all commands through uv so they use the locked environment:

- Tests: `uv run -m pytest`
- Lint and autofix: `uv run ruff check --fix .`
- Format: `uv run ruff format .`
- Type checks: `uv run pyrefly check`, `uv run ty check`
- Pre-commit hooks: `prek install` once, then hooks run on each commit

CI (`.github/workflows/test.yml`) runs the format check, both type checks, the tests, and the output checksums. Run them locally before you push.

## Output checksums

The output files must stay byte-identical unless the change intends a different output. Each directory in `tests/` (except `chargetransfer`) holds an ion handlers file and the MD5 checksums of the four output files. [tests/README.md](tests/README.md) gives the recipe to regenerate a set and explains which ions each set covers. Two rules from it matter for every run:

- Set `ARTISATOMIC_TESTMODE=1`. It redirects the Kurucz, QUB, MONS and Floers+25 readers to their committed samples.
- Remove `artisatomicionhandlers.json` from the repository root after the run. A copy left there silently overrides the ion selection of every later run.

When a change alters the output on purpose, verify the new values, then regenerate every set that the change touches.

## Code layout

- `artisatomic/__init__.py` re-exports nothing. The package has no public API. Import each name from the submodule that defines it, in the source and in the tests. Python binds a submodule to its package when something imports it, so `from artisatomic import readqubdata` needs no line in `__init__.py`.
- `artisatomic/base.py` holds shared helpers and constants. It imports nothing from the package, which prevents circular imports. Submodules import from `artisatomic.base`, not from `artisatomic`.
- `artisatomic/read*.py` modules each read one atomic data source. `artisatomic/ionhandlers.py` selects a handler for each ion.
- `artisatomic/output.py` writes the ARTIS output files. `artisatomic/phixs.py` processes photoionisation cross sections.
- `artisatomic/iondata.py` reads one ion and holds the result in an `IonData` record. Its `handlers` registry maps each handler name to the reader, the level-name parser, the return shape, and the optional collision-strength and photoionisation readers. Add a data source with an entry there.
- `artisatomic/levelnames.py` parses the parts of a level name that more than one reader needs, for example the parity of a configuration.
- `artisatomic/cli.py` contains the `makeartisatomicfiles` entry point. `artisatomic/makerecombratefile.py` contains the `makerecombratefile` entry point. `artisatomic/makechargetransferfile.py` contains the `makechargetransferfile` entry point.
- `tests/` contains test configurations and reference checksums for each data source. `artisatomic/test_artisatomic.py` and `artisatomic/test_chargetransfer.py` contain the test functions.

## Code style

- Ruff enforces the style: line length 120, `select = ["ALL"]` with the exceptions listed in `pyproject.toml`, single-line isort imports (`from x import y`, one name for each line).
- Do not add a `Returns:` or `Raises:` docstring section that repeats the summary line. Say it once in prose.
- Many float equality comparisons in this code are deliberate (divide-by-zero guards, exact values from data files). Do not "fix" them.
- Physics-style variable names (for example `A`, `lowerlevel`, `energy_ev`) are permitted; several ruff naming rules are disabled for this reason.
- Write a code comment only to state a constraint that the code cannot show.

## Version control

- Commit messages: write a short imperative summary line in STE.
- `_version.py` is generated by setuptools_scm. Do not edit or commit it.
- `uv.lock` is managed by the uv pre-commit hooks. Change dependencies in `pyproject.toml` and let `uv lock` update the lockfile.
