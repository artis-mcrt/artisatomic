"""Check the options of makeartisatomicfiles that give information or select the ion handlers file."""

import json
from pathlib import Path

import pytest

from artisatomic import cli
from artisatomic import readadasdata
from artisatomic import readhillierdata
from artisatomic.iondata import handlers
from artisatomic.ionhandlers import get_ion_handlers
from artisatomic.ionhandlers import inputhandlersfile

selection = [[26, [[1, "gsnist"], [2, "gsnist"]]], [34, [[5, "gsnist"]]]]


def run_main(monkeypatch, *arguments: str) -> None:
    """Run main() with these arguments, and with no move of the old ADAS folder of the repository."""
    monkeypatch.setattr(readadasdata, "rename_old_adas_directory", lambda: None)
    monkeypatch.setattr("sys.argv", ["makeartisatomicfiles", *arguments])
    cli.main()


def test_default_handlers_file_name_is_the_same_in_both_modules():
    """The parser names the file in its help text with no import of ionhandlers."""
    assert Path(cli.defaulthandlersfilename) == inputhandlersfile


@pytest.mark.parametrize("option", ["-ionhandlersfile", "--ionhandlersfile"])
def test_ionhandlersfile_option_selects_the_ions(tmp_path, monkeypatch, option):
    """The option reads a file of a different name, and a file in the working directory is not necessary."""
    monkeypatch.chdir(tmp_path)
    handlersfile = tmp_path / "selections" / "my_ions.json"
    handlersfile.parent.mkdir()
    # groundstates.dat holds Se V, so the run needs no data set outside the repository
    oneion = [[34, [[5, "gsnist"]]]]
    handlersfile.write_text(json.dumps(oneion), encoding="utf-8")
    run_main(monkeypatch, option, str(handlersfile), "--nophixs")
    rows = (tmp_path / "artis_files" / "compositiondata.txt").read_text(encoding="utf-8").splitlines()
    assert rows[3].split()[:4] == ["34", "1", "5", "5"]
    assert json.loads((tmp_path / "artis_files" / cli.handlersrecordname).read_text(encoding="utf-8")) == oneion


def test_ionhandlersfile_option_comes_before_the_file_in_the_working_directory(tmp_path, monkeypatch):
    """A file that an earlier run left in the working directory does not change a run with the option."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / inputhandlersfile).write_text(json.dumps([[8, [[1, "cmfgen"]]]]), encoding="utf-8")
    handlersfile = tmp_path / "my_ions.json"
    handlersfile.write_text(json.dumps(selection), encoding="utf-8")
    assert get_ion_handlers(1, 5, None, handlersfile=handlersfile) == [
        (26, [(1, "gsnist"), (2, "gsnist")]),
        (34, [(5, "gsnist")]),
    ]


def test_ionhandlersfile_that_does_not_exist_stops_the_run(tmp_path, monkeypatch, capsys):
    """A wrong path must not give the built-in selection, which can run for hours."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exit_info:
        run_main(monkeypatch, "-ionhandlersfile", "no_such_file.json")
    assert exit_info.value.code == 2
    assert "The ion handlers file no_such_file.json does not exist." in capsys.readouterr().err
    assert not (tmp_path / "artis_files").exists()


def test_ion_limit_with_the_ionhandlersfile_option_stops_the_run(tmp_path, monkeypatch, capsys):
    """The message does not tell the user to remove a file that the command line names."""
    monkeypatch.chdir(tmp_path)
    handlersfile = tmp_path / "my_ions.json"
    handlersfile.write_text(json.dumps(selection), encoding="utf-8")
    with pytest.raises(SystemExit) as exit_info:
        run_main(monkeypatch, "-ionhandlersfile", str(handlersfile), "-maxionstage", "3")
    assert exit_info.value.code == 2
    error = capsys.readouterr().err
    assert "so that file selects the ions. Remove -maxionstage." in error
    assert "Remove the file" not in error


def test_a_bad_entry_names_the_file_of_the_option(tmp_path, monkeypatch):
    """The message names the file that the user gave, and not the default name."""
    monkeypatch.chdir(tmp_path)
    handlersfile = tmp_path / "my_ions.json"
    handlersfile.write_text(json.dumps([[26, [[1, "no_such_handler"]]]]), encoding="utf-8")
    with pytest.raises(ValueError, match=r"my_ions\.json names the unknown handler 'no_such_handler'"):
        get_ion_handlers(1, 5, None, handlersfile=handlersfile)


def test_listions_prints_a_valid_handlers_file_and_writes_nothing(tmp_path, monkeypatch, capsys):
    """The standard output is the selection only, so a user can write it to a file and run it again."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / inputhandlersfile).write_text(json.dumps(selection), encoding="utf-8")
    run_main(monkeypatch, "--listions")
    captured = capsys.readouterr()
    assert json.loads(captured.out) == selection
    # one line for each element
    assert captured.out.splitlines()[1] == '    [26, [[1, "gsnist"], [2, "gsnist"]]],'
    assert "Reading" in captured.err
    assert not (tmp_path / "artis_files").exists()

    listfile = tmp_path / "listed.json"
    listfile.write_text(captured.out, encoding="utf-8")
    assert get_ion_handlers(1, 5, None, handlersfile=listfile) == get_ion_handlers(1, 5, None)


def test_listhandlers_prints_each_handler_with_its_source(tmp_path, monkeypatch, capsys):
    """Each handler of the registry has a line, also a handler whose description is a function of the ion."""
    monkeypatch.chdir(tmp_path)
    run_main(monkeypatch, "--listhandlers")
    lines = capsys.readouterr().out.splitlines()
    assert [line for line in lines if line and not line.startswith(" ")] == sorted(handlers)
    assert lines[lines.index("cmfgen") + 1] == f"    {readhillierdata.description}"
    assert all(handler.summary for handler in handlers.values())
    assert not (tmp_path / "artis_files").exists()


def test_version_option_prints_the_version(monkeypatch, capsys):
    """The option stops the program with status 0 before a check of another option."""
    with pytest.raises(SystemExit) as exit_info:
        run_main(monkeypatch, "--version", "-maxionstage", "0")
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.startswith("artisatomic ")


@pytest.mark.parametrize(
    ("onedash", "twodashes"),
    [
        (("-output_folder", "x"), ("--output-folder", "x")),
        (("-maxionstage", "3"), ("--maxionstage", "3")),
        (("-nlevels_hydrogenic_for_unknown_phixs", "7"), ("--nlevels-hydrogenic-for-unknown-phixs", "7")),
        (("-nophixs",), ("--nophixs",)),
    ],
)
def test_each_option_has_a_form_with_one_dash_and_with_two_dashes(onedash, twodashes):
    """The job scripts of the users give the form with one dash, so that form must stay."""
    parser = cli.build_parser()
    assert parser.parse_args(list(onedash)) == parser.parse_args(list(twodashes))
    for action in parser._actions:  # ruff: ignore[private-member-access]
        if action.dest != "help":
            dashes = {len(option) - len(option.lstrip("-")) for option in action.option_strings}
            assert dashes == {1, 2}, action.option_strings


def test_cmfgen_selection_with_no_data_folder_names_the_setup_script(tmp_path, monkeypatch):
    """The run stops before it makes the output folder, and the message gives the download step."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(readhillierdata, "hillier_datadir", tmp_path / "atomic-data-hillier")
    (tmp_path / inputhandlersfile).write_text(json.dumps([[8, [[1, "cmfgen"], [2, "cmfgen"]]]]), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match=r"setup_cmfgen_data\.sh"):
        run_main(monkeypatch)
    assert not (tmp_path / "artis_files").exists()


def test_hydrogenic_tables_with_no_data_folder_name_the_setup_script(tmp_path, monkeypatch):
    """The hydrogenic estimate of an ion of a different handler reads the CMFGEN hydrogen tables."""
    monkeypatch.setattr(readhillierdata, "hillier_datadir", tmp_path / "atomic-data-hillier")
    # the module keeps the tables of an earlier read, and then reads no file
    monkeypatch.setattr(readhillierdata, "max_hyd_l_n", -1)
    monkeypatch.setattr(readhillierdata, "max_hyd_gaunt_n", -1)
    with pytest.raises(FileNotFoundError, match=r"setup_cmfgen_data\.sh"):
        readhillierdata.read_hyd_phixsdata()
