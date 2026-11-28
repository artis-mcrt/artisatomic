"""Check the options of makeartisatomicfiles that give information or select the ion handlers file."""

import json
import os
from pathlib import Path

import pytest

from artisatomic import cli
from artisatomic import readadasdata
from artisatomic import readhillierdata
from artisatomic.iondata import handlers
from artisatomic.ionhandlers import find_handlers_file
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
    """A wrong path must not give the built-in ion selection, which can run for hours."""
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
    assert lines[lines.index("cmfgen") + 1].startswith(f"    {readhillierdata.description}. H II and He III: ")
    # the QUB cross sections of Co II are the difference between the two CMFGEN handlers
    assert lines[lines.index("cmfgen_qubphixs") + 1] != lines[lines.index("cmfgen") + 1]
    # U II and U III of the fac handler come from a different folder
    assert "Paper_Nd_U" in lines[lines.index("fac") + 1]
    assert "Co III has" in lines[lines.index("adas") + 1]
    assert all(handler.summary for handler in handlers.values())
    assert not (tmp_path / "artis_files").exists()


def test_listhandlers_prints_ascii_only(tmp_path, monkeypatch, capsys):
    """A standard output with the encoding cp1252 cannot take each character of the references."""
    monkeypatch.chdir(tmp_path)
    run_main(monkeypatch, "--listhandlers")
    output = capsys.readouterr().out
    assert output.isascii()
    assert "Flors" in output


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


@pytest.mark.parametrize("handlername", ["cmfgen", "cmfgen_qubphixs"])
@pytest.mark.parametrize(("atomic_number", "ion_stage"), [(1, 2), (2, 3)])
def test_bare_nucleus_needs_no_cmfgen_data_folder(tmp_path, monkeypatch, atomic_number, ion_stage, handlername):
    """The reader builds H II and He III from constants. Each is a top ion, so the run reads no CMFGEN file.

    The source: lines of adata.txt and transitiondata.txt name the constants, and not the CMFGEN data set.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(readhillierdata, "hillier_datadir", tmp_path / "atomic-data-hillier")
    handlersfile = tmp_path / "barenucleus.json"
    handlersfile.write_text(json.dumps([[atomic_number, [[ion_stage, handlername]]]]), encoding="utf-8")
    run_main(monkeypatch, "-ionhandlersfile", str(handlersfile))
    outputfolder = tmp_path / "artis_files"
    rows = (outputfolder / "compositiondata.txt").read_text(encoding="utf-8").splitlines()
    assert rows[3].split()[:4] == [str(atomic_number), "1", str(ion_stage), str(ion_stage)]
    for filename in ("adata.txt", "transitiondata.txt"):
        lines = (outputfolder / filename).read_text(encoding="utf-8").splitlines()
        assert [line for line in lines if line.startswith("# source:")] == [
            f"# source: {readhillierdata.bare_nucleus_description}"
        ]


@pytest.mark.parametrize(
    ("ion_handlers", "options", "needed"),
    [
        ([(1, [(2, "cmfgen")])], [], False),
        ([(1, [(2, "cmfgen_qubphixs")])], [], False),
        ([(2, [(3, "cmfgen")])], [], False),
        ([(8, [(1, "cmfgen"), (2, "cmfgen")])], ["--nophixs"], True),
        ([(38, [(1, "kurucz"), (2, "kurucz")])], [], True),
        ([(38, [(1, "kurucz"), (2, "kurucz")])], ["--nophixs"], False),
        ([(38, [(1, "kurucz"), (2, "kurucz")])], ["-nlevels_hydrogenic_for_unknown_phixs", "0"], False),
        # the top ion gets no cross sections
        ([(38, [(1, "kurucz")])], [], False),
        # the adas handler has cross sections only for Co III, so Sr I gets the estimate
        ([(38, [(1, "adas"), (2, "kurucz")])], [], True),
        ([(27, [(3, "adas"), (4, "adas")])], [], False),
        # the kurucz ion is the top ion
        ([(38, [(1, "adas"), (2, "kurucz")])], ["--nophixs"], False),
        # a handler with no parser for its level names makes no estimate
        ([(58, [(5, "mons"), (6, "mons")])], [], False),
        ([(34, [(4, "gsnist"), (5, "gsnist")])], [], False),
    ],
)
def test_cmfgen_data_needed(ion_handlers, options, needed):
    """The check before the run asks for the CMFGEN data set only if the run reads it for certain."""
    args = cli.build_parser().parse_args(options)
    assert cli.cmfgen_data_needed(ion_handlers, args) is needed


def test_kurucz_selection_with_no_data_folder_stops_before_the_output_folder(tmp_path, monkeypatch):
    """The hydrogenic estimate of Sr I reads the CMFGEN hydrogen tables, so the check comes before the read."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(readhillierdata, "hillier_datadir", tmp_path / "atomic-data-hillier")
    handlersfile = tmp_path / "sr.json"
    handlersfile.write_text(json.dumps([[38, [[1, "kurucz"], [2, "kurucz"]]]]), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match=r"setup_cmfgen_data\.sh"):
        run_main(monkeypatch, "-ionhandlersfile", str(handlersfile))
    assert not (tmp_path / "artis_files").exists()


def test_empty_handlers_file_stops_the_run(tmp_path, monkeypatch, capsys):
    """A shell redirect of --listions to the default name empties that file before the program reads it."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / inputhandlersfile).write_text("", encoding="utf-8")
    with pytest.raises(SystemExit) as exit_info:
        run_main(monkeypatch, "--listions")
    assert exit_info.value.code == 2
    assert f"The ion handlers file {inputhandlersfile} is empty." in capsys.readouterr().err


def test_ionhandlersfile_that_is_a_folder_stops_the_run(tmp_path, monkeypatch, capsys):
    """The message says that the path is a folder, and not that it does not exist."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "setdir").mkdir()
    with pytest.raises(SystemExit) as exit_info:
        run_main(monkeypatch, "-ionhandlersfile", "setdir")
    assert exit_info.value.code == 2
    assert "The ion handlers file setdir is a folder." in capsys.readouterr().err


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="the system has no named pipes")
def test_ionhandlersfile_can_be_a_named_pipe(tmp_path):
    """The checks of the file do not open a named pipe, because an open waits for a writer."""
    pipe = tmp_path / "selection.json"
    os.mkfifo(pipe)
    assert find_handlers_file(pipe) == pipe


@pytest.mark.skipif(not Path("/dev/fd").is_dir(), reason="the system has no /dev/fd")
def test_ionhandlersfile_can_be_a_pipe(tmp_path, monkeypatch, capsys):
    """The program reads the file one time only, so a pipe such as /dev/stdin can give the selection.

    The pipe holds the selection one time. A second read of the pipe finds no data, and the run stops.
    """
    monkeypatch.chdir(tmp_path)
    readend, writeend = os.pipe()
    os.write(writeend, json.dumps(selection).encode())
    os.close(writeend)
    try:
        run_main(monkeypatch, "--listions", "-ionhandlersfile", f"/dev/fd/{readend}")
    finally:
        os.close(readend)
    assert json.loads(capsys.readouterr().out) == selection


def test_handlers_file_that_is_not_json_names_the_file(tmp_path):
    """The message names the file, and not only the position of the JSON error."""
    handlersfile = tmp_path / "my_ions.json"
    handlersfile.write_text("[[26, [[1, ", encoding="utf-8")
    with pytest.raises(ValueError, match=r"my_ions\.json does not hold valid JSON"):
        get_ion_handlers(1, 5, None, handlersfile=handlersfile)
    # a byte that is not UTF-8, for example from a file in the encoding cp1252
    handlersfile.write_bytes(b'[[26, [[1, "gsn\xefst"]]]]')
    with pytest.raises(ValueError, match=r"my_ions\.json does not hold valid JSON"):
        get_ion_handlers(1, 5, None, handlersfile=handlersfile)


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16", "utf-16-be", "utf-32"])
def test_handlers_file_can_have_a_byte_order_mark(tmp_path, encoding):
    """Windows PowerShell 5.1 writes the output of a shell redirect in UTF-16 with a byte order mark."""
    handlersfile = tmp_path / "my_ions.json"
    # Python writes the byte order mark for utf-8-sig, utf-16 and utf-32, but not for utf-16-be
    bom = "﻿" if encoding == "utf-16-be" else ""
    handlersfile.write_bytes((bom + json.dumps(selection)).encode(encoding))
    assert get_ion_handlers(1, 5, None, handlersfile=handlersfile) == [
        (26, [(1, "gsnist"), (2, "gsnist")]),
        (34, [(5, "gsnist")]),
    ]


def test_folder_with_the_default_name_stops_the_run(tmp_path, monkeypatch, capsys):
    """A folder in the working directory gives the message of a folder, and not a traceback."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / inputhandlersfile).mkdir()
    with pytest.raises(SystemExit) as exit_info:
        run_main(monkeypatch, "--listions")
    assert exit_info.value.code == 2
    assert f"The ion handlers file {inputhandlersfile} is a folder." in capsys.readouterr().err


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="the system has no symbolic links")
def test_broken_link_with_the_default_name_stops_the_run(tmp_path, monkeypatch, capsys):
    """A link to no file must not give the built-in ion selection, which can run for hours."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / inputhandlersfile).symlink_to(tmp_path / "removed.json")
    with pytest.raises(SystemExit) as exit_info:
        run_main(monkeypatch, "--listions")
    assert exit_info.value.code == 2
    assert "is a link to a file that does not exist." in capsys.readouterr().err


def test_empty_file_of_the_option_needs_no_removal(tmp_path, monkeypatch, capsys):
    """The message tells the user to give a different file, and not to remove the file of the option."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "my_ions.json").write_text("", encoding="utf-8")
    with pytest.raises(SystemExit) as exit_info:
        run_main(monkeypatch, "-ionhandlersfile", "my_ions.json")
    assert exit_info.value.code == 2
    error = capsys.readouterr().err
    assert "Give a file that holds an ion selection." in error
    assert "Remove the file" not in error


def test_run_keeps_its_handlers_file_in_the_old_log_folder(tmp_path, monkeypatch):
    """An earlier release wrote a copy of the ion handlers into atomic_data_logs. A run can read that copy."""
    monkeypatch.chdir(tmp_path)
    oldfolder = tmp_path / "artis_files" / "atomic_data_logs"
    oldfolder.mkdir(parents=True)
    oldcopy = oldfolder / "artisatomicionhandlers.json"
    oldcopy.write_text(json.dumps([[34, [[5, "gsnist"]]]]), encoding="utf-8")
    (oldfolder / "se5.txt").write_text("the log of an earlier release\n", encoding="utf-8")
    run_main(monkeypatch, "-ionhandlersfile", str(oldcopy), "--nophixs")
    assert oldcopy.is_file()
    assert not (oldfolder / "se5.txt").exists()


def test_rename_of_the_old_adas_folder_does_not_change_the_handlers_file(tmp_path, monkeypatch, capsys):
    """The run reads the file that the checks found, also if the rename of the old ADAS folder moves it."""
    monkeypatch.chdir(tmp_path)
    oldfolder = tmp_path / "atomic-data-qub"
    oldfolder.mkdir()
    handlersfile = oldfolder / "ions.json"
    handlersfile.write_text(json.dumps(selection), encoding="utf-8")

    def move_old_folder() -> None:
        oldfolder.rename(tmp_path / "atomic-data-adas")

    monkeypatch.setattr("sys.argv", ["makeartisatomicfiles", "--listions", "-ionhandlersfile", str(handlersfile)])
    monkeypatch.setattr(readadasdata, "rename_old_adas_directory", move_old_folder)
    cli.main()
    assert json.loads(capsys.readouterr().out) == selection


def test_rename_that_moves_a_handlers_file_here_keeps_the_built_in_ion_selection(tmp_path, monkeypatch, capsys):
    """A file that the rename moves into the working directory does not replace the built-in ion selection."""
    monkeypatch.chdir(tmp_path)
    builtin = [(34, [(5, "gsnist")])]

    def move_a_file_here() -> None:
        (tmp_path / inputhandlersfile).write_text(json.dumps(selection), encoding="utf-8")

    monkeypatch.setattr("sys.argv", ["makeartisatomicfiles", "--listions"])
    monkeypatch.setattr(readadasdata, "rename_old_adas_directory", move_a_file_here)
    monkeypatch.setattr(cli, "get_builtin_ion_handlers", lambda **_limits: builtin)
    cli.main()
    assert json.loads(capsys.readouterr().out) == [[34, [[5, "gsnist"]]]]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("[[26, [[1, ", "does not hold valid JSON"),
        ('[[26, [[1, "no_such_handler"]]]]', "names the unknown handler 'no_such_handler'"),
        ("[[26, 1]]", "has no list of [ion_stage, handler] pairs"),
    ],
)
def test_bad_handlers_file_gives_a_usage_error(tmp_path, monkeypatch, capsys, content, message):
    """A bad file gives one line and the status 2, as a bad option does, and no traceback."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "my_ions.json").write_text(content, encoding="utf-8")
    with pytest.raises(SystemExit) as exit_info:
        run_main(monkeypatch, "-ionhandlersfile", "my_ions.json")
    assert exit_info.value.code == 2
    error = capsys.readouterr().err
    assert message in error.splitlines()[-1]
    assert "Traceback" not in error
    assert not (tmp_path / "artis_files").exists()
