"""Install an output set with a durable record and recovery after an interrupted move."""

import json
import os
import shutil
import tempfile
import typing as t
from pathlib import Path

journal_name = ".artisatomic_transaction.json"


class InstallRecord(t.TypedDict):
    """The file identities and folders of one install."""

    version: int
    complete: bool
    work: str
    earlier: str
    work_identity: list[int]
    earlier_identity: list[int]
    new: dict[str, list[int]]
    old: dict[str, list[int]]


def identity(path: Path) -> list[int] | None:
    """Give the identity of a regular file, or None for an absent file."""
    if path.is_symlink() or (path.exists() and not path.is_file()):
        msg = f"The install cannot replace {path}, because it is not a regular file."
        raise RuntimeError(msg)
    if not path.exists():
        return None
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]


def folder_identity(path: Path) -> list[int]:
    """Give the identity of a folder without a symbolic link."""
    if path.is_symlink() or not path.is_dir():
        msg = f"The install cannot use the folder {path}."
        raise RuntimeError(msg)
    stat = path.stat()
    return [stat.st_dev, stat.st_ino]


def sync_folder(path: Path) -> None:
    """Flush the directory entries on a system that permits a directory file descriptor."""
    if os.name == "posix":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def write_record(outputfolder: Path, record: InstallRecord) -> None:
    """Flush a new record before its atomic replacement of the earlier record."""
    from artisatomic.base import open_for_atomic_write

    with open_for_atomic_write(outputfolder / journal_name) as stream:
        json.dump(record, stream)
        stream.flush()
        os.fsync(stream.fileno())
    sync_folder(outputfolder)


def read_record(outputfolder: Path) -> InstallRecord:
    """Validate the record before it can name a file or folder for recovery."""
    journal = outputfolder / journal_name
    if journal.is_symlink():
        msg = f"The install record {journal} is a symbolic link."
        raise RuntimeError(msg)
    record = json.loads(journal.read_text(encoding="utf-8"))
    valid = isinstance(record, dict) and record.get("version") == 1 and isinstance(record.get("complete"), bool)
    for field, prefix in (("work", ".artisatomic_run_"), ("earlier", ".artisatomic_earlier_")):
        name = record.get(field) if isinstance(record, dict) else None
        valid = valid and isinstance(name, str) and Path(name).name == name and name.startswith(prefix)
        key = record.get(f"{field}_identity") if isinstance(record, dict) else None
        valid = valid and isinstance(key, list) and len(key) == 2 and all(isinstance(value, int) for value in key)
    for field in ("old", "new"):
        files = record.get(field) if isinstance(record, dict) else None
        valid = valid and isinstance(files, dict)
        if isinstance(files, dict):
            for name, key in files.items():
                valid = (
                    valid
                    and isinstance(name, str)
                    and bool(name)
                    and Path(name).name == name
                    and name not in {".", "..", journal_name}
                    and isinstance(key, list)
                    and len(key) == 4
                    and all(isinstance(value, int) for value in key)
                )
    if not valid:
        msg = f"The install record {journal} is invalid. The run keeps all recovery files."
        raise RuntimeError(msg)
    return t.cast("InstallRecord", record)


def check_recovery_folders(outputfolder: Path, record: InstallRecord) -> tuple[Path, Path]:
    """Check each recovery folder and each file before a move or deletion."""
    folders = []
    for field, files, expected_identity in (
        ("work", record["new"], record["work_identity"]),
        ("earlier", record["old"], record["earlier_identity"]),
    ):
        folder = outputfolder / record[field]
        if folder.exists() or folder.is_symlink():
            if folder_identity(folder) != expected_identity:
                msg = f"The recovery folder {folder} has a different identity. The run keeps all recovery files."
                raise RuntimeError(msg)
            for child in folder.iterdir():
                if child.name not in files or identity(child) != files[child.name]:
                    msg = f"The recovery file {child} changed. The run keeps all recovery files."
                    raise RuntimeError(msg)
        elif not record["complete"] and not (
            field == "earlier" and all(identity(outputfolder / name) == key for name, key in record["old"].items())
        ):
            msg = f"The recovery folder {folder} is absent. The run keeps all recovery files."
            raise RuntimeError(msg)
        folders.append(folder)
    return folders[0], folders[1]


def recover_install(outputfolder: Path) -> None:
    """Finish a committed install or restore its earlier files. The caller must hold the output lock."""
    journal = outputfolder / journal_name
    if not journal.exists() and not journal.is_symlink():
        return
    record = read_record(outputfolder)
    workfolder, earlierfolder = check_recovery_folders(outputfolder, record)
    names = sorted(record["new"].keys() | record["old"].keys())
    # Check the complete state before any move. An unknown file must remain in place.
    for name in names:
        current = identity(outputfolder / name)
        new = record["new"].get(name)
        old = record["old"].get(name)
        if record["complete"]:
            valid = current == new
        else:
            valid = any(current == candidate for candidate in (None, new, old))
            if new is not None:
                valid = valid and (int(identity(workfolder / name) == new) + int(current == new) == 1)
            if old is not None:
                valid = valid and (int(identity(earlierfolder / name) == old) + int(current == old) == 1)
        if not valid:
            msg = f"The output file {outputfolder / name} changed. The run keeps all recovery files."
            raise RuntimeError(msg)
    if not record["complete"]:
        for name, key in record["new"].items():
            if identity(outputfolder / name) == key:
                (outputfolder / name).replace(workfolder / name)
        sync_folder(workfolder)
        sync_folder(outputfolder)
        for name in record["old"]:
            if (earlierfolder / name).exists():
                (earlierfolder / name).replace(outputfolder / name)
        if earlierfolder.exists():
            sync_folder(earlierfolder)
        sync_folder(outputfolder)
        if earlierfolder.exists():
            earlierfolder.rmdir()
    else:
        if earlierfolder.exists():
            shutil.rmtree(earlierfolder)
        if workfolder.exists():
            workfolder.rmdir()
    journal.unlink()
    sync_folder(outputfolder)


def install_files(workfolder: Path, outputfolder: Path) -> Path:
    """Install a complete output set under the output lock. Keep its durable record until cleanup."""
    if (outputfolder / journal_name).exists():
        msg = f"The folder {outputfolder} has an unfinished install. Recover it before a new install."
        raise RuntimeError(msg)
    new = {path.name: identity(path) for path in workfolder.iterdir()}
    old = {
        name: key for name in sorted({*new, "phixsdata_v2.txt"}) if (key := identity(outputfolder / name)) is not None
    }
    earlierfolder = Path(tempfile.mkdtemp(prefix=".artisatomic_earlier_", dir=outputfolder))
    record: InstallRecord = {
        "version": 1,
        "complete": False,
        "work": workfolder.name,
        "earlier": earlierfolder.name,
        "work_identity": folder_identity(workfolder),
        "earlier_identity": folder_identity(earlierfolder),
        "new": {name: key for name, key in new.items() if key is not None},
        "old": old,
    }
    try:
        move_output_files(workfolder, outputfolder, earlierfolder, record)
    except BaseException:
        if (outputfolder / journal_name).exists():
            recover_install(outputfolder)
        elif earlierfolder.exists():
            earlierfolder.rmdir()
        raise
    return earlierfolder


def move_output_files(workfolder: Path, outputfolder: Path, earlierfolder: Path, record: InstallRecord) -> None:
    """Flush the record and files before each step of the install."""
    for name in record["new"]:
        with (workfolder / name).open("rb") as stream:
            os.fsync(stream.fileno())
    sync_folder(workfolder)
    write_record(outputfolder, record)
    for name in sorted(record["old"]):
        (outputfolder / name).replace(earlierfolder / name)
        sync_folder(earlierfolder)
        sync_folder(outputfolder)
    for name in sorted(record["new"]):
        (workfolder / name).replace(outputfolder / name)
        sync_folder(outputfolder)
        sync_folder(workfolder)
    record["complete"] = True
    write_record(outputfolder, record)
