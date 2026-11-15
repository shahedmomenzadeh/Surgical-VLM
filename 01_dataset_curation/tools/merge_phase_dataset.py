"""Merge accepted phase VLM samples into the final dataset.

The source phase dataset already has the final split/procedure layout. This
tool adds its procedure folders to ``dataset/{Train,Validation,Test}`` without
re-splitting, overwriting, or making any model calls.

Usage:
    python -m video_pipeline.tools.merge_phase_dataset
    python -m video_pipeline.tools.merge_phase_dataset --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from tqdm import tqdm

from ..config import build_settings


SPLITS = ("Train", "Validation", "Test")


def _procedure_dirs(source_dir: Path) -> list[tuple[str, Path]]:
    entries = []
    for split in SPLITS:
        split_dir = source_dir / split
        if not split_dir.is_dir():
            continue
        entries.extend(
            (split, path)
            for path in sorted(split_dir.iterdir())
            if path.is_dir() and not path.name.startswith(".")
        )
    return entries


def _copy_file(source: Path, destination: Path, materialization: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp")
    if temporary_path.exists() or temporary_path.is_symlink():
        temporary_path.unlink()

    try:
        if materialization == "copy":
            shutil.copy2(source, temporary_path)
        elif materialization == "hardlink":
            os.link(source, temporary_path)
        else:
            raise ValueError(f"Unsupported materialization: {materialization}")
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists() or temporary_path.is_symlink():
            temporary_path.unlink()


def _verify_merge(source_dir: Path, destination_dir: Path) -> dict[str, int]:
    source_files = []
    destination_files = []
    for split, source_procedure in _procedure_dirs(source_dir):
        destination_procedure = destination_dir / split / source_procedure.name
        for source_file in sorted(path for path in source_procedure.rglob("*") if path.is_file()):
            relative = source_file.relative_to(source_procedure)
            source_files.append((split, source_procedure.name, relative))
            destination_files.append(
                (split, source_procedure.name, relative,
                 destination_procedure / relative))

    missing = [item for item in destination_files if not item[3].is_file()]
    size_mismatch = [
        item for item in destination_files
        if item[3].is_file()
        and (source_dir / item[0] / item[1] / item[2]).stat().st_size != item[3].stat().st_size
    ]
    return {
        "source_files": len(source_files),
        "destination_files": len(destination_files) - len(missing),
        "missing_files": len(missing),
        "size_mismatches": len(size_mismatch),
    }


def merge_phase_dataset(
    source_dir: str | Path,
    destination_dir: str | Path,
    materialization: str = "copy",
    dry_run: bool = False,
) -> dict[str, int]:
    source_dir = Path(source_dir).resolve()
    destination_dir = Path(destination_dir).resolve()
    materialization = materialization.lower()

    if materialization not in {"copy", "hardlink"}:
        raise ValueError("materialization must be 'copy' or 'hardlink'")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Phase source directory not found: {source_dir}")
    destination_dir.mkdir(parents=True, exist_ok=True)

    procedures = _procedure_dirs(source_dir)
    collisions = [
        destination_dir / split / procedure.name
        for split, procedure in procedures
        if (destination_dir / split / procedure.name).exists()
    ]
    if collisions:
        examples = ", ".join(str(path) for path in collisions[:5])
        raise FileExistsError(
            f"Refusing to overwrite {len(collisions)} existing procedure folder(s): {examples}")

    file_count = sum(
        1 for _, procedure in procedures
        for path in procedure.rglob("*") if path.is_file()
    )
    clip_count = sum(
        1 for _, procedure in procedures
        for path in procedure.glob("clip_*.mp4") if path.is_file()
    )
    print(f"Phase procedures: {len(procedures)}")
    print(f"Phase clips: {clip_count}")
    print(f"Files to add: {file_count}")
    print(f"Destination: {destination_dir}")
    print(f"Materialization: {materialization}")

    if dry_run:
        return {
            "procedures": len(procedures), "clips": clip_count,
            "files": file_count, "copied": 0, "errors": 0,
        }

    copied = 0
    errors = 0
    for split, source_procedure in tqdm(procedures, desc="Merging phase procedures"):
        destination_procedure = destination_dir / split / source_procedure.name
        try:
            for source_file in sorted(path for path in source_procedure.rglob("*") if path.is_file()):
                relative = source_file.relative_to(source_procedure)
                _copy_file(
                    source_file,
                    destination_procedure / relative,
                    materialization,
                )
                copied += 1
        except Exception:
            errors += 1
            if destination_procedure.exists():
                shutil.rmtree(destination_procedure)

    verification = _verify_merge(source_dir, destination_dir)
    result = {
        "procedures": len(procedures),
        "clips": clip_count,
        "files": file_count,
        "copied": copied,
        "errors": errors,
        **verification,
    }
    print(
        "Completed: "
        f"copied={copied}, errors={errors}, "
        f"missing={verification['missing_files']}, "
        f"size_mismatches={verification['size_mismatches']}"
    )
    if errors or verification["missing_files"] or verification["size_mismatches"]:
        raise RuntimeError(f"Phase merge verification failed: {result}")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Merge accepted phase VLM samples into the final dataset.")
    parser.add_argument("--config", default=None,
                        help="Path to config.yaml")
    parser.add_argument("--source", default=None,
                        help="Phase VLM source directory")
    parser.add_argument("--destination", default=None,
                        help="Final dataset directory")
    parser.add_argument("--materialization", choices=("copy", "hardlink"),
                        default="copy", help="How to materialize files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report the merge without writing files")
    args = parser.parse_args(argv)

    settings = build_settings(config_path=args.config)
    merge_phase_dataset(
        args.source or settings.phase_sft.output_dir,
        args.destination or settings.dataset_dir,
        materialization=args.materialization,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
