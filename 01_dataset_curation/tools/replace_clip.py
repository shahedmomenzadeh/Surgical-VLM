"""Resolve ``.ref.txt`` placeholders left by apply_mapping_all.

On filesystems without symlink support, ``apply_mapping_all`` writes
``clip_XX.mp4.ref.txt`` placeholders pointing at the real video. This tool
copies the video over the placeholder and removes it.

Usage:
    python -m video_pipeline.tools.replace_clip [--mapped-dir DIR] [--dry-run]
"""

import os
import shutil
import argparse
from pathlib import Path


def find_ref_files(mapped_dir):
    return sorted(Path(mapped_dir).rglob("*.ref.txt"))


def read_source_path(ref_path):
    try:
        lines = ref_path.read_text(encoding="utf-8").strip().splitlines()
        for line in lines:
            line = line.strip()
            if line and not line.lower().startswith("video file"):
                return line
    except Exception as e:
        print(f"  ⚠️  Could not read {ref_path}: {e}")
    return None


def resolve_ref(ref_path, dry_run):
    src_str = read_source_path(ref_path)
    if not src_str:
        return False, f"Could not parse source path from {ref_path.name}"

    src = Path(src_str)
    if not src.exists():
        return False, f"Source not found: {src}"

    dst = ref_path.with_suffix("")      # removes .txt  → clip_01.mp4.ref
    if dst.suffix == ".ref":
        dst = dst.with_suffix("")       # removes .ref  → clip_01.mp4

    if dst.exists():
        return False, f"Destination already exists, skipping: {dst.name}"

    if dry_run:
        return True, f"[DRY RUN] Would copy {src} → {dst}"

    try:
        shutil.copy2(src, dst)
        ref_path.unlink()
        return True, f"✅ {dst.name}"
    except Exception as e:
        return False, f"❌ Copy failed for {dst.name}: {e}"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Replace .ref.txt placeholders with real video files.")
    parser.add_argument("--mapped-dir", default=None,
                        help="Root of the mapped clip folder "
                             "(default: <workspace>/clip_dataset_mapped)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would be copied without making any changes")
    args = parser.parse_args(argv)

    from ..config import build_settings
    settings = build_settings()
    mapped_dir = args.mapped_dir or os.path.join(
        os.path.dirname(settings.clips_dir), "clip_dataset_mapped")

    if not os.path.isdir(mapped_dir):
        print(f"❌ Directory not found: {mapped_dir}")
        return

    print(f"\n🔍 Scanning for .ref.txt files in: {os.path.abspath(mapped_dir)}")
    ref_files = find_ref_files(mapped_dir)

    if not ref_files:
        print("✅ No .ref.txt placeholders found — nothing to do.")
        return

    print(f"   Found {len(ref_files)} placeholder(s)\n")
    if args.dry_run:
        print("   ⚠️  DRY RUN — no files will be copied or deleted\n")

    success_count = skip_count = error_count = 0

    for ref_path in ref_files:
        ok, msg = resolve_ref(ref_path, dry_run=args.dry_run)
        if ok:
            print(f"  {msg}")
            success_count += 1
        elif "skipping" in msg.lower() or "already exists" in msg.lower():
            print(f"  ⏩ {msg}")
            skip_count += 1
        else:
            print(f"  ❌ {msg}")
            error_count += 1

    print("\n" + "=" * 50)
    if args.dry_run:
        print("  DRY RUN complete — no changes made")
    else:
        print(f"  ✅ Resolved : {success_count}")
        print(f"  ⏩ Skipped  : {skip_count}")
        print(f"  ❌ Errors   : {error_count}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
