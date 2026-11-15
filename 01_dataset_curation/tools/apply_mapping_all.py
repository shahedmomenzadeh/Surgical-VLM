"""Apply step/instrument mappings to canonicalize VLM and clip datasets.

Applies ``step_mapping.json`` and ``instrument_mapping.json`` to both:

  1. vlm_dataset/  — aggregate JSONL + one JSONL per video
                     (nested structure: record → vlm_annotations[])
  2. clip_dataset/ — one JSONL per clip inside per-video sub-folders
                     (flat structure: step_title / instruments / anatomy)

Outputs:
  vlm_dataset_mapped/      mapped individual files + aggregate
  clip_dataset_mapped/     mapped clip files (videos symlinked / .ref.txt)
  reports_vlm_mapped/      reports (summary + CSVs + unmapped values)
  reports_clip_mapped/     same style, sourced from clip JSONLs

Usage:
    python -m video_pipeline.tools.apply_mapping_all [--step-map ...] [--instrument-map ...]
"""

import json
import csv
import argparse
import os
import copy
from collections import Counter, defaultdict
from datetime import datetime

from ..common.io import load_jsonl, load_single_jsonl_record, write_jsonl, write_single_jsonl_record

MAPPING_EXTENSIONS = ('.json',)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# Mapping helpers
# ══════════════════════════════════════════════════════════════════════════════

def build_reverse_lookup(mapping):
    """
    Converts  { canonical: [variant, ...] }  →  { variant_lower: canonical }

    Rules applied to match both mapping scripts:
    - step_mapping.py  : keys are canonical, list items are variants (key NOT in list)
    - instrument_mapping.py : keys are canonical, list includes itself; OR key: []
                            means "unmapped individual" → maps to itself.
    In all cases the canonical key always maps to itself.
    """
    reverse = {}
    for canonical, variants in mapping.items():
        reverse[canonical.lower().strip()] = canonical
        for v in variants:
            v_key = v.lower().strip()
            if v_key and v_key not in reverse:
                reverse[v_key] = canonical
    return reverse


def map_value(raw, reverse, miss_set):
    key = raw.lower().strip()
    canonical = reverse.get(key)
    if canonical is not None:
        return canonical
    miss_set.add(raw)
    return raw


def dedup_preserving_order(values):
    seen = set()
    result = []
    for v in values:
        k = v.lower().strip()
        if k not in seen:
            seen.add(k)
            result.append(v)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# VLM dataset  — nested structure  (vlm_annotations[])
# ══════════════════════════════════════════════════════════════════════════════

def extract_annotations(record):
    annotations = record.get("vlm_annotations", [])
    if isinstance(annotations, str):
        try:
            annotations = json.loads(annotations)
        except json.JSONDecodeError:
            return []
    return annotations if isinstance(annotations, list) else []


def apply_mapping_to_vlm_record(record, step_rev, inst_rev, step_misses, inst_misses):
    rec = copy.deepcopy(record)
    annotations = extract_annotations(rec)
    mapped = []

    for step in annotations:
        raw_title = step.get("step_title", "").strip()
        if raw_title:
            step["step_title"] = map_value(raw_title, step_rev, step_misses)

        raw_insts = step.get("instruments", [])
        if isinstance(raw_insts, list):
            mapped_insts = [map_value(i.strip(), inst_rev, inst_misses)
                            for i in raw_insts if i.strip()]
            step["instruments"] = dedup_preserving_order(mapped_insts)

        mapped.append(step)

    if isinstance(record.get("vlm_annotations"), str):
        rec["vlm_annotations"] = json.dumps(mapped, ensure_ascii=False)
    else:
        rec["vlm_annotations"] = mapped

    return rec


# ══════════════════════════════════════════════════════════════════════════════
# Clip dataset  — flat structure
# ══════════════════════════════════════════════════════════════════════════════

def apply_mapping_to_clip_record(record, step_rev, inst_rev, step_misses, inst_misses):
    rec = copy.deepcopy(record)

    raw_title = rec.get("step_title", "").strip()
    if raw_title:
        rec["step_title"] = map_value(raw_title, step_rev, step_misses)

    raw_insts = rec.get("instruments", [])
    if isinstance(raw_insts, list):
        mapped_insts = [map_value(i.strip(), inst_rev, inst_misses)
                        for i in raw_insts if i.strip()]
        rec["instruments"] = dedup_preserving_order(mapped_insts)

    return rec


# ══════════════════════════════════════════════════════════════════════════════
# Report generation
# ══════════════════════════════════════════════════════════════════════════════

def normalise(value):
    return value.strip().lower()


def build_vlm_global_sets(records):
    step_ctr, inst_ctr, anat_ctr = Counter(), Counter(), Counter()

    for rec in records:
        local_steps, local_insts, local_anat = set(), set(), set()

        for step in extract_annotations(rec):
            t = step.get("step_title", "").strip()
            if t:
                local_steps.add(normalise(t))
            for inst in step.get("instruments", []):
                if inst.strip():
                    local_insts.add(normalise(inst))
            for anat in step.get("anatomy", []):
                if anat.strip():
                    local_anat.add(normalise(anat))

        step_ctr.update(local_steps)
        inst_ctr.update(local_insts)
        anat_ctr.update(local_anat)

    return step_ctr, inst_ctr, anat_ctr


def build_vlm_per_video_rows(records):
    rows = []
    for rec in records:
        step_titles, instruments, anatomy = [], [], []
        seen_t, seen_i, seen_a = set(), set(), set()

        for step in extract_annotations(rec):
            t = step.get("step_title", "").strip()
            if t and normalise(t) not in seen_t:
                step_titles.append(t)
                seen_t.add(normalise(t))
            for inst in step.get("instruments", []):
                inst = inst.strip()
                if inst and normalise(inst) not in seen_i:
                    instruments.append(inst)
                    seen_i.add(normalise(inst))
            for anat in step.get("anatomy", []):
                anat = anat.strip()
                if anat and normalise(anat) not in seen_a:
                    anatomy.append(anat)
                    seen_a.add(normalise(anat))

        rows.append({
            "video_id": rec.get("video_id", "unknown"),
            "video_title": rec.get("video_title", ""),
            "video_url": rec.get("video_url", ""),
            "status": rec.get("status", ""),
            "num_steps": len(extract_annotations(rec)),
            "step_titles": " | ".join(step_titles),
            "instruments": " | ".join(instruments),
            "anatomy": " | ".join(anatomy),
            "num_unique_steps": len(step_titles),
            "num_instruments": len(instruments),
            "num_anatomy": len(anatomy),
        })
    return rows


def build_clip_global_sets(clip_records):
    step_ctr, inst_ctr, anat_ctr = Counter(), Counter(), Counter()

    for rec in clip_records:
        t = rec.get("step_title", "").strip()
        if t:
            step_ctr[normalise(t)] += 1
        for inst in rec.get("instruments", []):
            if inst.strip():
                inst_ctr[normalise(inst)] += 1
        for anat in rec.get("anatomy", []):
            if anat.strip():
                anat_ctr[normalise(anat)] += 1

    return step_ctr, inst_ctr, anat_ctr


def build_clip_per_video_rows(clip_records):
    groups = defaultdict(list)
    for rec in clip_records:
        groups[rec.get("parent_video_id", "unknown")].append(rec)

    rows = []
    for video_id, clips in sorted(groups.items()):
        step_titles, instruments, anatomy = [], [], []
        seen_t, seen_i, seen_a = set(), set(), set()

        for clip in sorted(clips, key=lambda c: c.get("step_number", 0)):
            t = clip.get("step_title", "").strip()
            if t and normalise(t) not in seen_t:
                step_titles.append(t)
                seen_t.add(normalise(t))
            for inst in clip.get("instruments", []):
                inst = inst.strip()
                if inst and normalise(inst) not in seen_i:
                    instruments.append(inst)
                    seen_i.add(normalise(inst))
            for anat in clip.get("anatomy", []):
                anat = anat.strip()
                if anat and normalise(anat) not in seen_a:
                    anatomy.append(anat)
                    seen_a.add(normalise(anat))

        rows.append({
            "video_id": video_id,
            "video_title": clips[0].get("parent_video_title", ""),
            "video_url": clips[0].get("parent_video_url", ""),
            "num_clips": len(clips),
            "step_titles": " | ".join(step_titles),
            "instruments": " | ".join(instruments),
            "anatomy": " | ".join(anatomy),
            "num_unique_steps": len(step_titles),
            "num_instruments": len(instruments),
            "num_anatomy": len(anatomy),
        })
    return rows


# ── Writers ────────────────────────────────────────────────────────────────────

def write_summary_report(path, label, total_units, total_steps, step_ctr, inst_ctr, anat_ctr):
    lines = []
    lines.append("=" * 70)
    lines.append(f"  VLM SURGICAL DATASET — {label} SUMMARY REPORT")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append(f"\n  Total units processed  : {total_units}")
    lines.append(f"  Total surgical steps   : {total_steps}")
    lines.append(f"  Unique step titles     : {len(step_ctr)}")
    lines.append(f"  Unique instruments     : {len(inst_ctr)}")
    lines.append(f"  Unique anatomy terms   : {len(anat_ctr)}")

    for section_label, ctr, col_name in [
        ("UNIQUE STEP TITLES", step_ctr, "Step Title"),
        ("UNIQUE INSTRUMENTS", inst_ctr, "Instrument"),
        ("UNIQUE ANATOMY TERMS", anat_ctr, "Anatomy Term"),
    ]:
        lines.append("\n" + "─" * 70)
        lines.append(f"  {section_label}  ({len(ctr)} total)")
        lines.append("─" * 70)
        lines.append(f"  {'#':<5}  {col_name:<50}  {'Count':>6}")
        lines.append(f"  {'─'*5}  {'─'*50}  {'─'*6}")
        for rank, (val, count) in enumerate(ctr.most_common(), 1):
            lines.append(f"  {rank:<5}  {val.title():<50}  {count:>6}")

    lines.append("\n" + "=" * 70 + "\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  ✅ Summary report       → {os.path.abspath(path)}")


def write_per_video_csv(path, rows, fieldnames):
    if not rows:
        print("  ⚠️  No rows to write.")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  ✅ Per-video CSV        → {os.path.abspath(path)}")


def write_counter_csv(path, ctr, fieldnames, value_key):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, (val, count) in enumerate(ctr.most_common(), 1):
            writer.writerow({"index": idx, value_key: val, "number": count})
    print(f"  ✅ {value_key.title()} CSV          → {os.path.abspath(path)}")


def write_unmapped_report(path, step_misses, inst_misses):
    lines = ["=" * 70,
             "  UNMAPPED VALUES — values kept as-is (not found in mapping JSONs)",
             f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             "=" * 70]
    lines.append(f"\n  Step titles with no mapping ({len(step_misses)}):")
    for v in sorted(step_misses):
        lines.append(f"    • {v}")
    lines.append(f"\n  Instruments with no mapping ({len(inst_misses)}):")
    for v in sorted(inst_misses):
        lines.append(f"    • {v}")
    lines.append("\n" + "=" * 70 + "\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  ✅ Unmapped values log  → {os.path.abspath(path)}")


def generate_reports(report_dir, label, total_units, total_steps, step_ctr, inst_ctr,
                     anat_ctr, per_video_rows, per_video_fieldnames, step_misses, inst_misses):
    os.makedirs(report_dir, exist_ok=True)
    print(f"\n💾 Writing {label} reports → {os.path.abspath(report_dir)}/")

    write_summary_report(os.path.join(report_dir, "report_summary.txt"), label,
                         total_units, total_steps, step_ctr, inst_ctr, anat_ctr)
    write_per_video_csv(os.path.join(report_dir, "per_video_report.csv"),
                        per_video_rows, per_video_fieldnames)
    write_counter_csv(os.path.join(report_dir, "step_titles_report.csv"), step_ctr,
                      ["index", "step_title", "number"], "step_title")
    write_counter_csv(os.path.join(report_dir, "instruments_report.csv"), inst_ctr,
                      ["index", "instrument", "number"], "instrument")
    write_counter_csv(os.path.join(report_dir, "anatomy_report.csv"), anat_ctr,
                      ["index", "anatomy", "number"], "anatomy")
    write_unmapped_report(os.path.join(report_dir, "unmapped_values.txt"),
                          step_misses, inst_misses)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline A — VLM dataset
# ══════════════════════════════════════════════════════════════════════════════

def run_vlm_pipeline(vlm_dir, aggregate_in, vlm_out, aggregate_out, step_rev, inst_rev, report_dir):
    print("\n" + "═" * 60)
    print("  PIPELINE A — VLM DATASET")
    print("═" * 60)
    os.makedirs(vlm_out, exist_ok=True)

    step_misses, inst_misses = set(), set()
    step_hits = inst_hits = step_total = inst_total = 0

    individual_files = [f for f in os.listdir(vlm_dir)
                        if f.endswith(".jsonl") and not f.startswith("vlm_dataset_all")]
    print(f"\n📂 Found {len(individual_files)} individual video JSONL files in {vlm_dir}")

    mapped_individual = []

    for filename in individual_files:
        src_path = os.path.join(vlm_dir, filename)
        record = load_single_jsonl_record(src_path)
        if record is None:
            print(f"  ⚠️  Empty file skipped: {filename}")
            continue

        for step in extract_annotations(record):
            if step.get("step_title", "").strip():
                step_total += 1
                if step.get("step_title", "").strip().lower() in step_rev:
                    step_hits += 1
            for inst in step.get("instruments", []):
                if inst.strip():
                    inst_total += 1
                    if inst.strip().lower() in inst_rev:
                        inst_hits += 1

        mapped = apply_mapping_to_vlm_record(record, step_rev, inst_rev, step_misses, inst_misses)
        mapped_individual.append(mapped)
        write_single_jsonl_record(os.path.join(vlm_out, filename), mapped)

    print(f"  ✅ {len(mapped_individual)} individual files written to {vlm_out}/")

    agg_records = []
    if os.path.exists(aggregate_in):
        print(f"\n📂 Loading aggregate JSONL: {aggregate_in}")
        agg_records = load_jsonl(aggregate_in)
        print(f"   {len(agg_records)} records loaded.")
        mapped_agg = [apply_mapping_to_vlm_record(r, step_rev, inst_rev, step_misses, inst_misses)
                      for r in agg_records]
        write_jsonl(aggregate_out, mapped_agg)
        print(f"  ✅ Mapped aggregate written to {os.path.abspath(aggregate_out)}")
    else:
        print(f"\n⚠️  Aggregate file not found: {aggregate_in} — skipping.")
        mapped_agg = mapped_individual

    report_source = mapped_agg if agg_records else mapped_individual

    print(f"\n  Steps   — mapped: {step_hits:,} / unmapped: {len(step_misses):,} / total: {step_total:,}")
    print(f"  Instruments — mapped: {inst_hits:,} / unmapped: {len(inst_misses):,} / total: {inst_total:,}")

    print("\n📊 Building VLM report data...")
    step_ctr, inst_ctr, anat_ctr = build_vlm_global_sets(report_source)
    per_video_rows = build_vlm_per_video_rows(report_source)
    total_steps = sum(len(extract_annotations(r)) for r in report_source)

    generate_reports(report_dir, "VLM DATASET (MAPPED)", len(report_source), total_steps,
                     step_ctr, inst_ctr, anat_ctr, per_video_rows,
                     ["video_id", "video_title", "video_url", "status",
                      "num_steps", "num_unique_steps", "num_instruments", "num_anatomy",
                      "step_titles", "instruments", "anatomy"],
                     step_misses, inst_misses)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline B — Clip dataset
# ══════════════════════════════════════════════════════════════════════════════

def run_clip_pipeline(clip_dir, clip_out, step_rev, inst_rev, report_dir):
    print("\n" + "═" * 60)
    print("  PIPELINE B — CLIP DATASET")
    print("═" * 60)
    os.makedirs(clip_out, exist_ok=True)

    step_misses, inst_misses = set(), set()

    all_clip_records = []
    total_clips_written = 0
    total_video_folders = 0

    video_folders = sorted([d for d in os.listdir(clip_dir)
                            if os.path.isdir(os.path.join(clip_dir, d))])
    print(f"\n📂 Found {len(video_folders)} video sub-folders in {clip_dir}")

    for video_id in video_folders:
        src_video_dir = os.path.join(clip_dir, video_id)
        dst_video_dir = os.path.join(clip_out, video_id)
        os.makedirs(dst_video_dir, exist_ok=True)
        total_video_folders += 1

        clip_files = sorted([f for f in os.listdir(src_video_dir) if f.endswith(".jsonl")])

        for clip_filename in clip_files:
            record = load_single_jsonl_record(os.path.join(src_video_dir, clip_filename))
            if record is None:
                continue

            mapped = apply_mapping_to_clip_record(record, step_rev, inst_rev, step_misses, inst_misses)
            all_clip_records.append(mapped)
            write_single_jsonl_record(os.path.join(dst_video_dir, clip_filename), mapped)
            total_clips_written += 1

        # Symlink (or .ref.txt) video files so the mapped folder is self-contained
        video_files = [f for f in os.listdir(src_video_dir) if not f.endswith(".jsonl")]
        for vf in video_files:
            src_vf = os.path.abspath(os.path.join(src_video_dir, vf))
            dst_vf = os.path.join(dst_video_dir, vf)
            ref_path = dst_vf + ".ref.txt"
            if os.path.exists(dst_vf) or os.path.exists(ref_path):
                continue
            try:
                os.symlink(src_vf, dst_vf)
            except (OSError, NotImplementedError):
                with open(ref_path, "w", encoding="utf-8") as ref:
                    ref.write(f"Video file located at:\n{src_vf}\n")

    print(f"\n  ✅ {total_clips_written} clip JSONL files written across {total_video_folders} video folders")
    print(f"  ⚠️  Unmapped step titles : {len(step_misses)}")
    print(f"  ⚠️  Unmapped instruments : {len(inst_misses)}")

    print("\n📊 Building clip report data...")
    step_ctr, inst_ctr, anat_ctr = build_clip_global_sets(all_clip_records)
    per_video_rows = build_clip_per_video_rows(all_clip_records)

    generate_reports(report_dir, "CLIP DATASET (MAPPED)", len(all_clip_records),
                     len(all_clip_records), step_ctr, inst_ctr, anat_ctr, per_video_rows,
                     ["video_id", "video_title", "video_url",
                      "num_clips", "num_unique_steps", "num_instruments", "num_anatomy",
                      "step_titles", "instruments", "anatomy"],
                     step_misses, inst_misses)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Apply step/instrument mappings to vlm_dataset and clip_dataset.")
    parser.add_argument("--config", default=None,
                        help="Path to a config.yaml (default: bundled / $VIDEO_PIPELINE_CONFIG)")
    parser.add_argument("--vlm_dir", default=None, help="Folder with individual per-video JSONL files")
    parser.add_argument("--clip_dir", default=None, help="Folder with {video_id}/clip_XX.jsonl sub-folders")
    parser.add_argument("--step_map", default=None, help="Step mapping JSON (default: <workspace>/step_mapping.json)")
    parser.add_argument("--instrument_map", default=None, help="Instrument mapping JSON (default: <workspace>/instrument_mapping.json)")
    parser.add_argument("--aggregate_in", default=None, help="Aggregate JSONL produced by vlm_generator.py")
    parser.add_argument("--vlm_out", default=None, help="Destination for mapped individual video JSONLs")
    parser.add_argument("--clip_out", default=None, help="Destination for mapped clip sub-folders")
    parser.add_argument("--aggregate_out", default=None, help="Mapped aggregate JSONL")
    parser.add_argument("--vlm_report_dir", default=None, help="VLM mapped reports directory")
    parser.add_argument("--clip_report_dir", default=None, help="Clip mapped reports directory")
    parser.add_argument("--skip_vlm", action="store_true", help="Skip the VLM dataset pipeline")
    parser.add_argument("--skip_clip", action="store_true", help="Skip the clip dataset pipeline")
    args = parser.parse_args(argv)

    from ..config import build_settings, get_workspace_root
    settings = build_settings(config_path=args.config)
    workspace = get_workspace_root(args.config)

    vlm_dir = args.vlm_dir or settings.vlm_dir
    clip_dir = args.clip_dir or settings.clips_dir
    step_map = args.step_map or os.path.join(workspace, "step_mapping.json")
    inst_map = args.instrument_map or os.path.join(workspace, "instrument_mapping.json")
    aggregate_in = args.aggregate_in or os.path.join(vlm_dir, settings.vlm.aggregate_file)
    vlm_out = args.vlm_out or os.path.join(workspace, "vlm_dataset_mapped")
    clip_out = args.clip_out or os.path.join(workspace, "clip_dataset_mapped")
    aggregate_out = args.aggregate_out or os.path.join(workspace, "vlm_dataset_all_mapped.jsonl")
    vlm_report_dir = args.vlm_report_dir or os.path.join(workspace, "reports_vlm_mapped")
    clip_report_dir = args.clip_report_dir or os.path.join(workspace, "reports_clip_mapped")

    print("\n📂 Loading mappings...")
    if not os.path.exists(step_map):
        print(f"❌ Step mapping not found: {step_map}")
        return
    if not os.path.exists(inst_map):
        print(f"❌ Instrument mapping not found: {inst_map}")
        return

    step_mapping = load_json(step_map)
    inst_mapping = load_json(inst_map)

    step_rev = build_reverse_lookup(step_mapping)
    inst_rev = build_reverse_lookup(inst_mapping)

    print(f"   Step map     : {len(step_mapping)} canonical labels → {len(step_rev)} reverse entries")
    print(f"   Instrument map: {len(inst_mapping)} canonical labels → {len(inst_rev)} reverse entries")

    if not args.skip_vlm:
        if os.path.isdir(vlm_dir) or os.path.exists(aggregate_in):
            run_vlm_pipeline(vlm_dir, aggregate_in, vlm_out, aggregate_out,
                             step_rev, inst_rev, vlm_report_dir)
        else:
            print(f"\n⚠️  VLM dir not found ({vlm_dir}) and aggregate not found ({aggregate_in}) — skipping VLM pipeline.")

    if not args.skip_clip:
        if os.path.isdir(clip_dir):
            run_clip_pipeline(clip_dir, clip_out, step_rev, inst_rev, clip_report_dir)
        else:
            print(f"\n⚠️  Clip dir not found ({clip_dir}) — skipping clip pipeline.")

    print("\n🎉 All done.\n")


if __name__ == "__main__":
    main()
