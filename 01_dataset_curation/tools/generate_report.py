"""Generate an HTML analytics report for the final dataset.

Self-contained HTML (Chart.js via CDN) summarizing clips, SFT/GRPO records,
splits, steps, instruments, anatomy and text-length distributions.

Usage:
    python -m video_pipeline.tools.generate_report [--clips-dir DIR] [--dataset-dir DIR]
                                                   [--split-file FILE] [--report FILE]
"""

import os
import json
import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

from ..config import build_settings


# ─────────────────────────────────────────────
# 1. Data Loading
# ─────────────────────────────────────────────

def load_clip_metadata(clips_dir):
    """Load all clip JSONL metadata from the clips directory."""
    clips = []
    base = Path(clips_dir)
    for parent_dir in base.iterdir():
        if not parent_dir.is_dir():
            continue
        for f in parent_dir.glob("*.jsonl"):
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            clips.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
    return clips


def load_split_file(split_path):
    p = Path(split_path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    print(f"   ⚠️  Split file not found: {p.resolve()} — split sizes will be empty.")
    return {}


def load_grpo_records(output_dir):
    records = []
    for split in ("train", "val", "test"):
        p = Path(output_dir) / f"{split}_grpo.jsonl"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rec = json.loads(line)
                            rec["_split"] = split
                            records.append(rec)
                        except json.JSONDecodeError:
                            pass
    return records


def load_sft_records(output_dir):
    records = []
    for split in ("train", "val", "test"):
        p = Path(output_dir) / f"{split}_sft.jsonl"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rec = json.loads(line)
                            rec["_split"] = split
                            records.append(rec)
                        except json.JSONDecodeError:
                            pass
    return records


# ─────────────────────────────────────────────
# 2. Analysis
# ─────────────────────────────────────────────

def word_count(text):
    return len(re.findall(r"\S+", text)) if text else 0


def analyze(clips, grpo_records, sft_records, splits):
    stats = {}

    stats["total_clips"] = len(clips)

    instrument_counter = Counter()
    anatomy_counter = Counter()
    step_counter = Counter()
    desc_wc, reasoning_wc = [], []
    clips_per_parent = Counter()

    for c in clips:
        for inst in c.get("instruments", []):
            if inst.strip():
                instrument_counter[inst.strip()] += 1
        for anat in c.get("anatomy", []):
            if anat.strip():
                anatomy_counter[anat.strip()] += 1
        step = c.get("step_title", "").strip()
        if step:
            step_counter[step] += 1
        desc_wc.append(word_count(c.get("visual_description", "")))
        clips_per_parent[c.get("parent_video_id", "unknown")] += 1

    stats["instruments"] = {k: v for k, v in instrument_counter.most_common() if v >= 3}
    stats["steps"] = {k: v for k, v in step_counter.most_common() if v >= 3}
    stats["anatomy"] = {k: v for k, v in anatomy_counter.most_common() if v >= 3}
    stats["desc_word_counts"] = desc_wc

    if splits:
        stats["split_sizes"] = {k: len(v) for k, v in splits.items()}
    else:
        stats["split_sizes"] = {"train": 0, "val": 0, "test": 0}

    qt_counter = Counter()
    rt_counter = Counter()
    reasoning_wc_by_qt = defaultdict(list)

    for rec in grpo_records:
        qt = rec.get("question_type", "unknown")
        rt = rec.get("reward_type", "unknown")
        qt_counter[qt] += 1
        rt_counter[rt] += 1
        rw = word_count(rec.get("reference_reasoning", ""))
        if rw > 0:
            reasoning_wc.append(rw)
            reasoning_wc_by_qt[qt].append(rw)

    stats["question_types"] = dict(qt_counter)
    stats["reward_types"] = dict(rt_counter)
    stats["reasoning_word_counts"] = reasoning_wc
    stats["reasoning_wc_by_qt"] = {k: round(sum(v) / len(v), 1) for k, v in reasoning_wc_by_qt.items()}

    grpo_split_counts = Counter(r["_split"] for r in grpo_records)
    sft_split_counts = Counter(r["_split"] for r in sft_records)
    stats["grpo_per_split"] = dict(grpo_split_counts)
    stats["sft_per_split"] = dict(sft_split_counts)
    stats["total_grpo"] = len(grpo_records)
    stats["total_sft"] = len(sft_records)

    cpv = list(clips_per_parent.values())
    stats["clips_per_video_dist"] = cpv
    stats["avg_clips_per_video"] = round(sum(cpv) / len(cpv), 1) if cpv else 0

    def to_histogram(counts, bins=20):
        if not counts:
            return [], []
        lo, hi = min(counts), max(counts)
        step = max(1, (hi - lo) // bins)
        edges = list(range(lo, hi + step, step))
        buckets = [0] * (len(edges) - 1)
        for v in counts:
            for i in range(len(edges) - 1):
                if edges[i] <= v < edges[i + 1]:
                    buckets[i] += 1
                    break
            else:
                buckets[-1] += 1
        labels = [f"{edges[i]}–{edges[i+1]-1}" for i in range(len(edges) - 1)]
        return labels, buckets

    stats["desc_wc_hist"] = to_histogram(desc_wc)
    stats["reasoning_wc_hist"] = to_histogram(reasoning_wc)
    stats["cpv_hist"] = to_histogram(cpv, bins=15)

    return stats


# ─────────────────────────────────────────────
# 3. HTML Report
# ─────────────────────────────────────────────

PALETTE = {
    "blue": "#378ADD", "teal": "#1D9E75", "amber": "#EF9F27",
    "coral": "#D85A30", "purple": "#7F77DD", "green": "#639922",
    "pink": "#D4537E", "gray": "#888780",
}


def js_list(items):
    return json.dumps(list(items))


def build_html(stats):
    def sorted_kv(d, top=30):
        pairs = sorted(d.items(), key=lambda x: -x[1])[:top]
        return [p[0] for p in pairs], [p[1] for p in pairs]

    inst_labels, inst_vals = sorted_kv(stats["instruments"])
    step_labels, step_vals = sorted_kv(stats["steps"])
    anat_labels, anat_vals = sorted_kv(stats["anatomy"], top=20)

    split_labels = ["train", "val", "test"]
    split_vals = [stats["split_sizes"].get(s, 0) for s in split_labels]

    sft_vals_split = [stats["sft_per_split"].get(s, 0) for s in split_labels]
    grpo_vals_split = [stats["grpo_per_split"].get(s, 0) for s in split_labels]

    qt_labels, qt_vals = sorted_kv(stats["question_types"])
    rt_labels, rt_vals = sorted_kv(stats["reward_types"])

    desc_hist_labels, desc_hist_vals = stats["desc_wc_hist"]
    reasoning_hist_labels, reasoning_hist_vals = stats["reasoning_wc_hist"]
    cpv_hist_labels, cpv_hist_vals = stats["cpv_hist"]

    avg_desc = round(sum(stats["desc_word_counts"]) / len(stats["desc_word_counts"]), 1) if stats["desc_word_counts"] else 0
    avg_reasoning = round(sum(stats["reasoning_word_counts"]) / len(stats["reasoning_word_counts"]), 1) if stats["reasoning_word_counts"] else 0

    inst_bar_height = max(300, len(inst_labels) * 32 + 80)
    step_bar_height = max(300, len(step_labels) * 32 + 80)
    anat_bar_height = max(300, len(anat_labels) * 32 + 80)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dataset Analytics Report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
  :root{{
    --bg:#f8f7f4;--surface:#ffffff;--border:#e2e0d8;
    --text:#1a1a18;--muted:#6b6a63;--accent:#378ADD;
    --radius:10px;--shadow:0 1px 3px rgba(0,0,0,.07);
  }}
  @media(prefers-color-scheme:dark){{
    :root{{--bg:#1c1c1a;--surface:#262623;--border:#3a3936;
      --text:#e8e6de;--muted:#9c9a92;--accent:#85B7EB}}
  }}
  body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
    background:var(--bg);color:var(--text);font-size:15px;line-height:1.6;padding:2rem 1rem}}
  .page{{max-width:1100px;margin:0 auto}}
  h1{{font-size:1.75rem;font-weight:600;margin-bottom:.25rem}}
  .subtitle{{color:var(--muted);font-size:.9rem;margin-bottom:2.5rem}}
  h2{{font-size:1.1rem;font-weight:600;margin-bottom:1rem;color:var(--text)}}
  .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:2.5rem}}
  .card{{background:var(--surface);border:0.5px solid var(--border);border-radius:var(--radius);
    padding:1rem 1.25rem}}
  .card .label{{font-size:12px;color:var(--muted);margin-bottom:.3rem}}
  .card .value{{font-size:1.6rem;font-weight:600;line-height:1}}
  .card .sub{{font-size:11px;color:var(--muted);margin-top:.3rem}}
  .section{{margin-bottom:3rem}}
  .chart-wrap{{background:var(--surface);border:0.5px solid var(--border);
    border-radius:var(--radius);padding:1.25rem;box-shadow:var(--shadow)}}
  .chart-title{{font-size:.85rem;font-weight:600;color:var(--muted);
    text-transform:uppercase;letter-spacing:.05em;margin-bottom:1rem}}
  .grid-2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:1.5rem}}
  .badge{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:500}}
  .note{{font-size:12px;color:var(--muted);margin-top:.5rem}}
  canvas{{display:block}}
</style>
</head>
<body>
<div class="page">

<h1>Dataset Analytics Report</h1>
<p class="subtitle">Cataract surgery VLM training dataset — generated automatically</p>

<!-- Summary Cards -->
<div class="cards">
  <div class="card"><div class="label">Total clips</div><div class="value">{stats["total_clips"]:,}</div></div>
  <div class="card"><div class="label">SFT records</div><div class="value">{stats["total_sft"]:,}</div></div>
  <div class="card"><div class="label">GRPO records</div><div class="value">{stats["total_grpo"]:,}</div></div>
  <div class="card"><div class="label">Unique instruments</div><div class="value">{len(stats["instruments"]):,}</div><div class="sub">with ≥3 appearances</div></div>
  <div class="card"><div class="label">Unique steps</div><div class="value">{len(stats["steps"]):,}</div><div class="sub">with ≥3 appearances</div></div>
  <div class="card"><div class="label">Avg clips / video</div><div class="value">{stats["avg_clips_per_video"]}</div></div>
  <div class="card"><div class="label">Avg description</div><div class="value">{avg_desc}</div><div class="sub">words per clip</div></div>
  <div class="card"><div class="label">Avg reasoning</div><div class="value">{avg_reasoning}</div><div class="sub">words per QA pair</div></div>
</div>

<!-- ── SECTION: Splits ── -->
<div class="section">
  <h2>Dataset splits</h2>
  <div class="grid-2">
    <div class="chart-wrap">
      <div class="chart-title">Parent videos per split</div>
      <div style="position:relative;height:240px"><canvas id="splitChart"
        role="img" aria-label="Bar chart of parent video counts per train/val/test split">
        Train: {split_vals[0]}, Val: {split_vals[1]}, Test: {split_vals[2]}
      </canvas></div>
    </div>
    <div class="chart-wrap">
      <div class="chart-title">SFT vs GRPO records per split</div>
      <div style="position:relative;height:240px"><canvas id="sftGrpoChart"
        role="img" aria-label="Grouped bar chart of SFT and GRPO record counts per split">
      </canvas></div>
    </div>
  </div>
</div>

<!-- ── SECTION: Surgical steps ── -->
<div class="section">
  <h2>Surgical steps <span class="badge" style="background:#e6f1fb;color:#185FA5">≥ 3 instances shown</span></h2>
  <div class="chart-wrap">
    <div class="chart-title">Step frequency</div>
    <div style="position:relative;height:{step_bar_height}px"><canvas id="stepsChart"
      role="img" aria-label="Horizontal bar chart of surgical step frequencies">
    </canvas></div>
  </div>
</div>

<!-- ── SECTION: Instruments ── -->
<div class="section">
  <h2>Instruments <span class="badge" style="background:#e1f5ee;color:#0F6E56">≥ 3 instances shown</span></h2>
  <div class="chart-wrap">
    <div class="chart-title">Instrument frequency</div>
    <div style="position:relative;height:{inst_bar_height}px"><canvas id="instChart"
      role="img" aria-label="Horizontal bar chart of instrument frequencies">
    </canvas></div>
  </div>
</div>

<!-- ── SECTION: Anatomy ── -->
<div class="section">
  <h2>Anatomical structures <span class="badge" style="background:#faeeda;color:#633806">≥ 3 instances shown</span></h2>
  <div class="chart-wrap">
    <div class="chart-title">Anatomy frequency</div>
    <div style="position:relative;height:{anat_bar_height}px"><canvas id="anatChart"
      role="img" aria-label="Horizontal bar chart of anatomical structure frequencies">
    </canvas></div>
  </div>
</div>

<!-- ── SECTION: Text length distributions ── -->
<div class="section">
  <h2>Text length distributions</h2>
  <div class="grid-2">
    <div class="chart-wrap">
      <div class="chart-title">Scene description — word count distribution</div>
      <p class="note" style="margin-bottom:.75rem">Avg {avg_desc} words &nbsp;·&nbsp; {len(stats["desc_word_counts"]):,} clips</p>
      <div style="position:relative;height:260px"><canvas id="descWcChart"
        role="img" aria-label="Histogram of word counts for scene descriptions">
      </canvas></div>
    </div>
    <div class="chart-wrap">
      <div class="chart-title">Reference reasoning — word count distribution</div>
      <p class="note" style="margin-bottom:.75rem">Avg {avg_reasoning} words &nbsp;·&nbsp; {len(stats["reasoning_word_counts"]):,} QA pairs</p>
      <div style="position:relative;height:260px"><canvas id="reasoningWcChart"
        role="img" aria-label="Histogram of word counts for reference reasoning texts">
      </canvas></div>
    </div>
  </div>
</div>

<!-- ── SECTION: QA pairs ── -->
<div class="section">
  <h2>QA pair analysis</h2>
  <div class="grid-2">
    <div class="chart-wrap">
      <div class="chart-title">Question type distribution</div>
      <div style="position:relative;height:280px"><canvas id="qtChart"
        role="img" aria-label="Doughnut chart of question types">
      </canvas></div>
    </div>
    <div class="chart-wrap">
      <div class="chart-title">Reward type distribution</div>
      <div style="position:relative;height:280px"><canvas id="rtChart"
        role="img" aria-label="Doughnut chart of reward types">
      </canvas></div>
    </div>
  </div>
  <div class="chart-wrap" style="margin-top:1.5rem">
    <div class="chart-title">Average reasoning length by question type (words)</div>
    <div style="position:relative;height:220px"><canvas id="reasLenChart"
      role="img" aria-label="Bar chart of average reasoning word count by question type">
    </canvas></div>
  </div>
</div>

<!-- ── SECTION: Clips per video ── -->
<div class="section">
  <h2>Clips per parent video</h2>
  <div class="chart-wrap">
    <div class="chart-title">Distribution of clip counts per source video</div>
    <p class="note" style="margin-bottom:.75rem">Avg {stats["avg_clips_per_video"]} clips / video</p>
    <div style="position:relative;height:260px"><canvas id="cpvChart"
      role="img" aria-label="Histogram of clips per parent video">
    </canvas></div>
  </div>
</div>

</div><!-- /page -->

<script>
const isDark = matchMedia('(prefers-color-scheme: dark)').matches;
const TEXT  = isDark ? '#9c9a92' : '#6b6a63';
const GRID  = isDark ? 'rgba(255,255,255,.06)' : 'rgba(0,0,0,.06)';

const BLUE   = '#378ADD';
const TEAL   = '#1D9E75';
const AMBER  = '#EF9F27';
const CORAL  = '#D85A30';
const PURPLE = '#7F77DD';
const GREEN  = '#639922';
const GRAY   = '#888780';

const baseOpts = {{
  responsive: true,
  maintainAspectRatio: false,
  plugins: {{ legend: {{ display: false }} }},
  scales: {{
    x: {{ ticks: {{ color: TEXT, font: {{ size: 11 }} }}, grid: {{ color: GRID }} }},
    y: {{ ticks: {{ color: TEXT, font: {{ size: 11 }} }}, grid: {{ color: GRID }} }},
  }}
}};

function hBar(id, labels, data, color, opts={{}}) {{
  const h = Math.max(300, labels.length * 32 + 80);
  new Chart(document.getElementById(id), {{
    type: 'bar',
    data: {{
      labels: labels,
      datasets: [{{ data: data, backgroundColor: color + 'cc',
        borderColor: color, borderWidth: 1, borderSkipped: false }}]
    }},
    options: {{
      indexAxis: 'y',
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{ ticks: {{ color: TEXT, font:{{size:11}} }}, grid: {{ color: GRID }} }},
        y: {{ ticks: {{ color: TEXT, font:{{size:11}}, autoSkip: false }} , grid:{{display:false}} }},
      }},
      ...opts
    }}
  }});
}}

function histBar(id, labels, data, color) {{
  new Chart(document.getElementById(id), {{
    type: 'bar',
    data: {{
      labels: labels,
      datasets: [{{ data: data, backgroundColor: color + 'aa',
        borderColor: color, borderWidth: 1, borderRadius: 3 }}]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{ ticks: {{ color: TEXT, font:{{size:10}}, maxRotation:45, autoSkip: false }}, grid:{{display:false}} }},
        y: {{ ticks: {{ color: TEXT, font:{{size:11}} }}, grid: {{ color: GRID }} }},
      }}
    }}
  }});
}}

// ── Splits ──
new Chart(document.getElementById('splitChart'), {{
  type: 'bar',
  data: {{
    labels: {js_list(split_labels)},
    datasets: [{{ data: {js_list(split_vals)},
      backgroundColor: [BLUE+'cc', TEAL+'cc', AMBER+'cc'],
      borderColor:     [BLUE,      TEAL,      AMBER],
      borderWidth: 1, borderRadius: 4 }}]
  }},
  options: {{ ...baseOpts }}
}});

new Chart(document.getElementById('sftGrpoChart'), {{
  type: 'bar',
  data: {{
    labels: {js_list(split_labels)},
    datasets: [
      {{ label: 'SFT',  data: {js_list(sft_vals_split)},
        backgroundColor: BLUE+'cc', borderColor: BLUE, borderWidth:1, borderRadius:3 }},
      {{ label: 'GRPO', data: {js_list(grpo_vals_split)},
        backgroundColor: CORAL+'cc', borderColor: CORAL, borderWidth:1, borderRadius:3 }},
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{
        display: true,
        labels: {{ color: TEXT, font:{{size:11}}, usePointStyle:true, pointStyle:'rect' }}
      }}
    }},
    scales: {{
      x: {{ ticks: {{ color: TEXT, font:{{size:11}} }}, grid:{{ color: GRID }} }},
      y: {{ ticks: {{ color: TEXT, font:{{size:11}} }}, grid:{{ color: GRID }} }},
    }}
  }}
}});

// ── Steps ──
hBar('stepsChart', {js_list(step_labels)}, {js_list(step_vals)}, TEAL);

// ── Instruments ──
hBar('instChart', {js_list(inst_labels)}, {js_list(inst_vals)}, BLUE);

// ── Anatomy ──
hBar('anatChart', {js_list(anat_labels)}, {js_list(anat_vals)}, AMBER);

// ── Word count histograms ──
histBar('descWcChart',      {js_list(desc_hist_labels)},      {js_list(desc_hist_vals)},      PURPLE);
histBar('reasoningWcChart', {js_list(reasoning_hist_labels)}, {js_list(reasoning_hist_vals)}, CORAL);
histBar('cpvChart',         {js_list(cpv_hist_labels)},       {js_list(cpv_hist_vals)},       TEAL);

// ── QA Doughnuts ──
const donutOpts = {{
  responsive: true, maintainAspectRatio: false, cutout: '62%',
  plugins: {{
    legend: {{
      display: true, position: 'bottom',
      labels: {{ color: TEXT, font:{{size:11}}, usePointStyle:true, pointStyle:'circle', padding:14 }}
    }}
  }}
}};

new Chart(document.getElementById('qtChart'), {{
  type: 'doughnut',
  data: {{
    labels: {js_list(qt_labels)},
    datasets: [{{ data: {js_list(qt_vals)},
      backgroundColor: [BLUE+'cc', TEAL+'cc', CORAL+'cc', PURPLE+'cc'],
      borderColor:     [BLUE,      TEAL,      CORAL,      PURPLE],
      borderWidth: 1 }}]
  }},
  options: donutOpts
}});

new Chart(document.getElementById('rtChart'), {{
  type: 'doughnut',
  data: {{
    labels: {js_list(rt_labels)},
    datasets: [{{ data: {js_list(rt_vals)},
      backgroundColor: [AMBER+'cc', GREEN+'cc', GRAY+'cc'],
      borderColor:     [AMBER,      GREEN,      GRAY],
      borderWidth: 1 }}]
  }},
  options: donutOpts
}});

// ── Reasoning length by question type ──
const reasQTs    = {js_list(list(stats["reasoning_wc_by_qt"].keys()))};
const reasAvgs   = {js_list(list(stats["reasoning_wc_by_qt"].values()))};
const reasColors = [BLUE+'cc', TEAL+'cc', CORAL+'cc', PURPLE+'cc'];
new Chart(document.getElementById('reasLenChart'), {{
  type: 'bar',
  data: {{
    labels: reasQTs,
    datasets: [{{ data: reasAvgs,
      backgroundColor: reasColors.slice(0, reasQTs.length),
      borderColor:     [BLUE, TEAL, CORAL, PURPLE].slice(0, reasQTs.length),
      borderWidth: 1, borderRadius: 4 }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: {{ color: TEXT, font:{{size:12}} }}, grid:{{display:false}} }},
      y: {{
        ticks: {{ color: TEXT, font:{{size:11}} }}, grid: {{ color: GRID }},
        title: {{ display: true, text: 'avg words', color: TEXT, font:{{size:11}} }}
      }}
    }}
  }}
}});
</script>
</body>
</html>
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate dataset analytics report.")
    parser.add_argument("--config", default=None,
                        help="Path to a config.yaml (default: bundled / $VIDEO_PIPELINE_CONFIG)")
    parser.add_argument("--clips-dir", default=None, help="Directory with per-clip JSONL metadata")
    parser.add_argument("--dataset-dir", default=None,
                        help="Directory containing train/val/test SFT and GRPO JSONL files")
    parser.add_argument("--split-file", default=None, help="Path to dataset_split.json")
    parser.add_argument("--report", default=None, help="Output HTML filename")
    args = parser.parse_args(argv)

    settings = build_settings(config_path=args.config)
    clips_dir = args.clips_dir or settings.clips_dir
    dataset_dir = args.dataset_dir or settings.formatter_output_dir
    split_file = args.split_file or os.path.join(settings.formatter_output_dir, settings.formatter_split_file)
    report_path = Path(args.report) if args.report else Path("dataset_report.html")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print("📂 Loading clip metadata...")
    clips = load_clip_metadata(clips_dir)
    print(f"   {len(clips)} clips found")

    print("📂 Loading split file...")
    splits = load_split_file(split_file)
    if splits:
        total_vids = sum(len(v) for v in splits.values())
        print(f"   {total_vids} parent videos across {len(splits)} splits")

    print("📂 Loading GRPO records...")
    grpo = load_grpo_records(dataset_dir)
    print(f"   {len(grpo)} GRPO records found")

    print("📂 Loading SFT records...")
    sft = load_sft_records(dataset_dir)
    print(f"   {len(sft)} SFT records found")

    if not grpo and not sft:
        print(f"\n⚠️  No SFT/GRPO records loaded. Check that --dataset-dir points to the folder")
        print(f"   containing train_sft.jsonl, train_grpo.jsonl, etc.")
        print(f"   Current value: {Path(dataset_dir).resolve()}\n")

    print("🔍 Analyzing...")
    stats = analyze(clips, grpo, sft, splits)

    print(f"📝 Writing report → {report_path.resolve()}")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(build_html(stats))

    print("\n✅ Done!")
    print(f"   Open: {report_path.resolve()}")


if __name__ == "__main__":
    main()
