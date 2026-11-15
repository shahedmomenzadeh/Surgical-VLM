"""
Generate publication-quality figures for Cataract Surgery VLM Benchmark.
Outputs:
  - 03_evaluation_framework/results/plots/fig1_overall_benchmark_comparison.png (.pdf)
  - 03_evaluation_framework/results/plots/fig2_subtask_radar_and_breakdown.png (.pdf)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

# Set high-DPI and publication styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

OUT_DIR = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(OUT_DIR, exist_ok=True)

# Data definition
models = [
    {
        "name": "Qwen3-VL-2B-Cataract-GRPO (Ours)",
        "short": "GRPO (Ours)",
        "family": "ours",
        "clip_acc": 0.5869,
        "mcq": 0.8909,
        "phase": 0.2825,
        "visual_desc": 0.3024,
        "narration": 0.2667,
        "narration_norm": 0.0533,
        "subtasks": [0.3024, 0.8457, 0.8765, 0.9506, 0.2506, 0.3981, 0.2908, 0.1905]
    },
    {
        "name": "Qwen3-VL-2B-Cataract-SFT-Stage2 (Ours)",
        "short": "SFT-Stage2 (Ours)",
        "family": "ours",
        "clip_acc": 0.5421,
        "mcq": 0.8292,
        "phase": 0.2010,
        "visual_desc": 0.3133,
        "narration": 0.3333,
        "narration_norm": 0.0667,
        "subtasks": [0.3133, 0.7346, 0.8580, 0.8951, 0.2488, 0.2345, 0.2097, 0.1111]
    },
    {
        "name": "Qwen3-VL-2B-Cataract-SFT (Ours)",
        "short": "SFT-Stage1 (Ours)",
        "family": "ours",
        "clip_acc": 0.5376,
        "mcq": 0.8354,
        "phase": 0.1925,
        "visual_desc": 0.2935,
        "narration": 0.3333,
        "narration_norm": 0.0667,
        "subtasks": [0.2935, 0.7778, 0.8395, 0.8889, 0.2457, 0.2237, 0.1894, 0.1111]
    },
    {
        "name": "Hulu-Med-4B",
        "short": "Hulu-Med-4B",
        "family": "specialist",
        "clip_acc": 0.4376,
        "mcq": 0.6502,
        "phase": 0.1953,
        "visual_desc": 0.2594,
        "narration": 0.2000,
        "narration_norm": 0.0400,
        "subtasks": [0.2594, 0.4630, 0.6358, 0.8519, 0.1061, 0.3182, 0.2300, 0.1270]
    },
    {
        "name": "Qwen3-VL-2B-Thinking",
        "short": "Qwen3-VL-2B-Think",
        "family": "generalist",
        "clip_acc": 0.4338,
        "mcq": 0.6235,
        "phase": 0.1754,
        "visual_desc": 0.3044,
        "narration": 0.1333,
        "narration_norm": 0.0267,
        "subtasks": [0.3044, 0.3519, 0.6852, 0.8333, 0.2012, 0.1928, 0.1489, 0.1587]
    },
    {
        "name": "Lingshu-7B",
        "short": "Lingshu-7B",
        "family": "specialist",
        "clip_acc": 0.4208,
        "mcq": 0.6090,
        "phase": 0.2157,
        "visual_desc": 0.2601,
        "narration": 0.8000,
        "narration_norm": 0.1600,
        "subtasks": [0.2601, 0.4444, 0.6481, 0.7346, 0.3170, 0.2566, 0.2097, 0.0793]
    },
    {
        "name": "Qwen3-VL-4B-Instruct",
        "short": "Qwen3-VL-4B-Inst",
        "family": "generalist",
        "clip_acc": 0.4059,
        "mcq": 0.5926,
        "phase": 0.0825,
        "visual_desc": 0.3276,
        "narration": 0.1333,
        "narration_norm": 0.0267,
        "subtasks": [0.3276, 0.3148, 0.6543, 0.8086, 0.0577, 0.0530, 0.1084, 0.1111]
    },
    {
        "name": "Qwen3-VL-2B-Instruct",
        "short": "Qwen3-VL-2B-Inst (Base)",
        "family": "base",
        "clip_acc": 0.4057,
        "mcq": 0.5905,
        "phase": 0.1820,
        "visual_desc": 0.2608,
        "narration": 0.0667,
        "narration_norm": 0.0133,
        "subtasks": [0.2608, 0.3272, 0.6358, 0.8086, 0.2063, 0.1372, 0.2097, 0.1746]
    },
    {
        "name": "Hulu-Med-7B",
        "short": "Hulu-Med-7B",
        "family": "specialist",
        "clip_acc": 0.3978,
        "mcq": 0.5453,
        "phase": 0.2171,
        "visual_desc": 0.2826,
        "narration": 0.6000,
        "narration_norm": 0.1200,
        "subtasks": [0.2826, 0.3333, 0.5494, 0.7531, 0.0859, 0.4618, 0.2097, 0.1111]
    }
]

subtask_names = [
    "Visual Desc\n(n=293)",
    "Step ID\n(n=162)",
    "Instrument ID\n(n=162)",
    "Visual Obs\n(n=162)",
    "Boundary Det\n(n=49)",
    "Temporal Loc\n(n=54)",
    "Time->Phase\n(n=47)",
    "Context Phase\n(n=60)"
]

radar_labels = [
    "Visual Description",
    "Step Identification",
    "Instrument Identification",
    "Visual Observation",
    "Boundary Detection",
    "Temporal Localization",
    "Timestamp -> Phase",
    "Contextual Phase"
]

def get_color(family):
    if family == "ours":
        return "#10B981"  # Emerald / Teal green
    elif family == "specialist":
        return "#F59E0B"  # Amber / Warm gold
    elif family == "base":
        return "#64748B"  # Slate grey
    else:
        return "#3B82F6"  # Blue

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: Overall Benchmark Comparison (3 Panels)
# ─────────────────────────────────────────────────────────────────────────────
def generate_figure1():
    fig = plt.figure(figsize=(15, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.2, 0.8])

    # Panel A: Overall Clip Accuracy
    ax1 = fig.add_subplot(gs[0])
    y_pos = np.arange(len(models))
    names = [m["short"] for m in models]
    accs = [m["clip_acc"] for m in models]
    colors = [get_color(m["family"]) for m in models]

    # Invert so top is best
    y_pos = y_pos[::-1]
    names = names[::-1]
    accs = accs[::-1]
    colors = colors[::-1]

    bars = ax1.barh(y_pos, accs, color=colors, height=0.68, edgecolor="#222222", linewidth=0.6)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=9.5, fontweight='medium')
    ax1.set_xlabel("Normalized Accuracy", fontsize=10.5, fontweight='bold', labelpad=6)
    ax1.set_title("(a) Overall Clip Accuracy (n=989 records)", fontsize=11.5, fontweight='bold', pad=10)
    ax1.set_xlim(0, 0.68)
    ax1.grid(axis='x', linestyle='--', alpha=0.5)

    for bar, val in zip(bars, accs):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2, f"{val:.4f}",
                 va='center', ha='left', fontsize=8.5, fontweight='bold', color='#1E293B')

    # Add legend to Panel A
    legend_elements = [
        patches.Patch(facecolor="#10B981", edgecolor="#222222", label="Ours (SFT / GRPO)"),
        patches.Patch(facecolor="#F59E0B", edgecolor="#222222", label="Medical Specialist"),
        patches.Patch(facecolor="#3B82F6", edgecolor="#222222", label="Generalist VLM"),
        patches.Patch(facecolor="#64748B", edgecolor="#222222", label="Base Model")
    ]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=8.5, framealpha=0.9)

    # Panel B: MCQ Macro vs Phase Macro
    ax2 = fig.add_subplot(gs[1])
    x = np.arange(len(models))
    width = 0.38
    mcqs = [m["mcq"] for m in models]
    phases = [m["phase"] for m in models]

    b1 = ax2.bar(x - width/2, mcqs, width, label='MCQ Macro (n=486)', color='#4F46E5', edgecolor='#1E1B4B', linewidth=0.6)
    b2 = ax2.bar(x + width/2, phases, width, label='Phase Macro (n=210)', color='#EC4899', edgecolor='#831843', linewidth=0.6)

    ax2.set_ylabel("Macro Score (0.0 - 1.0)", fontsize=10.5, fontweight='bold', labelpad=6)
    ax2.set_title("(b) Categorical Perception vs Temporal Grounding", fontsize=11.5, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels([m["short"] for m in models], rotation=42, ha='right', fontsize=8.5)
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', fontsize=8.5, framealpha=0.9)

    # Panel C: Full Video Narration
    ax3 = fig.add_subplot(gs[2])
    narr_scores = [m["narration"] for m in models]
    y_narr = np.arange(len(models))[::-1]
    bars_narr = ax3.barh(y_narr, narr_scores[::-1], color=colors, height=0.68, edgecolor="#222222", linewidth=0.6)
    ax3.set_yticks(y_narr)
    ax3.set_yticklabels(["" for _ in models])  # share labels with left
    ax3.set_xlabel("Clinical Score (/5.0)", fontsize=10.5, fontweight='bold', labelpad=6)
    ax3.set_title("(c) Full Narration (n=15)", fontsize=11.5, fontweight='bold', pad=10)
    ax3.set_xlim(0, 1.05)
    ax3.grid(axis='x', linestyle='--', alpha=0.5)

    for bar, val in zip(bars_narr, narr_scores[::-1]):
        ax3.text(val + 0.02, bar.get_y() + bar.get_height()/2, f"{val:.2f}",
                 va='center', ha='left', fontsize=8.5, fontweight='bold', color='#1E293B')

    plt.savefig(os.path.join(OUT_DIR, "fig1_overall_benchmark_comparison.png"), bbox_inches='tight')
    plt.savefig(os.path.join(OUT_DIR, "fig1_overall_benchmark_comparison.pdf"), bbox_inches='tight')
    plt.close()
    print("Saved fig1_overall_benchmark_comparison")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: Radar Chart & Granular Subtask Heatmap (2 Panels)
# ─────────────────────────────────────────────────────────────────────────────
def generate_figure2():
    fig = plt.figure(figsize=(16, 7), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.1])

    # Panel A: Radar Chart
    ax1 = fig.add_subplot(gs[0], polar=True)
    radar_models = [
        {"name": "GRPO (Ours)", "color": "#10B981", "lw": 2.5, "idx": 0},
        {"name": "SFT-Stage2 (Ours)", "color": "#06B6D4", "lw": 1.8, "idx": 1},
        {"name": "Qwen3-VL-2B Base", "color": "#64748B", "lw": 1.5, "idx": 7},
        {"name": "Lingshu-7B", "color": "#8B5CF6", "lw": 1.8, "idx": 5},
        {"name": "Hulu-Med-4B", "color": "#F59E0B", "lw": 1.5, "idx": 3}
    ]

    num_vars = len(radar_labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # close loop

    for rm in radar_models:
        vals = models[rm["idx"]]["subtasks"]
        vals_closed = vals + vals[:1]
        ax1.plot(angles, vals_closed, label=rm["name"], color=rm["color"], linewidth=rm["lw"])
        ax1.fill(angles, vals_closed, color=rm["color"], alpha=0.1)

    ax1.set_theta_offset(np.pi / 2)
    ax1.set_theta_direction(-1)
    ax1.set_thetagrids(np.degrees(angles[:-1]), radar_labels, fontsize=8.5, fontweight='medium')
    ax1.set_ylim(0, 1.0)
    ax1.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0], angle=45, fontsize=7.5, color='#666666')
    ax1.set_title("(a) Multi-Task Clinical Competency Radar Profile", fontsize=11.5, fontweight='bold', pad=15)
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=8.5, framealpha=0.9)

    # Panel B: Subtask Heatmap
    ax2 = fig.add_subplot(gs[1])
    matrix = np.array([m["subtasks"] for m in models])
    y_labels = [m["short"] for m in models]

    cmap = sns.light_palette("#0D9488", as_cmap=True)
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap=cmap, ax=ax2, cbar_kws={'label': 'Normalized Accuracy', 'shrink': 0.8},
                xticklabels=subtask_names, yticklabels=y_labels, linewidths=0.5, linecolor='#EEEEEE', annot_kws={'size': 8.5})

    ax2.set_title("(b) Granular Performance Across 8 Clinical Subtasks", fontsize=11.5, fontweight='bold', pad=10)
    ax2.set_xticklabels(subtask_names, rotation=35, ha='right', fontsize=8.5)
    ax2.set_yticklabels(y_labels, fontsize=8.5, rotation=0)

    plt.savefig(os.path.join(OUT_DIR, "fig2_subtask_radar_and_breakdown.png"), bbox_inches='tight')
    plt.savefig(os.path.join(OUT_DIR, "fig2_subtask_radar_and_breakdown.pdf"), bbox_inches='tight')
    plt.close()
    print("Saved fig2_subtask_radar_and_breakdown")

if __name__ == "__main__":
    generate_figure1()
    generate_figure2()
    print("All plots generated successfully in:", OUT_DIR)
