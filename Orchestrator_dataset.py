"""
orchestrator.py
Master pipeline runner for the Lipstick Recommendation Dataset Generator.
Executes all module2 scripts sequentially, then produces a final dataset
in CSV + Parquet format and generates dataset health visualizations.
"""

import subprocess
import sys
import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MODULE_DIR   = Path("module2")
OUTPUT_DIR   = Path("output")
DATASET_DIR  = Path("Dataset")
GRAPHICS_DIR = Path("module2/output")
HEALTH_DIR   = None  # Will be set based on user choice
FINAL_DATASET = None  # Will be set based on user choice
GENERATE_GRAPHICS = None  # Will be set by user input

PIPELINE = [
    MODULE_DIR / "SkinToneGenerator.py",
    MODULE_DIR / "SkinTypesGenerator.py",
    MODULE_DIR / "ContrastCalculator.py",
    MODULE_DIR / "ShadeRule.py",
    MODULE_DIR / "LightingAugmentator.py",
    MODULE_DIR / "EdgeCaseDetector.py",
    MODULE_DIR / "DatasetAssembler.py",
]

sns.set_theme(style="whitegrid", context="talk")


# ─────────────────────────────────────────────
# PIPELINE RUNNER
# ─────────────────────────────────────────────
def run_pipeline():
    """Execute all modules sequentially and collect timing results."""
    results = []

    for script in PIPELINE:
        name = script.name
        bar  = "─" * 60
        print(f"\n{bar}")
        print(f"  RUNNING › {name}")
        print(bar)

        if not script.exists():
            print(f"  [ERROR] Script not found: {script}")
            print("  Aborting pipeline.")
            sys.exit(1)

        t0 = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                check=True,
                text=True,
                capture_output=False   # stream stdout live
            )
            elapsed = time.time() - t0
            status  = "✓ PASSED"
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - t0
            status  = "✗ FAILED"
            print(f"\n  [ERROR] {name} exited with code {e.returncode}")
            print("  Aborting pipeline.")
            sys.exit(1)

        results.append({"module": name, "status": status, "elapsed_s": round(elapsed, 2)})
        print(f"\n  {status}  ({elapsed:.2f}s)")

    return results


# ─────────────────────────────────────────────
# DATASET HEALTH REPORT
# ─────────────────────────────────────────────
def load_final_dataset():
    """Load the assembled dataset produced by DatasetAssembler."""
    if not FINAL_DATASET.exists():
        print(f"[ERROR] Final dataset not found at {FINAL_DATASET}")
        sys.exit(1)
    df = pd.read_csv(FINAL_DATASET)
    print(f"\nLoaded final dataset: {len(df)} rows × {len(df.columns)} columns")
    return df


# ── Health Plot 1 ─────────────────────────────
def plot_class_balance(df, out):
    """
    2-panel figure:
      Left  – Primary shade group distribution (bar)
      Right – Skin type distribution (bar)
    Validates that synthetic generation didn't over-represent any class.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=200)

    # Shade group
    shade_counts = df["Primary_Shade"].value_counts()
    sns.barplot(x=shade_counts.values, y=shade_counts.index,
                palette="Set2", ax=axes[0], orient="h")
    axes[0].set_title("Primary Shade Group Distribution", fontsize=14, pad=12)
    axes[0].set_xlabel("Count")
    axes[0].set_ylabel("Shade Group")
    for bar, val in zip(axes[0].patches, shade_counts.values):
        axes[0].text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                     str(val), va="center", fontsize=10)

    # Skin type
    type_counts = df["Skin_Type"].value_counts()
    sns.barplot(x=type_counts.values, y=type_counts.index,
                palette="Set3", ax=axes[1], orient="h")
    axes[1].set_title("Skin Type Distribution", fontsize=14, pad=12)
    axes[1].set_xlabel("Count")
    axes[1].set_ylabel("Skin Type")
    for bar, val in zip(axes[1].patches, type_counts.values):
        axes[1].text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                     str(val), va="center", fontsize=10)

    fig.suptitle("Class Balance Health Check", fontsize=16, y=1.01)
    _save(fig, out, "health_01_class_balance")


# ── Health Plot 2 ─────────────────────────────
def plot_lab_distributions(df, out):
    """
    3-panel KDE plot of L*, a*, b* coloured by Undertone.
    Validates that LAB sampling covers the full human skin gamut
    and that undertone clusters are where colour theory predicts.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=200)
    palette = {"Warm": "#FF6B35", "Cool": "#4ECDC4", "Neutral": "#95B46A"}
    channels = [("L", "L* (Lightness)"), ("a", "a* (Red-Green)"), ("b", "b* (Yellow-Blue)")]

    for ax, (col, label) in zip(axes, channels):
        for tone, grp in df.groupby("Undertone"):
            sns.kdeplot(grp[col], ax=ax, label=tone,
                        color=palette[tone], fill=True, alpha=0.35, linewidth=2)
        ax.set_title(f"{label} by Undertone", fontsize=13)
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.legend(title="Undertone", fontsize=9)

    fig.suptitle("LAB Channel Distributions (Gamut Health)", fontsize=16, y=1.02)
    plt.tight_layout()
    _save(fig, out, "health_02_lab_distributions")


# ── Health Plot 3 ─────────────────────────────
def plot_harmony_score_health(df, out):
    """
    2-panel figure:
      Left  – Harmony score distribution (histogram + KDE)
      Right – Harmony score boxplot by Primary Shade
    Validates that harmony scores are realistic (mostly > 0.6)
    and that shade groups don't all collapse to the same score.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=200)

    # Histogram + KDE
    sns.histplot(df["Harmony_Score"], bins=40, kde=True,
                 color="steelblue", edgecolor="white", ax=axes[0])
    axes[0].axvline(df["Harmony_Score"].mean(), color="red",
                    linestyle="--", linewidth=2,
                    label=f"Mean {df['Harmony_Score'].mean():.3f}")
    axes[0].axvline(0.7, color="orange", linestyle=":", linewidth=1.5,
                    label="Quality threshold (0.70)")
    axes[0].set_title("Harmony Score Distribution", fontsize=14, pad=12)
    axes[0].set_xlabel("Harmony Score")
    axes[0].legend(fontsize=10)

    # Boxplot by shade
    shade_order = df.groupby("Primary_Shade")["Harmony_Score"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, y="Primary_Shade", x="Harmony_Score",
                order=shade_order, palette="Set2", orient="h", ax=axes[1])
    axes[1].axvline(0.7, color="orange", linestyle=":", linewidth=1.5)
    axes[1].set_title("Harmony Score by Shade Group", fontsize=14, pad=12)
    axes[1].set_xlabel("Harmony Score")
    axes[1].set_ylabel("Primary Shade")

    fig.suptitle("Harmony Score Health Check", fontsize=16, y=1.01)
    _save(fig, out, "health_03_harmony_scores")


# ── Health Plot 4 ─────────────────────────────
def plot_mst_undertone_heatmap(df, out):
    """
    Heatmap of MST Class × Undertone (normalised row-wise).
    Validates that undertone assignment is spread across all MST classes
    and not clustering at the extremes — a symptom of a broken ratio formula.
    """
    fig, ax = plt.subplots(figsize=(14, 7), dpi=200)

    pivot = pd.crosstab(df["MST_Class"], df["Undertone"], normalize="index") * 100
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu",
                linewidths=0.5, cbar_kws={"label": "Row %"}, ax=ax)
    ax.set_title("MST Class × Undertone Distribution (%)\n"
                 "Validation: Each row should have all three undertones represented",
                 fontsize=14, pad=12)
    ax.set_xlabel("Undertone")
    ax.set_ylabel("Monk Skin Tone Class (1=Darkest, 10=Lightest)")

    plt.tight_layout()
    _save(fig, out, "health_04_mst_undertone_heatmap")


# ── Health Plot 5 ─────────────────────────────
def plot_lighting_delta(df, out):
    """
    Violin plot of b* grouped by Lighting_Context.
    Validates that the lighting augmentation shifted b* in the
    correct direction (warm → +b*, fluorescent → -b*).
    """
    if "Lighting_Context" not in df.columns:
        print("  [SKIP] Lighting_Context column not found — skipping plot 5.")
        return

    fig, ax = plt.subplots(figsize=(14, 7), dpi=200)
    order   = ["Outdoor Daylight", "Indoor Warm", "Office Fluorescent"]
    palette = {"Outdoor Daylight": "#FDB462",
               "Indoor Warm": "#FB8072",
               "Office Fluorescent": "#80B1D3"}

    sns.violinplot(data=df, x="Lighting_Context", y="b",
                   order=order, palette=palette, inner="box", ax=ax)

    # Expected direction arrows
    ax.annotate("Expected: higher b*\n(warm/yellow shift)",
                xy=(1, df[df["Lighting_Context"] == "Indoor Warm"]["b"].median()),
                xytext=(1.35, df["b"].max() * 0.9),
                arrowprops=dict(arrowstyle="->", color="black"),
                fontsize=10, ha="center")
    ax.annotate("Expected: lower b*\n(cool/blue shift)",
                xy=(2, df[df["Lighting_Context"] == "Office Fluorescent"]["b"].median()),
                xytext=(1.65, df["b"].min() * 1.2),
                arrowprops=dict(arrowstyle="->", color="black"),
                fontsize=10, ha="center")

    ax.set_title("b* Distribution by Lighting Context\n"
                 "Validation: Warm light → +b*, Fluorescent → -b*",
                 fontsize=14, pad=12)
    ax.set_xlabel("Lighting Context")
    ax.set_ylabel("b* (Yellow–Blue axis)")

    plt.tight_layout()
    _save(fig, out, "health_05_lighting_delta")


# ── Health Plot 6 ─────────────────────────────
def plot_edge_case_coverage(df, out):
    """
    2-panel figure:
      Left  – Profile type pie chart (Standard vs Edge Case)
      Right – a* vs b* scatter with edge cases highlighted in red
    Validates that edge cases occupy boundary zones, not random positions.
    """
    if "Profile_Type" not in df.columns:
        print("  [SKIP] Profile_Type column not found — skipping plot 6.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=200)

    # Pie chart
    counts = df["Profile_Type"].value_counts()
    axes[0].pie(counts.values, labels=counts.index,
                autopct="%1.1f%%", startangle=140,
                colors=["#AED6F1", "#E74C3C"],
                textprops={"fontsize": 12})
    axes[0].set_title("Standard vs Edge Case Split", fontsize=14, pad=12)

    # Scatter
    std  = df[df["Profile_Type"] == "standard"]
    edge = df[df["Profile_Type"] == "edge_case"]

    axes[1].scatter(std["a"],  std["b"],  c="lightgray", alpha=0.25, s=12,
                    label="Standard", edgecolors="none")
    axes[1].scatter(edge["a"], edge["b"], c="#E74C3C",   alpha=0.8,  s=30,
                    label="Edge Case", edgecolors="black", linewidths=0.4)

    # Neutral boundary diagonal
    ab_range = np.linspace(df["a"].min(), df["a"].max(), 100)
    axes[1].plot(ab_range, ab_range, "k--", alpha=0.3, linewidth=1,
                 label="a*/b* = 1 (neutral boundary)")

    axes[1].set_xlabel("a* (Red–Green)")
    axes[1].set_ylabel("b* (Yellow–Blue)")
    axes[1].set_title("Edge Case Positions in a*–b* Space\n"
                      "Validation: Red dots should cluster near boundaries",
                      fontsize=13, pad=12)
    axes[1].legend(fontsize=10)

    fig.suptitle("Edge Case Coverage Health Check", fontsize=16, y=1.01)
    _save(fig, out, "health_06_edge_case_coverage")


# ── Health Plot 7 ─────────────────────────────
def plot_null_and_dtype_report(df, out):
    """
    2-panel figure:
      Left  – Null count per column (should all be 0)
      Right – Data type summary bar chart
    Acts as a schema integrity check before ML training.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=200)

    # Null counts
    null_counts = df.isnull().sum().sort_values(ascending=False)
    colors = ["#E74C3C" if v > 0 else "#2ECC71" for v in null_counts.values]
    axes[0].barh(null_counts.index, null_counts.values, color=colors)
    axes[0].set_title("Null Values per Column\n(Green = 0 nulls — good)", fontsize=13, pad=12)
    axes[0].set_xlabel("Null Count")
    axes[0].axvline(0, color="black", linewidth=0.8)
    for i, v in enumerate(null_counts.values):
        axes[0].text(v + 0.1, i, str(v), va="center", fontsize=8)

    # Dtype summary
    dtype_summary = df.dtypes.astype(str).value_counts()
    sns.barplot(x=dtype_summary.values, y=dtype_summary.index,
                palette="pastel", ax=axes[1], orient="h")
    axes[1].set_title("Column Data Types Summary", fontsize=13, pad=12)
    axes[1].set_xlabel("Number of Columns")
    for bar, val in zip(axes[1].patches, dtype_summary.values):
        axes[1].text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                     str(val), va="center", fontsize=10)

    fig.suptitle("Schema Integrity Health Check", fontsize=16, y=1.01)
    _save(fig, out, "health_07_schema_integrity")


# ── Health Plot 8 ─────────────────────────────
def plot_correlation_matrix(df, out):
    """
    Full numeric correlation heatmap.
    Key validations:
      • L* ↔ MST_Class should be strongly negative (dark skin = low L*, high MST)
        or strongly positive depending on bin direction.
      • Harmony_Score ↔ Delta_E should be non-trivial.
      • a* and b* should not be perfectly correlated (would imply collinearity bug).
    """
    num_cols = df.select_dtypes(include="number").columns.tolist()
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(max(10, len(num_cols)), max(8, len(num_cols) - 1)), dpi=200)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)   # show lower triangle only

    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.4,
                cbar_kws={"label": "Pearson r"},
                mask=mask, ax=ax)
    ax.set_title("Numeric Feature Correlation Matrix\n"
                 "Validation: No perfect ±1.0 outside diagonal (except known pairs)",
                 fontsize=14, pad=12)

    plt.tight_layout()
    _save(fig, out, "health_08_correlation_matrix")


# ─────────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────────
def generate_summary_report(df, pipeline_results, out):
    """Write a plain-text + JSON health summary report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Compute quick stats
    total          = len(df)
    null_total     = int(df.isnull().sum().sum())
    n_features     = len(df.columns)
    std_count      = int((df["Profile_Type"] == "standard").sum()) if "Profile_Type" in df.columns else "N/A"
    edge_count     = int((df["Profile_Type"] == "edge_case").sum()) if "Profile_Type" in df.columns else "N/A"
    mean_harmony   = round(float(df["Harmony_Score"].mean()), 4) if "Harmony_Score" in df.columns else "N/A"
    high_harmony_pct = round(float((df["Harmony_Score"] > 0.7).mean() * 100), 2) if "Harmony_Score" in df.columns else "N/A"

    report = {
        "generated_at": now,
        "dataset": {
            "total_rows": total,
            "total_features": n_features,
            "null_values": null_total,
            "standard_profiles": std_count,
            "edge_case_profiles": edge_count,
        },
        "harmony": {
            "mean_score": mean_harmony,
            "pct_above_0_7": high_harmony_pct,
        },
        "pipeline": pipeline_results,
    }

    # JSON
    json_path = out / "health_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # Human-readable text
    txt_path = out / "health_report.txt"
    with open(txt_path, "w", encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("  DATASET HEALTH REPORT\n")
        f.write(f"  Generated: {now}\n")
        f.write("=" * 70 + "\n\n")

        f.write("DATASET SUMMARY\n")
        f.write(f"  Total rows      : {total}\n")
        f.write(f"  Total features  : {n_features}\n")
        f.write(f"  Null values     : {null_total}  {'✓ clean' if null_total == 0 else '✗ INVESTIGATE'}\n")
        f.write(f"  Standard rows   : {std_count}\n")
        f.write(f"  Edge case rows  : {edge_count}\n\n")

        f.write("HARMONY SCORE HEALTH\n")
        f.write(f"  Mean score      : {mean_harmony}\n")
        f.write(f"  % above 0.70    : {high_harmony_pct}%\n\n")

        f.write("PIPELINE TIMING\n")
        for r in pipeline_results:
            f.write(f"  {r['module']:<35} {r['status']}  ({r['elapsed_s']}s)\n")

        f.write("\nHEALTH PLOTS\n")
        for png in sorted(out.glob("health_*.png")):
            f.write(f"  {png.name}\n")

    print(f"\n  Report written → {txt_path}")
    print(f"  Report written → {json_path}")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _save(fig, out_dir, name):
    png = out_dir / f"{name}.png"
    svg = out_dir / f"{name}.svg"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {png.name}")


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    if GENERATE_GRAPHICS and HEALTH_DIR:
        HEALTH_DIR.mkdir(parents=True, exist_ok=True)


def prompt_user_for_graphics():
    """Ask user if they want to generate graphics"""
    global GENERATE_GRAPHICS, HEALTH_DIR, FINAL_DATASET
    
    print("\n" + "=" * 70)
    print("  GRAPHICS GENERATION OPTION")
    print("=" * 70)
    print("\nDo you want to generate health visualizations?")
    print("  Y - Generate graphics (saved to module2/output)")
    print("  N - Skip graphics (faster execution)")
    print()
    
    while True:
        choice = input("Enter your choice (Y/N): ").strip().upper()
        if choice in ['Y', 'N']:
            break
        print("Invalid input. Please enter Y or N.")
    
    GENERATE_GRAPHICS = (choice == 'Y')
    
    if GENERATE_GRAPHICS:
        HEALTH_DIR = GRAPHICS_DIR / "dataset_health"
        FINAL_DATASET = OUTPUT_DIR / "final_skin_tone_dataset.csv"
        print(f"\n✓ Graphics will be generated in: {GRAPHICS_DIR}")
        print(f"✓ Health plots will be saved to: {HEALTH_DIR}")
    else:
        FINAL_DATASET = DATASET_DIR / "final_skin_tone_dataset.csv"
        print(f"\n✓ Graphics generation skipped")
        print(f"✓ Dataset will be saved to: {DATASET_DIR}")
    
    print("=" * 70)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    banner = "═" * 70
    print(f"\n{banner}")
    print("  LIPSTICK RECOMMENDATION DATASET — ORCHESTRATOR")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(banner)

    # 0. Ask user about graphics
    prompt_user_for_graphics()

    # 1. Setup directories
    ensure_dirs()

    # 2. Run pipeline
    print("\n▶  PHASE 1 — PIPELINE EXECUTION")
    pipeline_results = run_pipeline()

    # 3. Load final dataset
    print(f"\n▶  PHASE 2 — LOADING FINAL DATASET")
    df = load_final_dataset()

    if GENERATE_GRAPHICS:
        # 4. Health plots
        print(f"\n▶  PHASE 3 — GENERATING HEALTH PLOTS → {HEALTH_DIR}")
        plots = [
            ("Class Balance",          plot_class_balance),
            ("LAB Distributions",      plot_lab_distributions),
            ("Harmony Score Health",   plot_harmony_score_health),
            ("MST × Undertone",        plot_mst_undertone_heatmap),
            ("Lighting Delta",         plot_lighting_delta),
            ("Edge Case Coverage",     plot_edge_case_coverage),
            ("Schema Integrity",       plot_null_and_dtype_report),
            ("Correlation Matrix",     plot_correlation_matrix),
        ]

        for label, fn in plots:
            print(f"\n  ── {label}")
            try:
                fn(df, HEALTH_DIR)
            except Exception as e:
                print(f"  [WARN] {label} failed: {e}")

        # 5. Summary report
        print(f"\n▶  PHASE 4 — WRITING SUMMARY REPORT")
        generate_summary_report(df, pipeline_results, HEALTH_DIR)
    else:
        print(f"\n▶  PHASE 3 — SKIPPING GRAPHICS GENERATION")
        print(f"  Graphics generation disabled by user")

    # 6. Copy final dataset to appropriate location
    if not GENERATE_GRAPHICS:
        print(f"\n▶  PHASE 4 — COPYING DATASET TO {DATASET_DIR}")
        import shutil
        source = OUTPUT_DIR / "final_skin_tone_dataset.csv"
        destination = FINAL_DATASET
        if source.exists():
            shutil.copy2(source, destination)
            print(f"  ✓ Dataset copied to: {destination}")
        else:
            print(f"  [ERROR] Source dataset not found: {source}")

    # 7. Final banner
    total_time = sum(r["elapsed_s"] for r in pipeline_results)
    print(f"\n{banner}")
    print("  ORCHESTRATION COMPLETE")
    print(f"  Total pipeline time : {total_time:.1f}s")
    print(f"  Final dataset       : {FINAL_DATASET}")
    
    if GENERATE_GRAPHICS:
        print(f"  Health plots        : {HEALTH_DIR}  ({len(list(HEALTH_DIR.glob('*.png')))} PNGs)")
        print(f"  Health report       : {HEALTH_DIR / 'health_report.txt'}")
    else:
        print(f"  Graphics            : Skipped (user choice)")
    
    print(banner + "\n")


if __name__ == "__main__":
    main()