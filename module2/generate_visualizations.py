#!/usr/bin/env python3
"""
generate_visualizations.py
Produces 5 comprehensive visualizations of the final lipstick recommendation dataset.
All plots saved as PNG and SVG to module2/output/.

Plots:
  1. Pairplot          — key feature relationships coloured by primary_group
  2. Shade band plot   — shade LAB ranges per sub_group (horizontal bands)
  3. Skin tone scatter — skin_L vs skin_a coloured by primary_group
  4. Class balance     — record count per sub_group (imbalance check)
  5. Texture triangle  — normal/oily/dry composition distribution
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
import os
# Get the absolute path to ensure we can find the file regardless of working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
INPUT_FILE  = os.path.join(ROOT_DIR, "output", "final_skin_tone_dataset.csv")
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "output")
DPI         = 300

sns.set_theme(style="whitegrid", context="talk")

# Consistent palette for primary groups
PRIMARY_PALETTE = {
    "Coral":  "#E07A5F",
    "Nude":   "#C9A98A",
    "Red":    "#9B2335",
    "Berry":  "#6B3057",
    "Mauve":  "#A78295",
    "Pink":   "#E8A0BF",
}


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
def load_dataset(filepath=None):
    """Load the final dataset and verify required columns exist."""
    if filepath is None:
        filepath = INPUT_FILE
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. "
            "Run the full pipeline first via Orchestrator_dataset.py"
        )
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows × {len(df.columns)} columns from {filepath}")

    required = [
        "skin_L", "skin_a", "skin_b",
        "normal_pct", "oily_pct", "dry_pct",
        "contrast_level", "primary_group", "sub_group",
        "shade_L_min", "shade_L_max",
        "shade_a_min", "shade_a_max",
        "shade_b_min", "shade_b_max",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df


# ─────────────────────────────────────────────
# PLOT 1 — PAIRPLOT
# ─────────────────────────────────────────────
def plot_pairplot(df, output_dir=OUTPUT_DIR):
    """
    Pairplot of skin LAB + texture features coloured by primary_group.
    Shows clustering, separation, and feature correlations simultaneously.
    """
    print("  Generating Plot 1 — Pairplot...")

    plot_cols = ["skin_L", "skin_a", "skin_b", "normal_pct", "oily_pct", "primary_group"]
    plot_df   = df[plot_cols].copy()

    # Build ordered palette so legend is consistent
    groups  = sorted(plot_df["primary_group"].unique())
    palette = {g: PRIMARY_PALETTE.get(g, "#888888") for g in groups}

    g = sns.pairplot(
        plot_df,
        hue="primary_group",
        palette=palette,
        diag_kind="kde",
        plot_kws={"alpha": 0.45, "s": 22, "edgecolor": "none"},
        diag_kws={"fill": True, "alpha": 0.4},
    )
    g.fig.suptitle(
        "Dataset Overview — Feature Pairplot by Primary Shade Group",
        y=1.02, fontsize=15, fontweight="bold"
    )
    g.fig.text(
        0.5, -0.01,
        "Validation: distinct colour clusters confirm rule engine separated skin profiles correctly.",
        ha="center", fontsize=10, style="italic"
    )

    _save(g.fig, output_dir, "plot1_pairplot")


# ─────────────────────────────────────────────
# PLOT 2 — SHADE BAND PLOT
# ─────────────────────────────────────────────
def plot_shade_bands(df, output_dir=OUTPUT_DIR):
    """
    Horizontal band plot showing shade_L_min → shade_L_max per sub_group.
    Sorted dark to light. Validates that ranges are distinct and non-degenerate.
    """
    print("  Generating Plot 2 — Shade band plot...")

    # Aggregate mean min/max per sub_group
    agg = (
        df.groupby(["sub_group", "primary_group"])
        .agg(L_min=("shade_L_min", "mean"), L_max=("shade_L_max", "mean"))
        .reset_index()
        .sort_values("L_min")
    )

    fig, ax = plt.subplots(figsize=(14, 8), dpi=DPI)

    for i, row in agg.iterrows():
        color = PRIMARY_PALETTE.get(row["primary_group"], "#888888")
        # Draw the band
        ax.barh(
            y=row["sub_group"],
            width=row["L_max"] - row["L_min"],
            left=row["L_min"],
            height=0.55,
            color=color,
            alpha=0.75,
            edgecolor="white",
            linewidth=1.2,
        )
        # Centre dot
        mid = (row["L_min"] + row["L_max"]) / 2
        ax.plot(mid, row["sub_group"], "o", color="white",
                markersize=6, zorder=5, markeredgecolor=color, markeredgewidth=1.5)
        # Labels
        ax.text(row["L_min"] - 0.8, row["sub_group"], f"{row['L_min']:.0f}",
                va="center", ha="right", fontsize=9, color="#444444")
        ax.text(row["L_max"] + 0.8, row["sub_group"], f"{row['L_max']:.0f}",
                va="center", ha="left", fontsize=9, color="#444444")

    # Legend
    handles = [mpatches.Patch(color=PRIMARY_PALETTE.get(g, "#888"), label=g)
               for g in sorted(agg["primary_group"].unique())]
    ax.legend(handles=handles, title="Primary Group", fontsize=10,
              title_fontsize=11, loc="lower right")

    ax.set_xlabel("L* (Lightness)", fontsize=13)
    ax.set_ylabel("Sub Group", fontsize=13)
    ax.set_title(
        "Compatible Shade L* Ranges by Sub Group\n"
        "Validation: bands should be distinct — overlapping bands indicate rule boundary issues",
        fontsize=14, pad=16
    )
    ax.set_xlim(0, 100)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    _save(fig, output_dir, "plot2_shade_bands")


# ─────────────────────────────────────────────
# PLOT 3 — SKIN TONE SCATTER
# ─────────────────────────────────────────────
def plot_skin_tone_scatter(df, output_dir=OUTPUT_DIR):
    """
    skin_L vs skin_a scatter coloured by primary_group.
    The most intuitive check that the right skin tones got the right colour groups.
    """
    print("  Generating Plot 3 — Skin tone scatter...")

    groups  = sorted(df["primary_group"].unique())
    palette = {g: PRIMARY_PALETTE.get(g, "#888888") for g in groups}

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=DPI)
    fig.suptitle(
        "Skin Tone Space — Primary Group Assignment",
        fontsize=15, fontweight="bold"
    )

    # Left: skin_L vs skin_a
    sns.scatterplot(
        data=df, x="skin_L", y="skin_a",
        hue="primary_group", palette=palette,
        alpha=0.5, s=28, edgecolor="none", ax=axes[0]
    )
    axes[0].set_title("Lightness (L*) vs Red-Green (a*)", fontsize=13)
    axes[0].set_xlabel("skin_L", fontsize=12)
    axes[0].set_ylabel("skin_a", fontsize=12)
    axes[0].legend(title="Primary Group", fontsize=9, title_fontsize=10)

    # Right: skin_L vs skin_b
    sns.scatterplot(
        data=df, x="skin_L", y="skin_b",
        hue="primary_group", palette=palette,
        alpha=0.5, s=28, edgecolor="none", ax=axes[1]
    )
    axes[1].set_title("Lightness (L*) vs Yellow-Blue (b*)", fontsize=13)
    axes[1].set_xlabel("skin_L", fontsize=12)
    axes[1].set_ylabel("skin_b", fontsize=12)
    axes[1].legend(title="Primary Group", fontsize=9, title_fontsize=10)

    fig.text(
        0.5, -0.01,
        "Validation: Cool groups (Berry, Mauve) should cluster at lower a*/b*. "
        "Warm groups (Coral, Nude) at higher b*.",
        ha="center", fontsize=10, style="italic"
    )

    plt.tight_layout()
    _save(fig, output_dir, "plot3_skin_tone_scatter")


# ─────────────────────────────────────────────
# PLOT 4 — CLASS BALANCE
# ─────────────────────────────────────────────
def plot_class_balance(df, output_dir=OUTPUT_DIR):
    """
    Horizontal bar chart of record count per sub_group.
    Critical imbalance check before KNN training.
    """
    print("  Generating Plot 4 — Class balance...")

    counts = df["sub_group"].value_counts().sort_values()
    target = len(df) // df["sub_group"].nunique()

    # Colour bars red if count < 70% of target
    colors = [
        "#E74C3C" if c < target * 0.70 else
        "#F39C12" if c < target * 0.90 else
        "#2ECC71"
        for c in counts.values
    ]

    fig, ax = plt.subplots(figsize=(13, 8), dpi=DPI)

    bars = ax.barh(counts.index, counts.values, color=colors,
                   edgecolor="white", linewidth=0.8)

    # Target line
    ax.axvline(target, color="#2C3E50", linestyle="--", linewidth=1.8,
               label=f"Target ({target} per class)")

    # Count labels
    for bar, val in zip(bars, counts.values):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=10)

    # Legend for colours
    legend_patches = [
        mpatches.Patch(color="#2ECC71", label="≥ 90% of target — good"),
        mpatches.Patch(color="#F39C12", label="70–90% of target — acceptable"),
        mpatches.Patch(color="#E74C3C", label="< 70% of target — rebalance needed"),
    ]
    ax.legend(handles=legend_patches + [
        plt.Line2D([0], [0], color="#2C3E50", linestyle="--", linewidth=1.8,
                   label=f"Target ({target} per class)")
    ], fontsize=9, loc="lower right")

    ax.set_xlabel("Record Count", fontsize=13)
    ax.set_ylabel("Sub Group", fontsize=13)
    ax.set_title(
        "Class Balance Check — Records per Sub Group\n"
        "Validation: all bars should be green before KNN training",
        fontsize=14, pad=16
    )
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    _save(fig, output_dir, "plot4_class_balance")

    # Print summary
    print(f"\n    Target per class : {target}")
    print(f"    Min count        : {counts.min()} ({counts.idxmin()})")
    print(f"    Max count        : {counts.max()} ({counts.idxmax()})")
    under = counts[counts < target * 0.70]
    if len(under) > 0:
        print(f"    WARNING UNDERREPRESENTED: {under.index.tolist()}")
    else:
        print("    PASS All classes within acceptable range")


# ─────────────────────────────────────────────
# PLOT 5 — TEXTURE COMPOSITION
# ─────────────────────────────────────────────
def plot_texture_composition(df, output_dir=OUTPUT_DIR):
    """
    Two-panel texture composition plot.
    Left: KDE of each texture percentage.
    Right: Stacked bar of dominant texture by primary_group.
    Validates that Dirichlet sampling produced realistic, non-collapsed distributions.
    """
    print("  Generating Plot 5 — Texture composition...")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=DPI)
    fig.suptitle("Skin Texture Composition Distribution", fontsize=15, fontweight="bold")

    # Left — KDE of each percentage
    tex_colors = {"normal_pct": "#2ECC71", "oily_pct": "#F39C12", "dry_pct": "#3498DB"}
    for col, color in tex_colors.items():
        label = col.replace("_pct", "").capitalize()
        sns.kdeplot(data=df, x=col, label=label, color=color,
                    fill=True, alpha=0.3, linewidth=2.2, ax=axes[0])

    axes[0].set_xlabel("Percentage (%)", fontsize=12)
    axes[0].set_ylabel("Density", fontsize=12)
    axes[0].set_xlim(0, 100)
    axes[0].set_title("KDE of Texture Percentages", fontsize=13)
    axes[0].legend(title="Texture", fontsize=10, title_fontsize=11)
    axes[0].text(
        0.98, 0.97,
        f"Mean normal: {df['normal_pct'].mean():.1f}%\n"
        f"Mean oily:   {df['oily_pct'].mean():.1f}%\n"
        f"Mean dry:    {df['dry_pct'].mean():.1f}%",
        transform=axes[0].transAxes,
        fontsize=9, va="top", ha="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )

    # Right — dominant texture stacked bar by primary_group
    df_copy = df.copy()
    df_copy["dominant"] = df_copy[["normal_pct", "oily_pct", "dry_pct"]].idxmax(axis=1)
    df_copy["dominant"] = df_copy["dominant"].str.replace("_pct", "")

    crosstab = pd.crosstab(df_copy["primary_group"], df_copy["dominant"])
    for col in ["normal", "oily", "dry"]:
        if col not in crosstab.columns:
            crosstab[col] = 0
    crosstab = crosstab[["normal", "oily", "dry"]]

    crosstab.plot(
        kind="bar", stacked=True,
        color=[tex_colors["normal_pct"], tex_colors["oily_pct"], tex_colors["dry_pct"]],
        ax=axes[1], edgecolor="white", linewidth=0.6, width=0.65
    )
    axes[1].set_xlabel("Primary Group", fontsize=12)
    axes[1].set_ylabel("Count", fontsize=12)
    axes[1].set_title("Dominant Texture by Primary Group", fontsize=13)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30, ha="right")
    axes[1].legend(title="Dominant Texture", fontsize=10, title_fontsize=11)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.text(
        0.5, -0.01,
        "Validation: KDE curves should not collapse to 0 or 100. "
        "Each primary group bar should show all three texture types.",
        ha="center", fontsize=10, style="italic"
    )

    plt.tight_layout()
    _save(fig, output_dir, "plot5_texture_composition")


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────
def _save(fig, output_dir, name):
    """Save figure as both PNG and SVG."""
    os.makedirs(output_dir, exist_ok=True)
    png = os.path.join(output_dir, f"{name}.png")
    svg = os.path.join(output_dir, f"{name}.svg")
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {png}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 70)
    print("DATASET VISUALIZATIONS — Lipstick Recommendation AI")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load
    print("\n[STEP 1] Loading dataset...")
    df = load_dataset()

    # Generate all 5 plots
    print("\n[STEP 2] Generating visualizations...")
    plot_pairplot(df)
    plot_shade_bands(df)
    plot_skin_tone_scatter(df)
    plot_class_balance(df)
    plot_texture_composition(df)

    # Summary
    print("\n" + "=" * 70)
    print("ALL VISUALIZATIONS COMPLETE")
    print(f"Saved to: {OUTPUT_DIR}/")
    print()
    print("  plot1_pairplot.png/svg          — feature relationships by shade group")
    print("  plot2_shade_bands.png/svg        — shade L* range bands per sub group")
    print("  plot3_skin_tone_scatter.png/svg  — skin tone space coloured by group")
    print("  plot4_class_balance.png/svg      — record count per sub group")
    print("  plot5_texture_composition.png/svg— texture distribution validation")
    print("=" * 70)


if __name__ == "__main__":
    main()