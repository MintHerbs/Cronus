"""
LightingAugmentator.py
Simulates three distinct lighting environments by augmenting LAB color space
Triples dataset size by creating Outdoor Daylight, Indoor Warm, and Office Fluorescent variants
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set seaborn theme
sns.set_theme(style="whitegrid", context="talk")


# Lighting environment parameters
LIGHTING_ENVIRONMENTS = {
    'Outdoor Daylight': {
        'L_shift': 0,
        'a_shift': 0,
        'b_shift': 0,
        'noise_sigma': 1
    },
    'Indoor Warm': {
        'L_shift': -3,
        'a_shift': 0,
        'b_shift': 5,
        'noise_sigma': 1
    },
    'Office Fluorescent': {
        'L_shift': 2,
        'a_shift': 0,
        'b_shift': -4,
        'noise_sigma': 1
    }
}

# LAB gamut constraints
LAB_CONSTRAINTS = {
    'L': (0, 100),
    'a': (-128, 127),
    'b': (-128, 127)
}


def load_final_dataset(filepath="output/final_product_recommendations.csv"):
    """Load the final product recommendations dataset"""
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        print("Please run ShadeRule.py first!")
        return None
    
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} profiles from {filepath}")
    return df


def apply_lighting_augmentation(df, random_state=42):
    """
    Apply lighting augmentation to create three variants per profile
    
    Args:
        df: Original DataFrame
        random_state: Random seed for reproducibility
    
    Returns:
        Augmented DataFrame with 3x rows and lighting_context column
    """
    np.random.seed(random_state)
    
    print("\nApplying lighting augmentation...")
    print(f"Original dataset size: {len(df)} profiles")
    
    # Add profile ID to track original profiles
    df['Profile_ID'] = range(len(df))
    
    augmented_dfs = []
    
    for lighting_name, params in LIGHTING_ENVIRONMENTS.items():
        print(f"\nGenerating {lighting_name} variant...")
        
        # Create copy of dataframe
        df_variant = df.copy()
        
        # Apply shifts with Gaussian noise
        noise_L = np.random.normal(0, params['noise_sigma'], len(df_variant))
        noise_a = np.random.normal(0, params['noise_sigma'], len(df_variant))
        noise_b = np.random.normal(0, params['noise_sigma'], len(df_variant))
        
        df_variant['L'] = df_variant['L'] + params['L_shift'] + noise_L
        df_variant['a'] = df_variant['a'] + params['a_shift'] + noise_a
        df_variant['b'] = df_variant['b'] + params['b_shift'] + noise_b
        
        # Clip to valid LAB gamut
        df_variant['L'] = np.clip(df_variant['L'], LAB_CONSTRAINTS['L'][0], LAB_CONSTRAINTS['L'][1])
        df_variant['a'] = np.clip(df_variant['a'], LAB_CONSTRAINTS['a'][0], LAB_CONSTRAINTS['a'][1])
        df_variant['b'] = np.clip(df_variant['b'], LAB_CONSTRAINTS['b'][0], LAB_CONSTRAINTS['b'][1])
        
        # Add lighting context column
        df_variant['Lighting_Context'] = lighting_name
        
        augmented_dfs.append(df_variant)
        
        print(f"  L* shift: {params['L_shift']:+d}, b* shift: {params['b_shift']:+d}")
        print(f"  Mean L*: {df_variant['L'].mean():.2f}, Mean b*: {df_variant['b'].mean():.2f}")
    
    # Combine all variants
    df_augmented = pd.concat(augmented_dfs, ignore_index=True)
    
    print(f"\nAugmented dataset size: {len(df_augmented)} profiles (3x original)")
    print(f"\nLighting Context Distribution:")
    print(df_augmented['Lighting_Context'].value_counts())
    
    return df_augmented


def plot_before_after_scatter(df, output_path="output", sample_size=50, random_state=42):
    """
    Plot L* vs b* scatter showing lighting shifts for sampled profiles
    
    Args:
        df: Augmented DataFrame with Profile_ID and Lighting_Context
        output_path: Output directory
        sample_size: Number of original profiles to sample
        random_state: Random seed
    """
    np.random.seed(random_state)
    
    # Sample random profile IDs
    unique_ids = df['Profile_ID'].unique()
    sampled_ids = np.random.choice(unique_ids, size=min(sample_size, len(unique_ids)), replace=False)
    
    # Filter to sampled profiles
    df_sample = df[df['Profile_ID'].isin(sampled_ids)].copy()
    
    print(f"\nPlotting before/after scatter for {len(sampled_ids)} sampled profiles...")
    
    fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
    
    # Define colors and markers for lighting contexts
    palette = {
        'Outdoor Daylight': '#FDB462',
        'Indoor Warm': '#FB8072',
        'Office Fluorescent': '#80B1D3'
    }
    
    markers = {
        'Outdoor Daylight': 'o',
        'Indoor Warm': 's',
        'Office Fluorescent': '^'
    }
    
    # Plot each lighting context
    for lighting in LIGHTING_ENVIRONMENTS.keys():
        df_lighting = df_sample[df_sample['Lighting_Context'] == lighting]
        ax.scatter(
            df_lighting['b'],
            df_lighting['L'],
            c=palette[lighting],
            marker=markers[lighting],
            s=80,
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5,
            label=lighting
        )
    
    # Draw lines connecting same profile across lighting contexts
    for profile_id in sampled_ids[:20]:  # Limit to 20 for clarity
        profile_data = df_sample[df_sample['Profile_ID'] == profile_id].sort_values('Lighting_Context')
        if len(profile_data) == 3:
            ax.plot(profile_data['b'], profile_data['L'], 
                   color='gray', alpha=0.2, linewidth=0.5, zorder=0)
    
    ax.set_xlabel('b* (Blue-Yellow Axis)', fontsize=14)
    ax.set_ylabel('L* (Lightness)', fontsize=14)
    ax.set_title(f'Lighting Augmentation Effect: L* vs b* ({len(sampled_ids)} Profiles)', 
                fontsize=16, pad=20)
    ax.legend(title='Lighting Context', fontsize=11, title_fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    annotation = ("Validation: Small clusters of 3 points per profile show lighting shift direction.\n"
                 "Warm light shifts toward +b* (yellow), Cool light shifts toward -b* (blue)")
    fig.text(0.5, 0.02, annotation, ha='center', fontsize=10,
            style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    png_path = os.path.join(output_path, "lighting_augmentation_scatter.png")
    svg_path = os.path.join(output_path, "lighting_augmentation_scatter.svg")
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")


def plot_lightness_violin(df, output_path="output"):
    """
    Plot violin plot of L* distribution by lighting context
    
    Args:
        df: Augmented DataFrame
        output_path: Output directory
    """
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    
    # Define color palette
    palette = {
        'Outdoor Daylight': '#FDB462',
        'Indoor Warm': '#FB8072',
        'Office Fluorescent': '#80B1D3'
    }
    
    # Create violin plot
    sns.violinplot(
        data=df,
        x='Lighting_Context',
        y='L',
        palette=palette,
        inner='box',
        ax=ax
    )
    
    ax.set_xlabel('Lighting Context', fontsize=14)
    ax.set_ylabel('L* (Lightness)', fontsize=14)
    ax.set_title('L* Distribution by Lighting Context', fontsize=16, pad=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add mean markers
    for i, lighting in enumerate(LIGHTING_ENVIRONMENTS.keys()):
        mean_L = df[df['Lighting_Context'] == lighting]['L'].mean()
        ax.plot(i, mean_L, 'D', color='red', markersize=8, zorder=10)
        ax.text(i, mean_L + 2, f'{mean_L:.1f}', ha='center', fontsize=10, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Add annotation
    annotation = ("Validation: Distribution 'breathes' based on light intensity.\n"
                 "Indoor Warm: Dimmer (L* down), Office Fluorescent: Brighter (L* up)")
    fig.text(0.5, 0.02, annotation, ha='center', fontsize=10,
            style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    png_path = os.path.join(output_path, "lighting_L_violin.png")
    svg_path = os.path.join(output_path, "lighting_L_violin.svg")
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")


def analyze_lighting_effects(df):
    """Analyze and display lighting augmentation statistics"""
    print("\n" + "=" * 70)
    print("LIGHTING AUGMENTATION ANALYSIS")
    print("=" * 70)
    
    # Statistics by lighting context
    print("\nLAB Statistics by Lighting Context:")
    stats = df.groupby('Lighting_Context')[['L', 'a', 'b']].agg(['mean', 'std'])
    print(stats.round(2).to_string())
    
    # Color temperature simulation validation
    print("\n" + "=" * 70)
    print("COLOR TEMPERATURE SIMULATION VALIDATION")
    print("=" * 70)
    
    for lighting in LIGHTING_ENVIRONMENTS.keys():
        df_lighting = df[df['Lighting_Context'] == lighting]
        mean_b = df_lighting['b'].mean()
        mean_L = df_lighting['L'].mean()
        
        print(f"\n{lighting}:")
        print(f"  Mean b* (Yellow-Blue): {mean_b:.2f}")
        print(f"  Mean L* (Lightness): {mean_L:.2f}")
        
        if 'Warm' in lighting:
            print(f"  → Simulates ~2700K (Warm/Yellow light)")
        elif 'Fluorescent' in lighting:
            print(f"  → Simulates ~6500K (Cool/Blue light)")
        else:
            print(f"  → Baseline (Natural daylight ~5500K)")
    
    # Gamut constraint validation
    print("\n" + "=" * 70)
    print("GAMUT CONSTRAINT VALIDATION")
    print("=" * 70)
    
    L_min, L_max = df['L'].min(), df['L'].max()
    a_min, a_max = df['a'].min(), df['a'].max()
    b_min, b_max = df['b'].min(), df['b'].max()
    
    print(f"\nL* range: [{L_min:.2f}, {L_max:.2f}] (valid: [0, 100])")
    print(f"a* range: [{a_min:.2f}, {a_max:.2f}] (valid: [-128, 127])")
    print(f"b* range: [{b_min:.2f}, {b_max:.2f}] (valid: [-128, 127])")
    
    if 0 <= L_min and L_max <= 100 and -128 <= a_min and a_max <= 127 and -128 <= b_min and b_max <= 127:
        print("\n✓ All values within valid LAB gamut")
    else:
        print("\n✗ WARNING: Some values outside valid LAB gamut")


def save_master_dataset(df, output_path="output", filename="master_dataset_augmented.csv"):
    """Save the master augmented dataset"""
    filepath = os.path.join(output_path, filename)
    df.to_csv(filepath, index=False)
    print(f"\nMaster dataset saved: {filepath}")
    print(f"Total profiles: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    return filepath


def main():
    print("=" * 70)
    print("LIGHTING AUGMENTATOR - Dataset Expansion via Lighting Simulation")
    print("=" * 70)
    
    # Ensure output directory exists
    output_path = "output"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Load data
    df = load_final_dataset()
    if df is None:
        return
    
    # Apply lighting augmentation
    print("\n[STEP 1] Applying Lighting Augmentation...")
    df_augmented = apply_lighting_augmentation(df)
    
    # Show sample
    print("\n[STEP 2] Sample Augmented Data (first 9 rows - 3 profiles x 3 lighting):")
    sample_cols = ['Profile_ID', 'L', 'a', 'b', 'Lighting_Context', 'Undertone', 'Primary_Shade']
    print(df_augmented[sample_cols].head(9).to_string(index=False))
    
    # Generate visualizations
    print("\n[STEP 3] Generating Visualizations...")
    plot_before_after_scatter(df_augmented, output_path)
    plot_lightness_violin(df_augmented, output_path)
    
    # Analyze effects
    print("\n[STEP 4] Analyzing Lighting Effects...")
    analyze_lighting_effects(df_augmented)
    
    # Save master dataset
    print("\n[STEP 5] Saving Master Dataset...")
    save_master_dataset(df_augmented, output_path)
    
    print("\n" + "=" * 70)
    print("LIGHTING AUGMENTATION COMPLETE")
    print("Dataset successfully tripled with realistic lighting variations!")
    print("=" * 70)
    
    # Final summary
    print("\n" + "=" * 70)
    print("IMPLEMENTATION INSIGHT")
    print("=" * 70)
    print("\nWhy shift b* for color temperature?")
    print("  • Warm Light (2700K): High b* (Yellow) - Indoor incandescent")
    print("  • Cool Light (6500K): Low b* (Blue) - Office fluorescent")
    print("  • Neutral (5500K): Baseline b* - Outdoor daylight")
    print("\nGaussian noise (σ=1) ensures:")
    print("  • Dataset variability and realism")
    print("  • ML models learn appearance as a range, not static point")
    print("  • Robustness to real-world lighting variations")


if __name__ == "__main__":
    main()
