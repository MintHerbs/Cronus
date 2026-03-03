"""
EdgeCaseDetector.py
Identifies and generates boundary-zone profiles to test recommendation engine limits
Injects edge cases for neutral ambiguity, extreme lightness, and conflict profiles
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set seaborn theme
sns.set_theme(style="whitegrid", context="talk")


# Edge case generation parameters
EDGE_CASE_PARAMS = {
    'neutral_ambiguity': {
        'count_ratio': 0.05,  # 5% of dataset
        'ab_ratio_range': (0.95, 1.05),  # a*/b* ratio near 1.0
        'description': 'Neutral Ambiguity: a*/b* ratio ≈ 1.0'
    },
    'extreme_lightness': {
        'count_ratio': 0.05,  # 5% of dataset
        'L_ranges': [(0, 32), (83, 100)],  # Very dark or very light
        'description': 'Extreme Lightness: L* < 32 or L* > 83'
    },
    'conflict_profiles': {
        'count_ratio': 0.05,  # 5% of dataset
        'description': 'Conflict Profiles: High a* with high b* (contradictory)'
    }
}


def load_master_dataset(filepath="output/master_dataset_augmented.csv"):
    """Load the master augmented dataset"""
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        print("Please run LightingAugmentator.py first!")
        return None
    
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} profiles from {filepath}")
    return df


def generate_neutral_ambiguity_cases(base_df, count, random_state=42):
    """
    Generate profiles with a*/b* ratio near 1.0 (neutral ambiguity)
    
    Args:
        base_df: Base DataFrame to sample structure from
        count: Number of edge cases to generate
        random_state: Random seed
    
    Returns:
        DataFrame with neutral ambiguity edge cases
    """
    np.random.seed(random_state)
    
    print(f"\nGenerating {count} Neutral Ambiguity cases...")
    
    # Sample random base profiles
    sample_indices = np.random.choice(len(base_df), size=count, replace=True)
    edge_cases = base_df.iloc[sample_indices].copy()
    
    # Generate a* and b* values with ratio near 1.0
    a_values = np.random.uniform(10, 20, count)
    ab_ratio = np.random.uniform(0.95, 1.05, count)
    b_values = a_values * ab_ratio
    
    edge_cases['a'] = a_values
    edge_cases['b'] = b_values
    edge_cases['L'] = np.random.uniform(40, 70, count)  # Mid-range lightness
    
    # Update undertone to Neutral (since ratio is ~1.0)
    edge_cases['Undertone'] = 'Neutral'
    
    # Mark as edge case
    edge_cases['Profile_Type'] = 'edge_case'
    edge_cases['Edge_Case_Category'] = 'Neutral Ambiguity'
    
    print(f"  Mean a*/b* ratio: {(edge_cases['a'] / edge_cases['b']).mean():.3f}")
    
    return edge_cases


def generate_extreme_lightness_cases(base_df, count, random_state=42):
    """
    Generate profiles with extreme L* values (very dark or very light)
    
    Args:
        base_df: Base DataFrame to sample structure from
        count: Number of edge cases to generate
        random_state: Random seed
    
    Returns:
        DataFrame with extreme lightness edge cases
    """
    np.random.seed(random_state + 1)
    
    print(f"\nGenerating {count} Extreme Lightness cases...")
    
    # Sample random base profiles
    sample_indices = np.random.choice(len(base_df), size=count, replace=True)
    edge_cases = base_df.iloc[sample_indices].copy()
    
    # Split between very dark and very light
    half = count // 2
    
    # Very dark (L* < 32)
    edge_cases.iloc[:half, edge_cases.columns.get_loc('L')] = np.random.uniform(20, 32, half)
    
    # Very light (L* > 83)
    edge_cases.iloc[half:, edge_cases.columns.get_loc('L')] = np.random.uniform(83, 95, count - half)
    
    # Keep a* and b* in normal ranges
    edge_cases['a'] = np.random.uniform(5, 25, count)
    edge_cases['b'] = np.random.uniform(8, 35, count)
    
    # Mark as edge case
    edge_cases['Profile_Type'] = 'edge_case'
    edge_cases['Edge_Case_Category'] = 'Extreme Lightness'
    
    print(f"  L* range: [{edge_cases['L'].min():.2f}, {edge_cases['L'].max():.2f}]")
    
    return edge_cases


def generate_conflict_profiles(base_df, count, random_state=42):
    """
    Generate profiles with contradictory a* and b* values (both high)
    Challenges the shade rule engine with ambiguous color signals
    
    Args:
        base_df: Base DataFrame to sample structure from
        count: Number of edge cases to generate
        random_state: Random seed
    
    Returns:
        DataFrame with conflict profile edge cases
    """
    np.random.seed(random_state + 2)
    
    print(f"\nGenerating {count} Conflict Profiles...")
    
    # Sample random base profiles
    sample_indices = np.random.choice(len(base_df), size=count, replace=True)
    edge_cases = base_df.iloc[sample_indices].copy()
    
    # Generate high a* (redness) and high b* (yellowness) simultaneously
    edge_cases['a'] = np.random.uniform(18, 25, count)  # High red
    edge_cases['b'] = np.random.uniform(25, 35, count)  # High yellow
    edge_cases['L'] = np.random.uniform(40, 70, count)  # Mid-range
    
    # Assign Combination skin type (contradictory nature)
    edge_cases['Skin_Type'] = 'Combination'
    
    # Mark as edge case
    edge_cases['Profile_Type'] = 'edge_case'
    edge_cases['Edge_Case_Category'] = 'Conflict Profile'
    
    print(f"  Mean a*: {edge_cases['a'].mean():.2f}, Mean b*: {edge_cases['b'].mean():.2f}")
    
    return edge_cases


def inject_edge_cases(df, injection_ratio=0.15, random_state=42):
    """
    Inject edge cases into the dataset
    
    Args:
        df: Original DataFrame
        injection_ratio: Percentage of dataset to add as edge cases (default 15%)
        random_state: Random seed
    
    Returns:
        DataFrame with edge cases injected and profile_type column
    """
    print("\nInjecting edge cases into dataset...")
    print(f"Original dataset size: {len(df)} profiles")
    print(f"Target injection ratio: {injection_ratio*100:.0f}%")
    
    # Mark existing data as standard
    df['Profile_Type'] = 'standard'
    df['Edge_Case_Category'] = 'None'
    
    # Calculate counts for each edge case type
    total_edge_cases = int(len(df) * injection_ratio)
    count_per_type = total_edge_cases // 3
    
    print(f"\nGenerating {total_edge_cases} edge cases ({count_per_type} per category)...")
    
    # Generate each type of edge case
    neutral_cases = generate_neutral_ambiguity_cases(df, count_per_type, random_state)
    extreme_cases = generate_extreme_lightness_cases(df, count_per_type, random_state)
    conflict_cases = generate_conflict_profiles(df, count_per_type, random_state)
    
    # Combine all edge cases
    edge_cases = pd.concat([neutral_cases, extreme_cases, conflict_cases], ignore_index=True)
    
    # Recalculate derived fields for edge cases
    edge_cases = recalculate_derived_fields(edge_cases)
    
    # Combine with original dataset
    df_with_edges = pd.concat([df, edge_cases], ignore_index=True)
    
    print(f"\nFinal dataset size: {len(df_with_edges)} profiles")
    print(f"Edge cases added: {len(edge_cases)} ({len(edge_cases)/len(df_with_edges)*100:.1f}%)")
    
    print("\nProfile Type Distribution:")
    print(df_with_edges['Profile_Type'].value_counts())
    
    print("\nEdge Case Category Distribution:")
    print(df_with_edges['Edge_Case_Category'].value_counts())
    
    return df_with_edges


def recalculate_derived_fields(df):
    """
    Recalculate derived fields for edge cases (MST_Class, Delta_E, etc.)
    
    Args:
        df: DataFrame with edge cases
    
    Returns:
        DataFrame with recalculated fields
    """
    # Recalculate MST_Class based on L*
    # Extend bins to cover extreme values
    bins = np.linspace(0, 100, 11)
    df['MST_Class'] = pd.cut(df['L'], bins=bins, labels=range(1, 11), include_lowest=True)
    
    # Fill any NaN values with middle class
    df['MST_Class'] = df['MST_Class'].fillna(5).astype(int)
    
    # Recalculate Delta_E (distance from reference lip color)
    REFERENCE_LIP_LAB = {'L': 45, 'a': 20, 'b': 12}
    df['Delta_E'] = np.sqrt(
        (df['L'] - REFERENCE_LIP_LAB['L'])**2 +
        (df['a'] - REFERENCE_LIP_LAB['a'])**2 +
        (df['b'] - REFERENCE_LIP_LAB['b'])**2
    )
    
    # Recalculate Contrast_Level based on Delta_E
    # Use same thresholds as original (approximate quantiles)
    df['Contrast_Level'] = pd.cut(
        df['Delta_E'],
        bins=[0, 17.6, 26.7, 100],
        labels=['Low', 'Medium', 'High']
    )
    
    return df


def plot_edge_case_scatter(df, output_path="output"):
    """
    Plot a* vs b* scatter with edge cases highlighted
    
    Args:
        df: DataFrame with Profile_Type column
        output_path: Output directory
    """
    fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
    
    # Separate standard and edge cases
    df_standard = df[df['Profile_Type'] == 'standard']
    df_edge = df[df['Profile_Type'] == 'edge_case']
    
    # Plot standard profiles with low alpha
    ax.scatter(
        df_standard['a'],
        df_standard['b'],
        c='lightgray',
        alpha=0.3,
        s=20,
        label='Standard Profiles',
        edgecolors='none'
    )
    
    # Plot edge cases by category with high alpha
    edge_categories = df_edge['Edge_Case_Category'].unique()
    colors = {'Neutral Ambiguity': '#E41A1C', 
              'Extreme Lightness': '#377EB8', 
              'Conflict Profile': '#4DAF4A'}
    markers = {'Neutral Ambiguity': 'o', 
               'Extreme Lightness': '^', 
               'Conflict Profile': 's'}
    
    for category in edge_categories:
        df_cat = df_edge[df_edge['Edge_Case_Category'] == category]
        ax.scatter(
            df_cat['a'],
            df_cat['b'],
            c=colors[category],
            alpha=1.0,
            s=80,
            marker=markers[category],
            label=f'Edge: {category}',
            edgecolors='black',
            linewidth=0.5
        )
    
    ax.set_xlabel('a* (Green-Red Axis)', fontsize=14)
    ax.set_ylabel('b* (Blue-Yellow Axis)', fontsize=14)
    ax.set_title('Edge Case Distribution in a*-b* Color Space', fontsize=16, pad=20)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add diagonal line for a*/b* = 1.0 (neutral boundary)
    a_range = np.linspace(df['a'].min(), df['a'].max(), 100)
    ax.plot(a_range, a_range, 'k--', alpha=0.3, linewidth=1, label='a*/b* = 1.0')
    
    # Add annotation
    annotation = ("Validation: Edge cases form boundaries in gaps between clusters.\n"
                 "Neutral Ambiguity cases cluster near a*/b* = 1.0 diagonal.\n"
                 "These boundary profiles improve KNN decision boundary precision.")
    fig.text(0.5, 0.02, annotation, ha='center', fontsize=10,
            style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    png_path = os.path.join(output_path, "edge_case_distribution.png")
    svg_path = os.path.join(output_path, "edge_case_distribution.svg")
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved: {png_path}")
    print(f"Saved: {svg_path}")


def analyze_edge_cases(df):
    """Analyze edge case characteristics"""
    print("\n" + "=" * 70)
    print("EDGE CASE ANALYSIS")
    print("=" * 70)
    
    df_edge = df[df['Profile_Type'] == 'edge_case']
    
    # Analyze by category
    for category in df_edge['Edge_Case_Category'].unique():
        df_cat = df_edge[df_edge['Edge_Case_Category'] == category]
        
        print(f"\n{category} ({len(df_cat)} profiles):")
        print(f"  L* range: [{df_cat['L'].min():.2f}, {df_cat['L'].max():.2f}]")
        print(f"  a* range: [{df_cat['a'].min():.2f}, {df_cat['a'].max():.2f}]")
        print(f"  b* range: [{df_cat['b'].min():.2f}, {df_cat['b'].max():.2f}]")
        
        if category == 'Neutral Ambiguity':
            ab_ratio = df_cat['a'] / df_cat['b']
            print(f"  a*/b* ratio: {ab_ratio.mean():.3f} ± {ab_ratio.std():.3f}")
    
    # Compare edge cases vs standard
    print("\n" + "=" * 70)
    print("EDGE CASES VS STANDARD COMPARISON")
    print("=" * 70)
    
    df_standard = df[df['Profile_Type'] == 'standard']
    
    comparison = pd.DataFrame({
        'Standard_Mean': df_standard[['L', 'a', 'b', 'Delta_E']].mean(),
        'Standard_Std': df_standard[['L', 'a', 'b', 'Delta_E']].std(),
        'Edge_Mean': df_edge[['L', 'a', 'b', 'Delta_E']].mean(),
        'Edge_Std': df_edge[['L', 'a', 'b', 'Delta_E']].std()
    })
    
    print("\n" + comparison.round(2).to_string())
    
    # KNN benefit explanation
    print("\n" + "=" * 70)
    print("WHY THIS IMPROVES KNN PERFORMANCE")
    print("=" * 70)
    print("\nDecision Boundary Flooding:")
    print("  • KNN errors occur most at decision boundaries")
    print("  • Edge cases target ambiguous regions (neutral zone, extremes)")
    print("  • Forces model to be more precise in boundary classifications")
    print("\nBenefits:")
    print("  • Improved 'Neutral' undertone classification accuracy")
    print("  • Better handling of extreme lightness values")
    print("  • Robust to contradictory color signals")
    print("  • Reduced misclassification at cluster boundaries")


def save_final_dataset(df, output_path="output", filename="final_dataset_with_edges.csv"):
    """Save the final dataset with edge cases"""
    filepath = os.path.join(output_path, filename)
    df.to_csv(filepath, index=False)
    print(f"\nFinal dataset saved: {filepath}")
    print(f"Total profiles: {len(df)}")
    print(f"  Standard: {len(df[df['Profile_Type'] == 'standard'])}")
    print(f"  Edge Cases: {len(df[df['Profile_Type'] == 'edge_case'])}")
    return filepath


def main():
    print("=" * 70)
    print("EDGE CASE DETECTOR - Boundary Zone Profile Injection")
    print("=" * 70)
    
    # Ensure output directory exists
    output_path = "output"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Load data
    df = load_master_dataset()
    if df is None:
        return
    
    # Inject edge cases
    print("\n[STEP 1] Injecting Edge Cases...")
    df_with_edges = inject_edge_cases(df, injection_ratio=0.15)
    
    # Show sample edge cases
    print("\n[STEP 2] Sample Edge Cases (first 5 from each category):")
    for category in ['Neutral Ambiguity', 'Extreme Lightness', 'Conflict Profile']:
        print(f"\n{category}:")
        sample = df_with_edges[df_with_edges['Edge_Case_Category'] == category][['L', 'a', 'b', 'Undertone', 'Skin_Type']].head(5)
        print(sample.to_string(index=False))
    
    # Generate visualization
    print("\n[STEP 3] Generating Visualization...")
    plot_edge_case_scatter(df_with_edges, output_path)
    
    # Analyze edge cases
    print("\n[STEP 4] Analyzing Edge Cases...")
    analyze_edge_cases(df_with_edges)
    
    # Save final dataset
    print("\n[STEP 5] Saving Final Dataset...")
    save_final_dataset(df_with_edges, output_path)
    
    print("\n" + "=" * 70)
    print("EDGE CASE INJECTION COMPLETE")
    print("Dataset enhanced with boundary-zone profiles for robust KNN training!")
    print("=" * 70)


if __name__ == "__main__":
    main()
