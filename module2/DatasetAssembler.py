"""
DatasetAssembler.py
Final integration script that merges all modules into production-ready dataset
Performs validation, constraint enforcement, metadata injection, and quality checks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set seaborn theme
sns.set_theme(style="whitegrid", context="talk")


# LAB gamut constraints
LAB_CONSTRAINTS = {
    'L': (0, 100),
    'a': (-128, 127),
    'b': (-128, 127)
}


def load_final_dataset(filepath="output/final_dataset_with_edges.csv"):
    """Load the final dataset with edge cases"""
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        print("Please run EdgeCaseDetector.py first!")
        return None
    
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} profiles from {filepath}")
    return df


def merge_and_deduplicate(df):
    """
    Merge and deduplicate dataset
    
    Args:
        df: Input DataFrame
    
    Returns:
        Deduplicated DataFrame
    """
    print("\nMerging and deduplicating dataset...")
    
    initial_count = len(df)
    
    # Drop exact duplicates
    df = df.drop_duplicates()
    
    duplicates_removed = initial_count - len(df)
    
    print(f"  Initial rows: {initial_count}")
    print(f"  Duplicates removed: {duplicates_removed}")
    print(f"  Final rows: {len(df)}")
    
    return df


def enforce_constraints(df):
    """
    Enforce LAB gamut constraints and data type requirements
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with enforced constraints
    """
    print("\nEnforcing constraints...")
    
    # Clip LAB values to valid gamut
    df['L'] = np.clip(df['L'], LAB_CONSTRAINTS['L'][0], LAB_CONSTRAINTS['L'][1])
    df['a'] = np.clip(df['a'], LAB_CONSTRAINTS['a'][0], LAB_CONSTRAINTS['a'][1])
    df['b'] = np.clip(df['b'], LAB_CONSTRAINTS['b'][0], LAB_CONSTRAINTS['b'][1])
    
    print(f"  L* clipped to [{LAB_CONSTRAINTS['L'][0]}, {LAB_CONSTRAINTS['L'][1]}]")
    print(f"  a* clipped to [{LAB_CONSTRAINTS['a'][0]}, {LAB_CONSTRAINTS['a'][1]}]")
    print(f"  b* clipped to [{LAB_CONSTRAINTS['b'][0]}, {LAB_CONSTRAINTS['b'][1]}]")
    
    # Ensure MST_Class is integer
    df['MST_Class'] = df['MST_Class'].astype(int)
    print(f"  MST_Class converted to integer")
    
    # Ensure Undertone is categorical
    df['Undertone'] = df['Undertone'].astype('category')
    print(f"  Undertone converted to categorical")
    
    # Ensure other categorical columns
    categorical_cols = ['Skin_Type', 'Texture_Descriptor', 'Contrast_Level', 
                       'Primary_Shade', 'Recommended_Finish', 'Lighting_Context',
                       'Profile_Type', 'Edge_Case_Category']
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    print(f"  All categorical columns converted")
    
    return df


def inject_metadata(df):
    """
    Inject metadata tags for generation method and rule confidence
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with metadata columns
    """
    print("\nInjecting metadata...")
    
    # Generation method based on profile type and lighting context
    def get_generation_method(row):
        if row['Profile_Type'] == 'edge_case':
            return 'synthetic_edge_case'
        elif row['Lighting_Context'] != 'Outdoor Daylight':
            return 'synthetic_augmented'
        else:
            return 'synthetic_standard'
    
    df['Generation_Method'] = df.apply(get_generation_method, axis=1)
    
    # Rule confidence based on generation method
    confidence_map = {
        'synthetic_standard': 1.0,
        'synthetic_augmented': 0.8,
        'synthetic_edge_case': 0.5
    }
    
    df['Rule_Confidence'] = df['Generation_Method'].map(confidence_map)
    
    print("\nGeneration Method Distribution:")
    print(df['Generation_Method'].value_counts())
    
    print("\nRule Confidence Distribution:")
    print(df['Rule_Confidence'].value_counts())
    
    return df


def shuffle_dataset(df, random_state=42):
    """
    Shuffle dataset to remove ordering bias
    
    Args:
        df: Input DataFrame
        random_state: Random seed
    
    Returns:
        Shuffled DataFrame
    """
    print("\nShuffling dataset to remove ordering bias...")
    
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"  Dataset shuffled with random_state={random_state}")
    
    return df_shuffled


def validate_dataset(df):
    """
    Validate dataset quality and completeness
    
    Args:
        df: Input DataFrame
    
    Returns:
        Validation report dictionary
    """
    print("\n" + "=" * 70)
    print("DATASET VALIDATION")
    print("=" * 70)
    
    validation_report = {}
    
    # Check for null values
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    
    print(f"\nNull Value Check:")
    print(f"  Total null values: {total_nulls}")
    
    if total_nulls > 0:
        print("\n  Columns with null values:")
        print(null_counts[null_counts > 0].to_string())
    else:
        print("  ✓ No null values found")
    
    validation_report['null_values'] = total_nulls
    
    # Check LAB constraints
    print(f"\nLAB Gamut Validation:")
    L_valid = (df['L'] >= LAB_CONSTRAINTS['L'][0]) & (df['L'] <= LAB_CONSTRAINTS['L'][1])
    a_valid = (df['a'] >= LAB_CONSTRAINTS['a'][0]) & (df['a'] <= LAB_CONSTRAINTS['a'][1])
    b_valid = (df['b'] >= LAB_CONSTRAINTS['b'][0]) & (df['b'] <= LAB_CONSTRAINTS['b'][1])
    
    print(f"  L* in valid range: {L_valid.sum()}/{len(df)} ({L_valid.sum()/len(df)*100:.1f}%)")
    print(f"  a* in valid range: {a_valid.sum()}/{len(df)} ({a_valid.sum()/len(df)*100:.1f}%)")
    print(f"  b* in valid range: {b_valid.sum()}/{len(df)} ({b_valid.sum()/len(df)*100:.1f}%)")
    
    validation_report['lab_valid'] = (L_valid & a_valid & b_valid).all()
    
    # Check data types
    print(f"\nData Type Validation:")
    print(f"  MST_Class is integer: {df['MST_Class'].dtype == 'int64' or df['MST_Class'].dtype == 'int32'}")
    print(f"  Undertone is categorical: {df['Undertone'].dtype.name == 'category'}")
    
    # Dataset statistics
    print(f"\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    print(f"\nTotal Profiles: {len(df)}")
    print(f"Total Features: {len(df.columns)}")
    print(f"\nProfile Breakdown:")
    print(f"  Standard: {len(df[df['Profile_Type'] == 'standard'])}")
    print(f"  Edge Cases: {len(df[df['Profile_Type'] == 'edge_case'])}")
    print(f"\nLighting Context Breakdown:")
    print(df['Lighting_Context'].value_counts().to_string())
    
    validation_report['total_profiles'] = len(df)
    validation_report['total_features'] = len(df.columns)
    
    return validation_report


def plot_correlation_heatmap(df, output_path="output"):
    """
    Generate correlation heatmap for numerical features
    
    Args:
        df: Input DataFrame
        output_path: Output directory
    """
    print("\nGenerating correlation heatmap...")
    
    # Select numerical columns
    numerical_cols = ['L', 'a', 'b', 'MST_Class', 'Delta_E', 'Harmony_Score']
    df_numerical = df[numerical_cols]
    
    # Calculate correlation matrix
    corr_matrix = df_numerical.corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.3f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Correlation Coefficient'},
        ax=ax
    )
    
    ax.set_title('Feature Correlation Heatmap', fontsize=16, pad=20)
    
    # Add annotation
    annotation = ("Validation: L* should correlate strongly with MST_Class.\n"
                 "Harmony_Score should have non-zero correlation with Contrast_Level.")
    fig.text(0.5, 0.02, annotation, ha='center', fontsize=10,
            style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    png_path = os.path.join(output_path, "correlation_heatmap.png")
    svg_path = os.path.join(output_path, "correlation_heatmap.svg")
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")
    
    # Print key correlations
    print("\nKey Correlations:")
    print(f"  L* vs MST_Class: {corr_matrix.loc['L', 'MST_Class']:.3f}")
    print(f"  Harmony_Score vs Delta_E: {corr_matrix.loc['Harmony_Score', 'Delta_E']:.3f}")


def plot_master_pairplot(df, output_path="output", sample_size=1000):
    """
    Generate master pairplot for LAB and harmony score
    
    Args:
        df: Input DataFrame
        output_path: Output directory
        sample_size: Number of samples to plot (for performance)
    """
    print(f"\nGenerating master pairplot (sampling {sample_size} profiles)...")
    
    # Sample for performance
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df
    
    # Select columns for pairplot
    pairplot_cols = ['L', 'a', 'b', 'Harmony_Score', 'Primary_Shade']
    df_pairplot = df_sample[pairplot_cols].copy()
    
    # Create pairplot
    pairplot = sns.pairplot(
        df_pairplot,
        hue='Primary_Shade',
        palette='Set2',
        diag_kind='kde',
        plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'none'},
        height=2.5
    )
    
    pairplot.fig.suptitle('Master Pairplot: LAB Color Space & Harmony Score by Shade Group', 
                         fontsize=16, y=1.02)
    
    # Add annotation
    pairplot.fig.text(0.5, -0.02, 
                     "Validation: Distinct colorful clusters show successful shade group partitioning.\n"
                     "Islands of color indicate rule engine successfully separates skin profiles.",
                     ha='center', fontsize=10, style='italic',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    png_path = os.path.join(output_path, "master_pairplot.png")
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {png_path}")
    print("\n[NOTE] Pairplot is the ultimate sanity check:")
    print("  • Distinct clusters = successful shade group separation")
    print("  • Overlapping clusters = rule engine may need rebalancing")
    print("  • Look for 'islands' of color in L* vs a* and a* vs b* plots")


def export_final_dataset(df, output_path="output"):
    """
    Export final dataset in multiple formats
    
    Args:
        df: Final DataFrame
        output_path: Output directory
    
    Returns:
        Dictionary of exported file paths
    """
    print("\n" + "=" * 70)
    print("EXPORTING FINAL DATASET")
    print("=" * 70)
    
    exported_files = {}
    
    # Export as CSV
    csv_path = os.path.join(output_path, "final_skin_tone_dataset.csv")
    df.to_csv(csv_path, index=False)
    exported_files['csv'] = csv_path
    print(f"\n✓ CSV exported: {csv_path}")
    print(f"  Size: {os.path.getsize(csv_path) / 1024 / 1024:.2f} MB")
    
    # Export as Parquet (more efficient for large datasets)
    parquet_path = os.path.join(output_path, "final_skin_tone_dataset.parquet")
    df.to_parquet(parquet_path, index=False, engine='pyarrow')
    exported_files['parquet'] = parquet_path
    print(f"\n✓ Parquet exported: {parquet_path}")
    print(f"  Size: {os.path.getsize(parquet_path) / 1024 / 1024:.2f} MB")
    
    # Export metadata summary
    metadata_path = os.path.join(output_path, "dataset_metadata.txt")
    with open(metadata_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("FINAL SKIN TONE DATASET - METADATA SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total Profiles: {len(df)}\n")
        f.write(f"Total Features: {len(df.columns)}\n")
        f.write(f"Generation Date: {pd.Timestamp.now()}\n\n")
        f.write("Profile Breakdown:\n")
        f.write(df['Profile_Type'].value_counts().to_string() + "\n\n")
        f.write("Generation Method:\n")
        f.write(df['Generation_Method'].value_counts().to_string() + "\n\n")
        f.write("Lighting Context:\n")
        f.write(df['Lighting_Context'].value_counts().to_string() + "\n\n")
        f.write("Column Names:\n")
        f.write("\n".join(df.columns.tolist()) + "\n\n")
        f.write("Data Types:\n")
        f.write(df.dtypes.to_string() + "\n")
    
    exported_files['metadata'] = metadata_path
    print(f"\n✓ Metadata exported: {metadata_path}")
    
    return exported_files


def main():
    print("=" * 70)
    print("DATASET ASSEMBLER - Final Integration & Production Export")
    print("=" * 70)
    
    # Ensure output directory exists
    output_path = "output"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Load data
    print("\n[STEP 1] Loading Dataset...")
    df = load_final_dataset()
    if df is None:
        return
    
    # Merge and deduplicate
    print("\n[STEP 2] Merging and Deduplicating...")
    df = merge_and_deduplicate(df)
    
    # Enforce constraints
    print("\n[STEP 3] Enforcing Constraints...")
    df = enforce_constraints(df)
    
    # Inject metadata
    print("\n[STEP 4] Injecting Metadata...")
    df = inject_metadata(df)
    
    # Shuffle dataset
    print("\n[STEP 5] Shuffling Dataset...")
    df = shuffle_dataset(df)
    
    # Validate dataset
    print("\n[STEP 6] Validating Dataset...")
    validation_report = validate_dataset(df)
    
    # Generate visualizations
    print("\n[STEP 7] Generating Visualizations...")
    plot_correlation_heatmap(df, output_path)
    plot_master_pairplot(df, output_path)
    
    # Export final dataset
    print("\n[STEP 8] Exporting Final Dataset...")
    exported_files = export_final_dataset(df, output_path)
    
    # Final summary
    print("\n" + "=" * 70)
    print("DATASET ASSEMBLY COMPLETE")
    print("=" * 70)
    print(f"\n✓ Production-ready dataset created")
    print(f"✓ Total profiles: {len(df)}")
    print(f"✓ Total features: {len(df.columns)}")
    print(f"✓ Null values: {validation_report['null_values']}")
    print(f"✓ LAB gamut valid: {validation_report['lab_valid']}")
    print(f"\nExported Files:")
    for format_type, filepath in exported_files.items():
        print(f"  • {format_type.upper()}: {filepath}")
    
    print("\n" + "=" * 70)
    print("READY FOR MACHINE LEARNING")
    print("=" * 70)
    print("\nDataset is now ready for:")
    print("  • KNN classification training")
    print("  • Shade recommendation modeling")
    print("  • Undertone prediction")
    print("  • Harmony score regression")
    print("  • Production deployment")


if __name__ == "__main__":
    main()
