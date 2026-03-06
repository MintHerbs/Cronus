"""
ContrastCalculator.py
Calculates perceptual contrast between skin tone and reference lip color using Delta E
"""

import numpy as np
import pandas as pd
import os


# Reference lip color in LAB space
REFERENCE_LIP_LAB = {
    'L': 45,
    'a': 20,
    'b': 12
}


def load_skin_profiles(filepath="output/skin_profiles_with_texture.csv"):
    """Load the skin profiles dataset"""
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        print("Please run SkinTextureGenerator.py first!")
        return None
    
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} skin profiles from {filepath}")
    return df


def calculate_delta_e(df, lip_lab=REFERENCE_LIP_LAB):
    """
    Calculate Delta E (Euclidean distance) between skin LAB and lip LAB
    
    Args:
        df: DataFrame with L, a, b columns
        lip_lab: Dictionary with reference lip L, a, b values
    
    Returns:
        DataFrame with added Delta_E column
    """
    print(f"\nCalculating Delta E using reference lip color:")
    print(f"  L*={lip_lab['L']}, a*={lip_lab['a']}, b*={lip_lab['b']}")
    
    # Calculate Euclidean distance in LAB space
    df['Delta_E'] = np.sqrt(
        (df['L'] - lip_lab['L'])**2 +
        (df['a'] - lip_lab['a'])**2 +
        (df['b'] - lip_lab['b'])**2
    )
    
    print(f"\nDelta E Statistics:")
    print(f"  Mean: {df['Delta_E'].mean():.2f}")
    print(f"  Std: {df['Delta_E'].std():.2f}")
    print(f"  Min: {df['Delta_E'].min():.2f}")
    print(f"  Max: {df['Delta_E'].max():.2f}")
    
    return df


def categorize_contrast(df, method='qcut'):
    """
    Categorize contrast into Low, Medium, High bins
    
    Args:
        df: DataFrame with Delta_E column
        method: 'qcut' for equal-sized groups or 'cut' for specific thresholds
    
    Returns:
        DataFrame with added Contrast_Level column and bin edges
    """
    if method == 'qcut':
        # Equal-sized groups (quantile-based)
        df['Contrast_Level'], bins = pd.qcut(
            df['Delta_E'],
            q=3,
            labels=['Low', 'Medium', 'High'],
            retbins=True
        )
        print(f"\nContrast categorization (quantile-based):")
        print(f"  Low: Delta E < {bins[1]:.2f}")
        print(f"  Medium: {bins[1]:.2f} <= Delta E < {bins[2]:.2f}")
        print(f"  High: Delta E >= {bins[2]:.2f}")
    else:
        # Fixed thresholds
        bins = [0, 15, 30, 100]
        df['Contrast_Level'] = pd.cut(
            df['Delta_E'],
            bins=bins,
            labels=['Low', 'Medium', 'High']
        )
        print(f"\nContrast categorization (fixed thresholds):")
        print(f"  Low: Delta E < 15")
        print(f"  Medium: 15 <= Delta E < 30")
        print(f"  High: Delta E >= 30")
    
    print(f"\nContrast Level Distribution:")
    print(df['Contrast_Level'].value_counts().sort_index())
    
    return df, bins


def analyze_contrast_by_mst(df):
    """Analyze relationship between MST class and contrast level"""
    print("\n" + "=" * 70)
    print("CONTRAST ANALYSIS BY MONK SKIN TONE CLASS")
    print("=" * 70)
    
    # Group by MST class and calculate mean Delta E
    mst_analysis = df.groupby('MST_Class')['Delta_E'].agg(['mean', 'std', 'min', 'max'])
    print("\nDelta E Statistics by MST Class:")
    print(mst_analysis.to_string())
    
    # Cross-tabulation of MST Class and Contrast Level
    print("\n" + "=" * 70)
    print("CONTRAST LEVEL DISTRIBUTION BY MST CLASS")
    print("=" * 70)
    crosstab = pd.crosstab(df['MST_Class'], df['Contrast_Level'], normalize='index') * 100
    print("\nPercentage distribution (row-wise):")
    print(crosstab.round(1).to_string())
    
    # Correlation analysis
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)
    correlation = df[['L', 'a', 'b', 'Delta_E']].corr()['Delta_E'].drop('Delta_E')
    print("\nCorrelation with Delta E:")
    print(f"  L* (Lightness): {correlation['L']:.3f}")
    print(f"  a* (Green-Red): {correlation['a']:.3f}")
    print(f"  b* (Blue-Yellow): {correlation['b']:.3f}")
    
    if abs(correlation['L']) > 0.7:
        print("\n[INSIGHT] L* (Lightness) is the dominant driver of contrast.")
        print("Consider normalizing L* if you want a* and b* to play a bigger role.")


def save_dataset(df, output_path="output", filename="skin_profiles_with_contrast.csv"):
    """Save the enriched dataset"""
    filepath = os.path.join(output_path, filename)
    df.to_csv(filepath, index=False)
    print(f"\nDataset saved: {filepath}")
    return filepath


def main():
    print("=" * 70)
    print("CONTRAST CALCULATOR - Delta E Analysis")
    print("=" * 70)
    
    # Ensure output directory exists
    output_path = "output"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Load data
    df = load_skin_profiles()
    if df is None:
        return
    
    # Calculate Delta E
    print("\n[STEP 1] Calculating Delta E...")
    df = calculate_delta_e(df)
    
    # Categorize contrast
    print("\n[STEP 2] Categorizing Contrast Levels...")
    df, bins = categorize_contrast(df, method='qcut')
    
    # Show sample
    print("\n[STEP 3] Sample Data (first 10 rows):")
    print(df[['L', 'a', 'b', 'MST_Class', 'Delta_E', 'Contrast_Level']].head(10).to_string(index=False))
    
    # Analyze relationships
    print("\n[STEP 4] Analyzing Relationships...")
    analyze_contrast_by_mst(df)
    
    # Save dataset
    print("\n[STEP 5] Saving Enriched Dataset...")
    save_dataset(df, output_path)
    
    print("\n" + "=" * 70)
    print("CONTRAST CALCULATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
