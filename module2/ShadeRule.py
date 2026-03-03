"""
ShadeRule.py
Rule engine for mapping user attributes to product recommendations and harmony scores
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set seaborn theme
sns.set_theme(style="whitegrid", context="talk")


# Shade group mapping based on undertone
SHADE_GROUPS = {
    'Warm': ['Coral', 'Terracotta', 'Golden Nude'],
    'Cool': ['Berry', 'Mauve', 'Rose'],
    'Neutral': ['True Red', 'Dusty Rose', 'Taupe']
}

# Finish recommendations based on skin type
FINISH_RECOMMENDATIONS = {
    'Oily': ['Matte', 'Long-wear'],
    'Dry': ['Dewy', 'Cream'],
    'Normal': ['Satin', 'Natural'],
    'Combination': ['Satin', 'Natural'],
    'Sensitive': ['Cream', 'Natural']
}

# Intensity mapping for harmony score calculation
SHADE_INTENSITY = {
    'Berry': 'High',
    'Coral': 'Medium',
    'Terracotta': 'High',
    'Golden Nude': 'Low',
    'Mauve': 'Medium',
    'Rose': 'Medium',
    'True Red': 'High',
    'Dusty Rose': 'Low',
    'Taupe': 'Low'
}


def load_contrast_data(filepath="output/skin_profiles_with_contrast.csv"):
    """Load the skin profiles with contrast data"""
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        print("Please run ContrastCalculator.py first!")
        return None
    
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} skin profiles from {filepath}")
    return df


def assign_shade_groups(df):
    """
    Assign primary and sub-shade groups based on undertone and L* depth
    
    Args:
        df: DataFrame with Undertone and L columns
    
    Returns:
        DataFrame with added Primary_Shade and Sub_Shades columns
    """
    print("\nAssigning shade groups based on undertone...")
    
    # Assign primary shade based on undertone and L* depth
    def get_primary_shade(row):
        undertone = row['Undertone']
        L = row['L']
        shades = SHADE_GROUPS[undertone]
        
        # Select shade based on L* depth
        if undertone == 'Warm':
            if L < 50:
                return 'Terracotta'  # Darker warm tones
            elif L < 70:
                return 'Coral'  # Medium warm tones
            else:
                return 'Golden Nude'  # Lighter warm tones
        elif undertone == 'Cool':
            if L < 50:
                return 'Berry'  # Darker cool tones
            elif L < 70:
                return 'Mauve'  # Medium cool tones
            else:
                return 'Rose'  # Lighter cool tones
        else:  # Neutral
            if L < 50:
                return 'True Red'  # Darker neutral tones
            elif L < 70:
                return 'Dusty Rose'  # Medium neutral tones
            else:
                return 'Taupe'  # Lighter neutral tones
    
    df['Primary_Shade'] = df.apply(get_primary_shade, axis=1)
    
    # Assign sub-shades (all shades in the undertone group)
    df['Sub_Shades'] = df['Undertone'].map(lambda u: ', '.join(SHADE_GROUPS[u]))
    
    print("\nPrimary Shade Distribution:")
    print(df['Primary_Shade'].value_counts())
    
    return df


def assign_finish_recommendations(df):
    """
    Assign finish recommendations based on skin type
    
    Args:
        df: DataFrame with Skin_Type column
    
    Returns:
        DataFrame with added Recommended_Finish column
    """
    print("\nAssigning finish recommendations based on skin type...")
    
    # Select primary finish for each skin type
    def get_recommended_finish(skin_type):
        finishes = FINISH_RECOMMENDATIONS[skin_type]
        # Return primary recommendation (first in list)
        return finishes[0]
    
    df['Recommended_Finish'] = df['Skin_Type'].map(get_recommended_finish)
    
    print("\nRecommended Finish Distribution:")
    print(df['Recommended_Finish'].value_counts())
    
    return df


def calculate_harmony_score(df, random_state=42):
    """
    Calculate harmony score (0.0 to 1.0) based on contrast-shade intensity match
    Uses beta distribution for realistic skew toward high scores
    
    Args:
        df: DataFrame with Contrast_Level and Primary_Shade columns
        random_state: Random seed for reproducibility
    
    Returns:
        DataFrame with added Harmony_Score column
    """
    print("\nCalculating harmony scores...")
    np.random.seed(random_state)
    
    def get_base_harmony(row):
        contrast = row['Contrast_Level']
        shade = row['Primary_Shade']
        intensity = SHADE_INTENSITY[shade]
        
        # Base score logic: match contrast with shade intensity
        if contrast == 'High' and intensity == 'High':
            return 0.90  # Perfect match
        elif contrast == 'Low' and intensity == 'Low':
            return 0.85  # Good match
        elif contrast == 'Medium' and intensity == 'Medium':
            return 0.88  # Good match
        elif contrast == 'High' and intensity == 'Medium':
            return 0.75  # Acceptable
        elif contrast == 'Medium' and intensity == 'Low':
            return 0.70  # Acceptable
        elif contrast == 'Low' and intensity == 'Medium':
            return 0.72  # Acceptable
        else:
            return 0.60  # Suboptimal match
    
    # Calculate base harmony scores
    base_scores = df.apply(get_base_harmony, axis=1)
    
    # Add Gaussian noise for realism (μ=0, σ=0.05)
    noise = np.random.normal(0, 0.05, size=len(df))
    
    # Combine and clip to [0, 1]
    df['Harmony_Score'] = np.clip(base_scores + noise, 0.0, 1.0)
    
    print(f"\nHarmony Score Statistics:")
    print(f"  Mean: {df['Harmony_Score'].mean():.3f}")
    print(f"  Std: {df['Harmony_Score'].std():.3f}")
    print(f"  Min: {df['Harmony_Score'].min():.3f}")
    print(f"  Max: {df['Harmony_Score'].max():.3f}")
    
    return df


def plot_undertone_shade_heatmap(df, output_path="output"):
    """Plot heatmap of undertone vs primary shade group"""
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    
    # Create pivot table
    pivot = pd.crosstab(df['Undertone'], df['Primary_Shade'])
    
    # Plot heatmap
    sns.heatmap(
        pivot,
        annot=True,
        fmt='d',
        cmap='YlOrRd',
        cbar_kws={'label': 'Count'},
        linewidths=0.5,
        ax=ax
    )
    
    ax.set_xlabel('Primary Shade Group', fontsize=14)
    ax.set_ylabel('Undertone', fontsize=14)
    ax.set_title('Undertone vs Primary Shade Group Distribution', fontsize=16, pad=20)
    
    # Add annotation
    annotation = "Validation: High-density hot spots should align with color theory rules"
    fig.text(0.5, 0.02, annotation, ha='center', fontsize=10,
            style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    png_path = os.path.join(output_path, "undertone_shade_heatmap.png")
    svg_path = os.path.join(output_path, "undertone_shade_heatmap.svg")
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved: {png_path}")
    print(f"Saved: {svg_path}")


def plot_finish_by_skin_type(df, output_path="output"):
    """Plot stacked bar chart of recommended finish by skin type"""
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    
    # Create crosstab for stacked bar
    crosstab = pd.crosstab(df['Skin_Type'], df['Recommended_Finish'])
    
    # Plot stacked bar
    crosstab.plot(
        kind='bar',
        stacked=True,
        ax=ax,
        colormap='Set3',
        edgecolor='black',
        linewidth=0.5
    )
    
    ax.set_xlabel('Skin Type', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title('Recommended Finish by Skin Type', fontsize=16, pad=20)
    ax.legend(title='Finish', fontsize=11, title_fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add annotation
    annotation = 'Validation: "Matte" should dominate the "Oily" segment'
    fig.text(0.5, 0.02, annotation, ha='center', fontsize=10,
            style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    png_path = os.path.join(output_path, "finish_by_skin_type.png")
    svg_path = os.path.join(output_path, "finish_by_skin_type.svg")
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")


def plot_harmony_score_distribution(df, output_path="output"):
    """Plot histogram of harmony scores with KDE"""
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    
    # Plot histogram with KDE
    sns.histplot(
        data=df,
        x='Harmony_Score',
        bins=30,
        kde=True,
        color='steelblue',
        edgecolor='black',
        linewidth=0.5,
        ax=ax
    )
    
    ax.set_xlabel('Harmony Score', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_title('Distribution of Harmony Scores', fontsize=16, pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add mean line
    mean_score = df['Harmony_Score'].mean()
    ax.axvline(mean_score, color='red', linestyle='--', linewidth=2,
              label=f'Mean: {mean_score:.3f}')
    ax.legend(fontsize=11)
    
    # Add annotation
    annotation = ("Realistic distribution: Skewed high toward harmonious matches.\n"
                 "Most recommendations are designed to be harmonious (score > 0.7)")
    fig.text(0.5, 0.02, annotation, ha='center', fontsize=10,
            style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    png_path = os.path.join(output_path, "harmony_score_distribution.png")
    svg_path = os.path.join(output_path, "harmony_score_distribution.svg")
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")


def analyze_recommendations(df):
    """Analyze and display recommendation statistics"""
    print("\n" + "=" * 70)
    print("RECOMMENDATION ANALYSIS")
    print("=" * 70)
    
    # Shade recommendations by undertone
    print("\nShade Recommendations by Undertone:")
    shade_by_undertone = pd.crosstab(df['Undertone'], df['Primary_Shade'], normalize='index') * 100
    print(shade_by_undertone.round(1).to_string())
    
    # Finish recommendations by skin type
    print("\n" + "=" * 70)
    print("Finish Recommendations by Skin Type:")
    finish_by_type = pd.crosstab(df['Skin_Type'], df['Recommended_Finish'], normalize='index') * 100
    print(finish_by_type.round(1).to_string())
    
    # Harmony score by contrast level
    print("\n" + "=" * 70)
    print("Harmony Score by Contrast Level:")
    harmony_by_contrast = df.groupby('Contrast_Level')['Harmony_Score'].agg(['mean', 'std', 'min', 'max'])
    print(harmony_by_contrast.round(3).to_string())
    
    # High harmony matches (score > 0.85)
    high_harmony = df[df['Harmony_Score'] > 0.85]
    print(f"\n" + "=" * 70)
    print(f"High Harmony Matches (Score > 0.85): {len(high_harmony)} ({len(high_harmony)/len(df)*100:.1f}%)")
    print("\nTop 5 Harmony Matches:")
    top_matches = df.nlargest(5, 'Harmony_Score')[['Undertone', 'Primary_Shade', 'Contrast_Level', 'Harmony_Score']]
    print(top_matches.to_string(index=False))


def save_final_dataset(df, output_path="output", filename="final_product_recommendations.csv"):
    """Save the final enriched dataset with all recommendations"""
    filepath = os.path.join(output_path, filename)
    df.to_csv(filepath, index=False)
    print(f"\nFinal dataset saved: {filepath}")
    return filepath


def main():
    print("=" * 70)
    print("SHADE RULE ENGINE - Product Recommendation System")
    print("=" * 70)
    
    # Ensure output directory exists
    output_path = "output"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Load data
    df = load_contrast_data()
    if df is None:
        return
    
    # Apply rule engine
    print("\n[STEP 1] Assigning Shade Groups...")
    df = assign_shade_groups(df)
    
    print("\n[STEP 2] Assigning Finish Recommendations...")
    df = assign_finish_recommendations(df)
    
    print("\n[STEP 3] Calculating Harmony Scores...")
    df = calculate_harmony_score(df)
    
    # Show sample
    print("\n[STEP 4] Sample Recommendations (first 10 rows):")
    sample_cols = ['Undertone', 'Skin_Type', 'Contrast_Level', 'Primary_Shade', 
                   'Recommended_Finish', 'Harmony_Score']
    print(df[sample_cols].head(10).to_string(index=False))
    
    # Generate visualizations
    print("\n[STEP 5] Generating Visualizations...")
    plot_undertone_shade_heatmap(df, output_path)
    plot_finish_by_skin_type(df, output_path)
    plot_harmony_score_distribution(df, output_path)
    
    # Analyze recommendations
    print("\n[STEP 6] Analyzing Recommendations...")
    analyze_recommendations(df)
    
    # Save final dataset
    print("\n[STEP 7] Saving Final Dataset...")
    save_final_dataset(df, output_path)
    
    print("\n" + "=" * 70)
    print("SHADE RULE ENGINE COMPLETE")
    print("All product recommendations generated successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
