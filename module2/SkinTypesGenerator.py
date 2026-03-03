"""
SkinTypesGenerator.py
Assigns skin type and texture descriptors using conditional probability
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set seaborn theme
sns.set_theme(style="whitegrid", context="talk")


def load_skin_tone_data(filepath="output/synthetic_skin_tones.csv"):
    """Load the synthetic skin tone dataset"""
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        print("Please run SkinToneGenerator.py first!")
        return None
    
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} skin tone profiles from {filepath}")
    return df


def assign_skin_types(df, random_state=42):
    """Assign skin types based on population weights"""
    np.random.seed(random_state)
    
    skin_types = ['Normal', 'Oily', 'Dry', 'Combination', 'Sensitive']
    weights = [0.30, 0.25, 0.25, 0.15, 0.05]
    
    df['Skin_Type'] = np.random.choice(skin_types, size=len(df), p=weights)
    
    print("\nSkin Type Distribution:")
    print(df['Skin_Type'].value_counts())
    
    return df


def assign_textures(df, random_state=42):
    """Assign texture descriptors based on conditional probability"""
    np.random.seed(random_state)
    
    # Conditional probability matrix
    texture_probs = {
        'Normal': [0.70, 0.10, 0.05, 0.15],      # Smooth, Shiny, Rough, Uneven
        'Oily': [0.15, 0.70, 0.05, 0.10],
        'Dry': [0.10, 0.05, 0.60, 0.25],
        'Combination': [0.30, 0.30, 0.10, 0.30],
        'Sensitive': [0.20, 0.10, 0.30, 0.40]
    }
    
    textures = ['Smooth', 'Shiny', 'Rough', 'Uneven']
    
    # Assign texture for each row
    texture_list = []
    for skin_type in df['Skin_Type']:
        probs = texture_probs[skin_type]
        texture = np.random.choice(textures, p=probs)
        texture_list.append(texture)
    
    df['Texture_Descriptor'] = texture_list
    
    print("\nTexture Descriptor Distribution:")
    print(df['Texture_Descriptor'].value_counts())
    
    return df


def plot_skin_type_distribution(df, output_path="output"):
    """Plot skin type distribution"""
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    
    palette = sns.color_palette("Set2", n_colors=5)
    order = ['Normal', 'Oily', 'Dry', 'Combination', 'Sensitive']
    
    sns.countplot(data=df, x='Skin_Type', order=order, palette=palette, ax=ax)
    
    ax.set_xlabel('Skin Type', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title('Skin Type Distribution (Population Weights)', fontsize=16, pad=20)
    
    # Add count labels
    for container in ax.containers:
        ax.bar_label(container, fontsize=11)
    
    plt.tight_layout()
    
    png_path = os.path.join(output_path, "skin_type_distribution.png")
    svg_path = os.path.join(output_path, "skin_type_distribution.svg")
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved: {png_path}")
    print(f"Saved: {svg_path}")


def plot_texture_by_skin_type(df, output_path="output"):
    """Plot texture distribution by skin type"""
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    
    palette = sns.color_palette("husl", n_colors=4)
    skin_order = ['Normal', 'Oily', 'Dry', 'Combination', 'Sensitive']
    texture_order = ['Smooth', 'Shiny', 'Rough', 'Uneven']
    
    sns.countplot(
        data=df,
        x='Skin_Type',
        hue='Texture_Descriptor',
        order=skin_order,
        hue_order=texture_order,
        palette=palette,
        ax=ax
    )
    
    ax.set_xlabel('Skin Type', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title('Texture Distribution by Skin Type (Conditional Probability)', 
                fontsize=16, pad=20)
    ax.legend(title='Texture Descriptor', fontsize=11, title_fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    png_path = os.path.join(output_path, "texture_by_skin_type.png")
    svg_path = os.path.join(output_path, "texture_by_skin_type.svg")
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")


def save_dataset(df, output_path="output", filename="skin_profiles_with_type.csv"):
    """Save the updated dataset"""
    filepath = os.path.join(output_path, filename)
    df.to_csv(filepath, index=False)
    print(f"\nDataset saved: {filepath}")
    return filepath


def main():
    print("=" * 70)
    print("SKIN TYPES GENERATOR - Conditional Probability Assignment")
    print("=" * 70)
    
    # Ensure output directory exists
    output_path = "output"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Load data
    df = load_skin_tone_data()
    if df is None:
        return
    
    # Assign skin types
    print("\n[STEP 1] Assigning Skin Types...")
    df = assign_skin_types(df, random_state=42)
    
    # Assign textures
    print("\n[STEP 2] Assigning Texture Descriptors...")
    df = assign_textures(df, random_state=42)
    
    # Show sample
    print("\n[STEP 3] Sample Data (first 10 rows):")
    print(df.head(10).to_string(index=False))
    
    # Generate visualizations
    print("\n[STEP 4] Generating Visualizations...")
    plot_skin_type_distribution(df, output_path)
    plot_texture_by_skin_type(df, output_path)
    
    # Save dataset
    print("\n[STEP 5] Saving Dataset...")
    save_dataset(df, output_path)
    
    print("\n" + "=" * 70)
    print("ASSIGNMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
