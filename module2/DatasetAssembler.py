#!/usr/bin/env python3
"""
DatasetAssembler.py
Final dataset assembly, validation, and visualization
"""

import pandas as pd

# Load and prepare final dataset
df = pd.read_csv("output/skin_profiles_with_shades.csv")
print(f"DATASET ASSEMBLER: Loading {len(df)} records from shades file")

# Prepare final schema
column_mapping = {
    'L': 'skin_L',
    'a': 'skin_a', 
    'b': 'skin_b',
    'Undertone': 'undertone',
    'Delta_E': 'delta_e',
    'normal_pct': 'normal_pct',
    'oily_pct': 'oily_pct',
    'dry_pct': 'dry_pct',
    'Contrast_Level': 'contrast_level',
    'primary_group': 'primary_group',
    'sub_group': 'sub_group',
    'shade_L_min': 'shade_L_min',
    'shade_L_max': 'shade_L_max',
    'shade_a_min': 'shade_a_min',
    'shade_a_max': 'shade_a_max',
    'shade_b_min': 'shade_b_min',
    'shade_b_max': 'shade_b_max'
}

# Select and rename columns
final_df = df[list(column_mapping.keys())].copy()
final_df = final_df.rename(columns=column_mapping)

# Convert contrast level to lowercase
final_df['contrast_level'] = final_df['contrast_level'].str.lower()

# Shuffle rows
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save final dataset
filepath = "output/final_skin_tone_dataset.csv"
final_df.to_csv(filepath, index=False)
print(f"DATASET ASSEMBLER: Final dataset saved with {len(final_df)} records to {filepath}")

# Print sub-group balance
final_counts = final_df['sub_group'].value_counts().sort_index()
print(f"DATASET ASSEMBLER: Sub-group balance - Min: {final_counts.min()}, Max: {final_counts.max()}")
print("DATASET ASSEMBLER: Assembly complete")