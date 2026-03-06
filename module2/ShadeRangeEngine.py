"""
ShadeRangeEngine.py
Rule-based engine that assigns primary/sub colour groups and LAB shade ranges
to skin profiles based on undertone, lightness, and contrast level
"""

import numpy as np
import pandas as pd
import os

# ─────────────────────────────────────────────
# RULE ENGINE CONSTANTS
# ─────────────────────────────────────────────

# Primary group candidates by undertone
PRIMARY_CANDIDATES = {
    'Warm': ['Coral', 'Nude', 'Red'],
    'Cool': ['Berry', 'Mauve', 'Pink'],
    'Neutral': ['Red', 'Nude', 'Mauve']
}

# Sub group assignment rules: (undertone, L_min, L_max) -> (primary, sub)
SUB_GROUP_RULES = [
    # Warm undertones
    ('Warm', 0, 45, 'Coral', 'Deep Coral'),
    ('Warm', 45, 60, 'Coral', 'True Coral'),
    ('Warm', 60, 72, 'Nude', 'Warm Nude'),
    ('Warm', 72, 100, 'Nude', 'Peachy Nude'),
    
    # Cool undertones
    ('Cool', 0, 45, 'Berry', 'Deep Berry'),
    ('Cool', 45, 58, 'Berry', 'True Berry'),
    ('Cool', 58, 70, 'Mauve', 'Cool Mauve'),
    ('Cool', 70, 100, 'Pink', 'Soft Pink'),
    
    # Neutral undertones
    ('Neutral', 0, 45, 'Red', 'True Red'),
    ('Neutral', 45, 58, 'Red', 'Blue Red'),
    ('Neutral', 58, 70, 'Mauve', 'Dusty Rose'),
    ('Neutral', 70, 100, 'Nude', 'Taupe Nude'),
]

# Centre LAB points for each sub group
SUB_GROUP_CENTRES = {
    'Deep Coral': {'L': 40, 'a': 32, 'b': 28},
    'True Coral': {'L': 52, 'a': 30, 'b': 26},
    'Warm Nude': {'L': 62, 'a': 18, 'b': 22},
    'Peachy Nude': {'L': 70, 'a': 14, 'b': 20},
    'Deep Berry': {'L': 32, 'a': 38, 'b': -5},
    'True Berry': {'L': 42, 'a': 35, 'b': -3},
    'Cool Mauve': {'L': 55, 'a': 22, 'b': 2},
    'Soft Pink': {'L': 68, 'a': 25, 'b': 5},
    'True Red': {'L': 38, 'a': 48, 'b': 20},
    'Blue Red': {'L': 45, 'a': 45, 'b': 10},
    'Dusty Rose': {'L': 58, 'a': 20, 'b': 8},
    'Taupe Nude': {'L': 65, 'a': 12, 'b': 10},
}

# Range widths by contrast level
CONTRAST_RANGE_WIDTHS = {
    'High': {'L': 12, 'a': 10, 'b': 10},
    'Medium': {'L': 8, 'a': 7, 'b': 7},
    'Low': {'L': 5, 'a': 4, 'b': 4},
}

# LAB gamut limits
LAB_LIMITS = {
    'L': (0, 100),
    'a': (-128, 127),
    'b': (-128, 127),
}

# File paths
INPUT_FILE = "output/skin_profiles_with_contrast.csv"
OUTPUT_DIR = "output"
OUTPUT_FILE = "skin_profiles_with_shades.csv"


def load_contrast_profiles(filepath=INPUT_FILE):
    """
    Load the contrast-enriched skin profiles
    
    Args:
        filepath: Path to the input CSV file
    
    Returns:
        pandas DataFrame with skin tone and contrast data
    """
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        print("Please run ContrastCalculator.py first!")
        return None
    
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} skin profiles from {filepath}")
    print(f"Columns: {df.columns.tolist()}")
    
    return df


def assign_primary_and_sub_group(df):
    """
    Assign primary_group and sub_group based on Undertone and L* value
    
    Args:
        df: DataFrame with Undertone and L columns
    
    Returns:
        DataFrame with added primary_group and sub_group columns
    """
    print("\nAssigning primary and sub groups based on rule engine...")
    
    primary_groups = []
    sub_groups = []
    
    for _, row in df.iterrows():
        undertone = row['Undertone']
        L_value = row['L']
        
        # Find matching rule
        matched = False
        for rule in SUB_GROUP_RULES:
            rule_undertone, L_min, L_max, primary, sub = rule
            
            if undertone == rule_undertone and L_min <= L_value < L_max:
                primary_groups.append(primary)
                sub_groups.append(sub)
                matched = True
                break
        
        if not matched:
            # Fallback (should not happen with proper rules)
            print(f"[WARNING] No rule matched for Undertone={undertone}, L={L_value:.2f}")
            primary_groups.append('Unknown')
            sub_groups.append('Unknown')
    
    df['primary_group'] = primary_groups
    df['sub_group'] = sub_groups
    
    print(f"\nPrimary group distribution:")
    print(df['primary_group'].value_counts())
    
    print(f"\nSub group distribution:")
    print(df['sub_group'].value_counts())
    
    return df


def calculate_shade_ranges(df):
    """
    Calculate shade LAB ranges based on sub_group and Contrast_Level
    
    Args:
        df: DataFrame with sub_group and Contrast_Level columns
    
    Returns:
        DataFrame with added shade range columns
    """
    print("\nCalculating shade LAB ranges...")
    
    shade_L_min = []
    shade_L_max = []
    shade_a_min = []
    shade_a_max = []
    shade_b_min = []
    shade_b_max = []
    
    for _, row in df.iterrows():
        sub_group = row['sub_group']
        contrast_level = row['Contrast_Level']
        
        # Get centre point for this sub group
        if sub_group not in SUB_GROUP_CENTRES:
            print(f"[WARNING] Unknown sub_group: {sub_group}")
            # Use default centre
            centre = {'L': 50, 'a': 20, 'b': 10}
        else:
            centre = SUB_GROUP_CENTRES[sub_group]
        
        # Get range width for this contrast level
        if contrast_level not in CONTRAST_RANGE_WIDTHS:
            print(f"[WARNING] Unknown Contrast_Level: {contrast_level}")
            # Use medium as default
            widths = CONTRAST_RANGE_WIDTHS['Medium']
        else:
            widths = CONTRAST_RANGE_WIDTHS[contrast_level]
        
        # Calculate ranges: centre ± width
        L_min = centre['L'] - widths['L']
        L_max = centre['L'] + widths['L']
        a_min = centre['a'] - widths['a']
        a_max = centre['a'] + widths['a']
        b_min = centre['b'] - widths['b']
        b_max = centre['b'] + widths['b']
        
        # Clip to LAB gamut
        L_min = np.clip(L_min, LAB_LIMITS['L'][0], LAB_LIMITS['L'][1])
        L_max = np.clip(L_max, LAB_LIMITS['L'][0], LAB_LIMITS['L'][1])
        a_min = np.clip(a_min, LAB_LIMITS['a'][0], LAB_LIMITS['a'][1])
        a_max = np.clip(a_max, LAB_LIMITS['a'][0], LAB_LIMITS['a'][1])
        b_min = np.clip(b_min, LAB_LIMITS['b'][0], LAB_LIMITS['b'][1])
        b_max = np.clip(b_max, LAB_LIMITS['b'][0], LAB_LIMITS['b'][1])
        
        shade_L_min.append(L_min)
        shade_L_max.append(L_max)
        shade_a_min.append(a_min)
        shade_a_max.append(a_max)
        shade_b_min.append(b_min)
        shade_b_max.append(b_max)
    
    df['shade_L_min'] = shade_L_min
    df['shade_L_max'] = shade_L_max
    df['shade_a_min'] = shade_a_min
    df['shade_a_max'] = shade_a_max
    df['shade_b_min'] = shade_b_min
    df['shade_b_max'] = shade_b_max
    
    print(f"Calculated shade ranges for {len(df)} profiles")
    
    return df


def validate_ranges(df):
    """
    Validate that all shade ranges are properly ordered (min < max)
    
    Args:
        df: DataFrame with shade range columns
    
    Returns:
        bool: True if all validations pass
    """
    print("\n" + "=" * 70)
    print("SHADE RANGE VALIDATION")
    print("=" * 70)
    
    all_valid = True
    
    # Check L ranges
    L_violations = df['shade_L_min'] >= df['shade_L_max']
    if L_violations.any():
        n_violations = L_violations.sum()
        print(f"[WARNING] {n_violations} rows have shade_L_min >= shade_L_max")
        print("Example violations:")
        print(df[L_violations][['sub_group', 'Contrast_Level', 'shade_L_min', 'shade_L_max']].head())
        all_valid = False
    else:
        print(f"PASS - All rows have shade_L_min < shade_L_max")
    
    # Check a ranges
    a_violations = df['shade_a_min'] >= df['shade_a_max']
    if a_violations.any():
        n_violations = a_violations.sum()
        print(f"[WARNING] {n_violations} rows have shade_a_min >= shade_a_max")
        all_valid = False
    else:
        print(f"PASS - All rows have shade_a_min < shade_a_max")
    
    # Check b ranges
    b_violations = df['shade_b_min'] >= df['shade_b_max']
    if b_violations.any():
        n_violations = b_violations.sum()
        print(f"[WARNING] {n_violations} rows have shade_b_min >= shade_b_max")
        all_valid = False
    else:
        print(f"PASS - All rows have shade_b_min < shade_b_max")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SHADE RANGE STATISTICS")
    print("=" * 70)
    
    print(f"\nL* range widths:")
    L_widths = df['shade_L_max'] - df['shade_L_min']
    print(f"  Mean: {L_widths.mean():.2f}")
    print(f"  Std:  {L_widths.std():.2f}")
    print(f"  Min:  {L_widths.min():.2f}")
    print(f"  Max:  {L_widths.max():.2f}")
    
    print(f"\na* range widths:")
    a_widths = df['shade_a_max'] - df['shade_a_min']
    print(f"  Mean: {a_widths.mean():.2f}")
    print(f"  Std:  {a_widths.std():.2f}")
    print(f"  Min:  {a_widths.min():.2f}")
    print(f"  Max:  {a_widths.max():.2f}")
    
    print(f"\nb* range widths:")
    b_widths = df['shade_b_max'] - df['shade_b_min']
    print(f"  Mean: {b_widths.mean():.2f}")
    print(f"  Std:  {b_widths.std():.2f}")
    print(f"  Min:  {b_widths.min():.2f}")
    print(f"  Max:  {b_widths.max():.2f}")
    
    return all_valid


def save_dataset(df, output_dir=OUTPUT_DIR, filename=OUTPUT_FILE):
    """
    Save the enriched dataset with shade assignments
    
    Args:
        df: DataFrame with all columns
        output_dir: Directory to save the file
        filename: Output filename
    """
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"\nDataset saved: {filepath}")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    return filepath


def main():
    """Main execution function"""
    print("=" * 70)
    print("SHADE RANGE ENGINE - Rule-Based Colour Assignment")
    print("=" * 70)
    
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Step 1: Load contrast profiles
    print("\n[STEP 1] Loading Contrast-Enriched Profiles...")
    df = load_contrast_profiles()
    if df is None:
        return
    
    # Step 2: Assign primary and sub groups
    print("\n[STEP 2] Assigning Primary and Sub Groups...")
    df = assign_primary_and_sub_group(df)
    
    # Step 3: Calculate shade LAB ranges
    print("\n[STEP 3] Calculating Shade LAB Ranges...")
    df = calculate_shade_ranges(df)
    
    # Step 4: Validate ranges
    print("\n[STEP 4] Validating Shade Ranges...")
    is_valid = validate_ranges(df)
    
    if not is_valid:
        print("\n[WARNING] Validation issues detected. Review output above.")
    
    # Step 5: Show sample data
    print("\n[STEP 5] Sample Data (first 10 rows):")
    display_cols = ['L', 'Undertone', 'Contrast_Level', 'primary_group', 'sub_group',
                   'shade_L_min', 'shade_L_max']
    print(df[display_cols].head(10).to_string(index=False))
    
    # Step 6: Save dataset
    print("\n[STEP 6] Saving Enriched Dataset...")
    save_dataset(df, OUTPUT_DIR)
    
    print("\n" + "=" * 70)
    print("SHADE RANGE ASSIGNMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
