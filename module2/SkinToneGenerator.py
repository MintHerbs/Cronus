"""
SkinToneGenerator.py
Generates synthetic skin tone profiles using CIELAB color space
"""

import numpy as np
import pandas as pd
import os


class SkinToneGenerator:
    def __init__(self, n_samples=1000, random_state=42):
        """
        Initialize skin tone generator
        
        Args:
            n_samples: Number of synthetic samples to generate
            random_state: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.random_state = random_state
        self.output_path = "output"
        
        # CIELAB bounds for skin tones
        self.L_min, self.L_max = 30, 85
        self.a_min, self.a_max = 5, 25
        self.b_min, self.b_max = 8, 35
        
        # Undertone thresholds
        self.warm_threshold = 1.2
        self.cool_threshold = 0.8
        
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        
        np.random.seed(self.random_state)
    
    def generate_dataset(self):
        """
        Generate synthetic skin tone dataset using stratified sampling
        to ensure balanced sub-groups in the final pipeline output

        Returns:
            pandas DataFrame with columns: L, a, b, MST_Class, Undertone
        """
        print(f"Generating {self.n_samples} synthetic skin tone profiles using stratified sampling...")

        # Define sub-group L* boundaries (matching ShadeRangeEngine.py)
        subgroup_buckets = [
            # Warm undertones
            ('Warm', 30, 45, 'Deep Coral'),
            ('Warm', 45, 60, 'True Coral'),
            ('Warm', 60, 72, 'Warm Nude'),
            ('Warm', 72, 85, 'Peachy Nude'),

            # Cool undertones
            ('Cool', 30, 45, 'Deep Berry'),
            ('Cool', 45, 58, 'True Berry'),
            ('Cool', 58, 70, 'Cool Mauve'),
            ('Cool', 70, 85, 'Soft Pink'),

            # Neutral undertones
            ('Neutral', 30, 45, 'True Red'),
            ('Neutral', 45, 58, 'Blue Red'),
            ('Neutral', 58, 70, 'Dusty Rose'),
            ('Neutral', 70, 85, 'Taupe Nude'),
        ]

        # Calculate records per sub-group
        records_per_subgroup = self.n_samples // 12
        remainder = self.n_samples % 12

        print(f"Target records per sub-group: {records_per_subgroup}")
        if remainder > 0:
            print(f"Distributing {remainder} extra records across first {remainder} buckets")

        # Generate stratified samples
        all_samples = []
        bucket_counts = {}

        for i, (undertone, L_min, L_max, subgroup_name) in enumerate(subgroup_buckets):
            # Calculate exact count for this bucket
            bucket_size = records_per_subgroup + (1 if i < remainder else 0)

            # Generate L* values within this bucket's range
            L_values = np.random.uniform(L_min, L_max, bucket_size)

            # Generate a* and b* values across full valid ranges
            a_values = np.random.uniform(self.a_min, self.a_max, bucket_size)
            b_values = np.random.uniform(self.b_min, self.b_max, bucket_size)

            # Create DataFrame for this bucket
            bucket_df = pd.DataFrame({
                'L': L_values,
                'a': a_values,
                'b': b_values,
                'Undertone': [undertone] * bucket_size  # Assign undertone deterministically
            })

            all_samples.append(bucket_df)
            bucket_counts[f"{undertone} ({L_min}-{L_max})"] = bucket_size

        # Concatenate all buckets
        df = pd.concat(all_samples, ignore_index=True)

        # Assign MST classes based on L* values
        df['MST_Class'] = self._assign_mst_class(df['L'])

        # Shuffle the dataset to avoid ordering by sub-group
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"Dataset generated: {len(df)} samples")

        # Print stratification validation
        print(f"\nStratification validation:")
        print("Records per undertone-L* bucket:")
        for bucket_name, count in bucket_counts.items():
            print(f"  {bucket_name}: {count}")

        # Check for imbalance
        min_count = min(bucket_counts.values())
        max_count = max(bucket_counts.values())
        if max_count - min_count > 1:
            print(f"WARNING: Bucket imbalance detected! Min: {min_count}, Max: {max_count}")
        else:
            print("PASS: All buckets balanced (differ by at most 1)")

        # Print overall undertone distribution
        print(f"\nOverall undertone distribution:")
        undertone_counts = df['Undertone'].value_counts().sort_index()
        print(undertone_counts)

        print(f"\nMST Class distribution:")
        print(df['MST_Class'].value_counts().sort_index())

        return df

    
    def _assign_mst_class(self, L_values):
        """
        Assign Monk Skin Tone class (1-10) based on L* value
        Uses equal-width bins across the L* range
        
        Args:
            L_values: pandas Series of L* values
        
        Returns:
            pandas Series of MST classes (1-10)
        """
        # Create 10 equal-width bins
        bins = np.linspace(self.L_min, self.L_max, 11)
        labels = list(range(1, 11))
        
        mst_classes = pd.cut(L_values, bins=bins, labels=labels, include_lowest=True)
        
        return mst_classes.astype(int)
    
    def _derive_undertone(self, a_values, b_values):
        """
        Derive undertone classification from a*/b* ratio
        
        Args:
            a_values: pandas Series of a* values
            b_values: pandas Series of b* values
        
        Returns:
            pandas Series of undertone classifications
        """
        ratio = b_values / a_values
        
        undertones = []
        for r in ratio:
            if r > self.warm_threshold:
                undertones.append('Warm')
            elif r < self.cool_threshold:
                undertones.append('Cool')
            else:
                undertones.append('Neutral')
        
        return pd.Series(undertones)
    

    def save_dataset(self, df, filename="synthetic_skin_tones.csv"):
        """
        Save dataset to CSV
        
        Args:
            df: pandas DataFrame
            filename: output filename
        """
        filepath = os.path.join(self.output_path, filename)
        df.to_csv(filepath, index=False)
        print(f"\nDataset saved: {filepath}")
        return filepath


def main():
    """Main execution function"""
    print("=" * 70)
    print("SKIN TONE GENERATOR - Synthetic Dataset Creation")
    print("=" * 70)
    
    # Initialize generator
    generator = SkinToneGenerator(n_samples=3000, random_state=42)
    
    # Generate dataset
    df = generator.generate_dataset()
    
    # Display sample data
    print("\nSample data (first 10 rows):")
    print(df.head(10).to_string(index=False))
    
    # Save dataset
    generator.save_dataset(df)
    
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
