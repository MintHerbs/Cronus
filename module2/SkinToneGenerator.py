"""
SkinToneGenerator.py
Generates synthetic skin tone profiles using CIELAB color space
and visualizes distribution using Seaborn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set seaborn theme
sns.set_theme(style="whitegrid", context="talk")


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
        Generate synthetic skin tone dataset
        
        Returns:
            pandas DataFrame with columns: L, a, b, MST_Class, Undertone
        """
        print(f"Generating {self.n_samples} synthetic skin tone profiles...")
        
        # Sample L*, a*, b* values
        L_values = np.random.uniform(self.L_min, self.L_max, self.n_samples)
        a_values = np.random.uniform(self.a_min, self.a_max, self.n_samples)
        b_values = np.random.uniform(self.b_min, self.b_max, self.n_samples)
        
        # Create DataFrame
        df = pd.DataFrame({
            'L': L_values,
            'a': a_values,
            'b': b_values
        })
        
        # Map Monk Skin Tone (MST) classes based on L* value
        df['MST_Class'] = self._assign_mst_class(df['L'])
        
        # Derive undertone from a*/b* ratio
        df['Undertone'] = self._derive_undertone(df['a'], df['b'])
        
        print(f"Dataset generated: {len(df)} samples")
        print(f"\nUndertone distribution:")
        print(df['Undertone'].value_counts())
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
    
    def visualize_dataset(self, df):
        """
        Generate visualizations of the synthetic dataset
        
        Args:
            df: pandas DataFrame with skin tone data
        """
        print("\nGenerating visualizations...")
        
        # Graph A: a* vs b* scatter plot with undertone hue
        self._plot_ab_scatter(df)
        
        # Graph B: L* histogram with MST class hue
        self._plot_L_histogram(df)
        
        print("Visualizations complete")
    
    def _plot_ab_scatter(self, df):
        """Generate a* vs b* scatter plot colored by undertone"""
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
        
        # Define color palette for undertones
        palette = {
            'Warm': '#FF6B35',    # Orange-red
            'Cool': '#4ECDC4',    # Cyan-blue
            'Neutral': '#95B46A'  # Olive-green
        }
        
        sns.scatterplot(
            data=df,
            x='a',
            y='b',
            hue='Undertone',
            palette=palette,
            alpha=0.6,
            s=50,
            edgecolor='none',
            ax=ax
        )
        
        ax.set_xlabel('a* (Green-Red Axis)', fontsize=14)
        ax.set_ylabel('b* (Blue-Yellow Axis)', fontsize=14)
        ax.set_title('Skin Tone Distribution: a* vs b* by Undertone', fontsize=16, pad=20)
        ax.legend(title='Undertone', fontsize=12, title_fontsize=13)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/ab_scatter_undertone.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.output_path}/ab_scatter_undertone.svg", bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {self.output_path}/ab_scatter_undertone.png")
        print(f"Saved: {self.output_path}/ab_scatter_undertone.svg")
    
    def _plot_L_histogram(self, df):
        """Generate L* histogram colored by MST class"""
        fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
        
        # Convert MST_Class to string for proper categorical handling
        df_plot = df.copy()
        df_plot['MST_Class_str'] = df_plot['MST_Class'].astype(str)
        
        # Use seaborn color palette for MST classes
        palette = sns.color_palette("Spectral", n_colors=10)
        
        sns.histplot(
            data=df_plot,
            x='L',
            hue='MST_Class_str',
            palette=palette,
            multiple="stack",
            bins=30,
            edgecolor='white',
            linewidth=0.5,
            ax=ax,
            legend=True
        )
        
        ax.set_xlabel('L* (Lightness)', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        ax.set_title('Skin Tone Distribution: L* by Monk Skin Tone Class', fontsize=16, pad=20)
        
        # Update legend title
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, title='MST Class', fontsize=10, title_fontsize=12, ncol=2)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add vertical lines for MST class boundaries
        bins = np.linspace(self.L_min, self.L_max, 11)
        for bin_edge in bins[1:-1]:
            ax.axvline(bin_edge, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/L_histogram_mst.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.output_path}/L_histogram_mst.svg", bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {self.output_path}/L_histogram_mst.png")
        print(f"Saved: {self.output_path}/L_histogram_mst.svg")
    
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
    generator = SkinToneGenerator(n_samples=1000, random_state=42)
    
    # Generate dataset
    df = generator.generate_dataset()
    
    # Display sample data
    print("\nSample data (first 10 rows):")
    print(df.head(10).to_string(index=False))
    
    # Generate visualizations
    generator.visualize_dataset(df)
    
    # Save dataset
    generator.save_dataset(df)
    
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
