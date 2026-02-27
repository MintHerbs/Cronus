"""
KMeansSegmentation.py
Performs K-Means clustering on masked skin pixels only in LAB color space
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.cluster import KMeans
import os

# Set seaborn theme
sns.set_theme(style="whitegrid", context="talk")


class KMeansSegmentation:
    def __init__(self, k=3, random_state=42):
        """
        Initialize K-Means segmentation module
        
        Args:
            k: Number of clusters (default 3)
            random_state: Random seed for reproducibility
        """
        self.k = k
        self.random_state = random_state
        self.output_path = "module1/output"
        
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
    
    def process_skin_pixels(self, skin_pixels_rgb):
        """
        Apply K-Means clustering to masked skin pixels only
        
        Args:
            skin_pixels_rgb: numpy array of shape (num_skin_pixels, 3) in RGB format
        
        Returns:
            dict with k, centroids_lab, centroids_rgb, pixel_counts, labels
        """
        if skin_pixels_rgb.shape[0] == 0:
            raise ValueError("No skin pixels provided for clustering")
        
        print(f"Clustering {skin_pixels_rgb.shape[0]:,} skin pixels in LAB space...")
        
        # Convert RGB to LAB for perceptually uniform clustering
        # Reshape to image format for cv2 conversion
        pixels_reshaped = skin_pixels_rgb.reshape(1, -1, 3).astype(np.uint8)
        pixels_lab = cv2.cvtColor(pixels_reshaped, cv2.COLOR_RGB2LAB)
        pixels_lab = pixels_lab.reshape(-1, 3).astype(np.float32)
        
        # Apply K-Means clustering in LAB space
        kmeans = KMeans(n_clusters=self.k, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(pixels_lab)
        centroids_lab = kmeans.cluster_centers_
        
        # Convert centroids back to RGB for visualization
        centroids_lab_uint8 = centroids_lab.astype(np.uint8).reshape(1, -1, 3)
        centroids_rgb_uint8 = cv2.cvtColor(centroids_lab_uint8, cv2.COLOR_LAB2RGB)
        centroids_rgb = centroids_rgb_uint8.reshape(-1, 3).astype(np.float32)
        
        # Count pixels per cluster
        unique, counts = np.unique(labels, return_counts=True)
        pixel_counts = dict(zip(unique, counts))
        
        # Generate visualizations
        self._generate_segmented_visualization(
            skin_pixels_rgb, labels, centroids_rgb, pixel_counts
        )
        self._generate_lab_distribution(
            pixels_lab, labels, centroids_lab, pixel_counts
        )
        
        return {
            "k": self.k,
            "centroids_lab": centroids_lab.tolist(),
            "centroids_rgb": centroids_rgb.tolist(),
            "pixel_counts": [pixel_counts.get(i, 0) for i in range(self.k)],
            "labels": labels
        }
    
    def _generate_segmented_visualization(self, skin_pixels_rgb, labels, centroids_rgb, pixel_counts):
        """Generate segmented skin pixel visualization"""
        # Reconstruct segmented pixels
        segmented_pixels = centroids_rgb[labels].astype(np.uint8)
        
        # Create a grid visualization of clustered pixels
        num_pixels = len(skin_pixels_rgb)
        grid_size = int(np.ceil(np.sqrt(num_pixels)))
        
        # Pad to square
        pad_size = grid_size * grid_size - num_pixels
        if pad_size > 0:
            segmented_pixels = np.vstack([
                segmented_pixels,
                np.zeros((pad_size, 3), dtype=np.uint8)
            ])
        
        segmented_image = segmented_pixels.reshape(grid_size, grid_size, 3)
        
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
        ax.imshow(segmented_image)
        ax.axis('off')
        ax.set_title("K-Means Clustered Skin Pixels", fontsize=14, pad=20)
        
        # Create annotation text
        annotation_lines = [f"K = {self.k}"]
        annotation_lines.append("Clustering performed on masked skin pixels only")
        annotation_lines.append("Background removed via skin masking")
        annotation_lines.append("")
        
        for i, centroid in enumerate(centroids_rgb):
            r, g, b = centroid.astype(int)
            count = pixel_counts.get(i, 0)
            percentage = (count / sum(pixel_counts.values())) * 100
            annotation_lines.append(f"Cluster {i}: RGB({r}, {g}, {b})")
            annotation_lines.append(f"  Pixels: {count:,} ({percentage:.1f}%)")
        
        annotation_text = "\n".join(annotation_lines)
        
        # Add text with semi-transparent background
        props = dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray')
        ax.text(0.02, 0.98, annotation_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', bbox=props,
               family='monospace')
        
        plt.tight_layout()
        
        # Save as PNG and SVG
        plt.savefig(f"{self.output_path}/kmeans_segmented.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.output_path}/kmeans_segmented.svg", bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {self.output_path}/kmeans_segmented.png")
        print(f"Saved: {self.output_path}/kmeans_segmented.svg")
    
    def _generate_lab_distribution(self, pixels_lab, labels, centroids_lab, pixel_counts):
        """Generate 3D LAB scatter plot"""
        fig = plt.figure(figsize=(12, 10), dpi=300)
        ax = fig.add_subplot(111, projection='3d')
        
        # Use seaborn color palette
        palette = sns.color_palette("deep", self.k)
        
        # Sample pixels for visualization
        max_points = 5000
        if len(pixels_lab) > max_points:
            indices = np.random.choice(len(pixels_lab), max_points, replace=False)
            pixels_sample = pixels_lab[indices]
            labels_sample = labels[indices]
        else:
            pixels_sample = pixels_lab
            labels_sample = labels
        
        # Plot each cluster
        legend_labels = []
        for i in range(self.k):
            cluster_pixels = pixels_sample[labels_sample == i]
            if len(cluster_pixels) > 0:
                ax.scatter(cluster_pixels[:, 0], cluster_pixels[:, 1], cluster_pixels[:, 2],
                          c=[palette[i]], alpha=0.3, s=1, label=f'Cluster {i}')
            
            # Plot centroid
            centroid = centroids_lab[i]
            ax.scatter(centroid[0], centroid[1], centroid[2],
                      c=[palette[i]], s=200, marker='*', 
                      edgecolors='black', linewidths=2)
            
            L, a, b = centroid.astype(int)
            count = pixel_counts.get(i, 0)
            percentage = (count / sum(pixel_counts.values())) * 100
            legend_labels.append(f'Cluster {i}: LAB({L},{a},{b}) - {percentage:.1f}%')
        
        ax.set_xlabel('L (Lightness)', fontsize=12)
        ax.set_ylabel('a (Green-Red)', fontsize=12)
        ax.set_zlabel('b (Blue-Yellow)', fontsize=12)
        ax.set_title(f'K-Means Skin Tone Distribution in LAB Space (K = {self.k})', 
                    fontsize=14, pad=20)
        
        # Add legend
        ax.legend(legend_labels, loc='upper left', fontsize=9)
        
        # Add explanatory text
        explanation = ("K-Means clustering performed on masked skin pixels only.\n"
                      "LAB color space ensures perceptually uniform clustering.\n"
                      "Background removed via skin masking.")
        fig.text(0.5, 0.02, explanation, ha='center', fontsize=10,
                style='italic', wrap=True, bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        
        # Save as PNG and SVG
        plt.savefig(f"{self.output_path}/kmeans_color_distribution.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.output_path}/kmeans_color_distribution.svg", bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {self.output_path}/kmeans_color_distribution.png")
        print(f"Saved: {self.output_path}/kmeans_color_distribution.svg")


if __name__ == "__main__":
    print("KMeansSegmentation module - use via Orchestrator")
    print("This module now operates on masked skin pixels only")
