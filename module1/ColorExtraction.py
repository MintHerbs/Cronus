"""
ColorExtraction.py
Extracts dominant skin tone from clustered data
"""

import numpy as np
import cv2


class ColorExtraction:
    def __init__(self):
        """Initialize color extraction module"""
        pass
    
    def extract_dominant_tone(self, clustering_result, skin_pixels_rgb):
        """
        Extract dominant skin tone using median LAB value
        
        Args:
            clustering_result: dict from KMeansSegmentation
            skin_pixels_rgb: original skin pixels in RGB
        
        Returns:
            dict with dominant_cluster, tone_vector_lab, tone_vector_rgb
        """
        # Find dominant cluster by highest pixel count
        pixel_counts = clustering_result["pixel_counts"]
        dominant_cluster_idx = np.argmax(pixel_counts)
        dominant_pixel_count = pixel_counts[dominant_cluster_idx]
        
        print(f"\nDominant cluster: {dominant_cluster_idx} "
              f"({dominant_pixel_count:,} pixels, "
              f"{(dominant_pixel_count / sum(pixel_counts)) * 100:.1f}%)")
        
        # Get pixels belonging to dominant cluster
        labels = clustering_result["labels"]
        dominant_pixels_rgb = skin_pixels_rgb[labels == dominant_cluster_idx]
        
        # Convert to LAB for median calculation
        pixels_reshaped = dominant_pixels_rgb.reshape(1, -1, 3).astype(np.uint8)
        pixels_lab = cv2.cvtColor(pixels_reshaped, cv2.COLOR_RGB2LAB)
        pixels_lab = pixels_lab.reshape(-1, 3)
        
        # Compute median LAB value (more robust than mean)
        median_L = np.median(pixels_lab[:, 0])
        median_a = np.median(pixels_lab[:, 1])
        median_b = np.median(pixels_lab[:, 2])
        
        tone_vector_lab = {
            "L": float(median_L),
            "a": float(median_a),
            "b": float(median_b)
        }
        
        # Convert median LAB back to RGB
        median_lab_array = np.array([[[median_L, median_a, median_b]]], dtype=np.uint8)
        median_rgb_array = cv2.cvtColor(median_lab_array, cv2.COLOR_LAB2RGB)
        median_rgb = median_rgb_array[0, 0]
        
        tone_vector_rgb = {
            "R": int(median_rgb[0]),
            "G": int(median_rgb[1]),
            "B": int(median_rgb[2])
        }
        
        print(f"Dominant tone (LAB): L={tone_vector_lab['L']:.1f}, "
              f"a={tone_vector_lab['a']:.1f}, b={tone_vector_lab['b']:.1f}")
        print(f"Dominant tone (RGB): R={tone_vector_rgb['R']}, "
              f"G={tone_vector_rgb['G']}, B={tone_vector_rgb['B']}")
        
        # Check for background contamination
        self._check_contamination(clustering_result)
        
        return {
            "dominant_cluster": dominant_cluster_idx,
            "tone_vector_lab": tone_vector_lab,
            "tone_vector_rgb": tone_vector_rgb,
            "pixel_count": dominant_pixel_count
        }
    
    def _check_contamination(self, clustering_result):
        """Check for potential background contamination"""
        centroids_rgb = clustering_result["centroids_rgb"]
        
        for i, centroid in enumerate(centroids_rgb):
            r, g, b = centroid
            
            # Check for extreme green (background contamination)
            if g > r * 1.3 and g > b * 1.3:
                print(f"\n[WARNING] Cluster {i} shows green contamination: "
                      f"RGB({int(r)}, {int(g)}, {int(b)})")
                print("Background contamination detected. Verify mask.")
            
            # Check for extreme values that don't match skin
            if r < 50 or (r < 80 and g < 80 and b < 80):
                print(f"\n[WARNING] Cluster {i} shows unusual darkness: "
                      f"RGB({int(r)}, {int(g)}, {int(b)})")
                print("Possible non-skin contamination. Verify mask.")


if __name__ == "__main__":
    print("ColorExtraction module - use via Orchestrator")
