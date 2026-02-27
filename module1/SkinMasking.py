"""
SkinMasking.py
Generates binary skin mask to isolate epidermal pixels
Removes hair, eyes, lips, eyebrows, and background
"""

import cv2
import numpy as np
import os


class SkinMasking:
    def __init__(self, min_skin_pixels=1000):
        """
        Initialize skin masking module
        
        Args:
            min_skin_pixels: Minimum required skin pixels (default 1000)
        """
        self.min_skin_pixels = min_skin_pixels
        self.output_path = "module1/output"
        
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
    
    def create_skin_mask(self, image):
        """
        Create binary skin mask using YCrCb color space
        
        Args:
            image: numpy array in BGR format (cropped face)
        
        Returns:
            dict with mask, masked_image, skin_pixels_rgb, pixel_count
        """
        # Convert to YCrCb color space (better for skin detection)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        # Define skin color range in YCrCb
        # Y: 0-255, Cr: 133-173, Cb: 77-127
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        
        # Create initial mask
        mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        
        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Apply Gaussian blur to smooth edges
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Threshold to binary
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Count skin pixels
        skin_pixel_count = np.count_nonzero(mask)
        
        # Validate sufficient skin pixels
        if skin_pixel_count < self.min_skin_pixels:
            raise ValueError(
                f"Insufficient skin pixels detected: {skin_pixel_count} "
                f"(minimum required: {self.min_skin_pixels})"
            )
        
        # Extract skin pixels in RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        skin_pixels_rgb = image_rgb[mask > 0]
        
        # Create masked visualization
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        
        # Save mask visualization
        self._save_mask_visualization(image, mask, masked_image)
        
        print(f"Skin mask created: {skin_pixel_count:,} skin pixels extracted")
        
        return {
            "mask": mask,
            "masked_image": masked_image,
            "skin_pixels_rgb": skin_pixels_rgb,
            "pixel_count": skin_pixel_count
        }
    
    def _save_mask_visualization(self, original, mask, masked):
        """Save mask visualization for debugging"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        sns.set_theme(style="whitegrid", context="talk")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)
        
        # Original cropped face
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Cropped Face")
        axes[0].axis('off')
        
        # Binary mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title("Skin Mask")
        axes[1].axis('off')
        
        # Masked result
        axes[2].imshow(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
        axes[2].set_title("Masked Skin Pixels")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/skin_mask_visualization.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {self.output_path}/skin_mask_visualization.png")


if __name__ == "__main__":
    print("SkinMasking module - use via Orchestrator")
