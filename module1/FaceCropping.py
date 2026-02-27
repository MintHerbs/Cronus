"""
FaceCropping.py
Crops image strictly to detected face bounding box
"""

import cv2
import numpy as np


class FaceCropping:
    def __init__(self):
        """Initialize face cropping module"""
        pass
    
    def crop_to_face(self, image, bbox):
        """
        Crop image to face bounding box
        
        Args:
            image: numpy array in BGR format
            bbox: tuple (x_min, y_min, x_max, y_max)
        
        Returns:
            dict with cropped_image and bbox_info
        """
        if bbox is None:
            raise ValueError("No bounding box provided for cropping")
        
        x_min, y_min, x_max, y_max = bbox
        
        # Validate bounding box
        h, w = image.shape[:2]
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)
        
        if x_max <= x_min or y_max <= y_min:
            raise ValueError("Invalid bounding box dimensions")
        
        # Crop to face
        cropped_image = image[y_min:y_max, x_min:x_max].copy()
        
        print(f"Face cropped: {cropped_image.shape[1]}x{cropped_image.shape[0]} pixels")
        
        return {
            "cropped_image": cropped_image,
            "bbox_info": {
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "width": x_max - x_min,
                "height": y_max - y_min
            }
        }


if __name__ == "__main__":
    print("FaceCropping module - use via Orchestrator")
