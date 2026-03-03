"""
Orchestrator.py
Main pipeline controller implementing skin-focused clustering
Pipeline: CameraModule → FaceDetection → FaceCropping → SkinMasking → 
          KMeansSegmentation (skin pixels only) → ColorExtraction → ToneClassification
"""

import sys
import os

# Add module1 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'module1'))

from module1.CameraModule import CameraModule
from module1.FaceCropping import FaceCropping
from module1.SkinMasking import SkinMasking
from module1.KMeansSegmentation import KMeansSegmentation
from module1.ColorExtraction import ColorExtraction
from module1.ToneClassification import ToneClassification


class Orchestrator:
    def __init__(self, k_clusters=3, min_skin_pixels=1000):
        """
        Initialize the orchestrator
        
        Args:
            k_clusters: Number of clusters for K-Means segmentation
            min_skin_pixels: Minimum required skin pixels for masking
        """
        self.k_clusters = k_clusters
        self.min_skin_pixels = min_skin_pixels
    
    def run(self):
        """Execute the full skin-focused clustering pipeline"""
        print("=" * 70)
        print("ORCHESTRATOR: Starting Skin-Focused Clustering Pipeline")
        print("=" * 70)
        
        try:
            # Step 1: Capture image with CameraModule (includes face detection)
            print("\n[STEP 1/6] CameraModule - Capturing aligned face image...")
            camera_module = CameraModule()
            capture_result = camera_module.run()
            
            if not capture_result["success"]:
                print("\n[ERROR] Camera capture failed or was cancelled.")
                print("Pipeline aborted.")
                return False
            
            print("[SUCCESS] Image captured with face detected")
            
            # Step 2: Crop to face bounding box
            print("\n[STEP 2/6] FaceCropping - Removing background pixels...")
            face_cropping = FaceCropping()
            crop_result = face_cropping.crop_to_face(
                image=capture_result["image"],
                bbox=capture_result["bbox"]
            )
            print("[SUCCESS] Face cropped, background eliminated")
            
            # Step 3: Create skin mask
            print("\n[STEP 3/6] SkinMasking - Isolating epidermal pixels...")
            skin_masking = SkinMasking(min_skin_pixels=self.min_skin_pixels)
            mask_result = skin_masking.create_skin_mask(
                image=crop_result["cropped_image"]
            )
            print("[SUCCESS] Skin mask created, non-skin regions removed")
            
            # Step 4: K-Means clustering on skin pixels only
            print(f"\n[STEP 4/6] KMeansSegmentation - Clustering skin pixels in LAB space (K={self.k_clusters})...")
            segmentation = KMeansSegmentation(k=self.k_clusters)
            clustering_result = segmentation.process_skin_pixels(
                skin_pixels_rgb=mask_result["skin_pixels_rgb"]
            )
            print("[SUCCESS] K-Means clustering completed on masked skin pixels")
            
            # Step 5: Extract dominant color
            print("\n[STEP 5/6] ColorExtraction - Extracting dominant skin tone...")
            color_extraction = ColorExtraction()
            color_result = color_extraction.extract_dominant_tone(
                clustering_result=clustering_result,
                skin_pixels_rgb=mask_result["skin_pixels_rgb"]
            )
            print("[SUCCESS] Dominant tone extracted")
            
            # Step 6: Classify undertone
            print("\n[STEP 6/6] ToneClassification - Analyzing undertone...")
            tone_classification = ToneClassification()
            classification_result = tone_classification.classify_undertone(
                tone_vector_lab=color_result["tone_vector_lab"]
            )
            print("[SUCCESS] Undertone classified")
            
            # Display final results
            self._display_results(
                mask_result, clustering_result, color_result, classification_result
            )
            
            return True
            
        except Exception as e:
            print(f"\n[ERROR] Pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _display_results(self, mask_result, clustering_result, color_result, classification_result):
        """Display comprehensive pipeline results"""
        print("\n" + "=" * 70)
        print("PIPELINE RESULTS - SKIN-FOCUSED CLUSTERING")
        print("=" * 70)
        
        print(f"\nSkin Pixels Analyzed: {mask_result['pixel_count']:,}")
        print(f"K-Means Clusters: {clustering_result['k']}")
        
        print("\nCluster Distribution (LAB Space):")
        for i in range(clustering_result['k']):
            centroid_lab = clustering_result['centroids_lab'][i]
            centroid_rgb = clustering_result['centroids_rgb'][i]
            pixel_count = clustering_result['pixel_counts'][i]
            percentage = (pixel_count / mask_result['pixel_count']) * 100
            
            L, a, b = [int(c) for c in centroid_lab]
            R, G, B = [int(c) for c in centroid_rgb]
            
            print(f"  Cluster {i}:")
            print(f"    LAB: ({L:3d}, {a:3d}, {b:3d})")
            print(f"    RGB: ({R:3d}, {G:3d}, {B:3d})")
            print(f"    Pixels: {pixel_count:,} ({percentage:.1f}%)")
        
        print(f"\nDominant Skin Tone:")
        print(f"  Cluster: {color_result['dominant_cluster']}")
        print(f"  LAB: L={color_result['tone_vector_lab']['L']:.1f}, "
              f"a={color_result['tone_vector_lab']['a']:.1f}, "
              f"b={color_result['tone_vector_lab']['b']:.1f}")
        print(f"  RGB: R={color_result['tone_vector_rgb']['R']}, "
              f"G={color_result['tone_vector_rgb']['G']}, "
              f"B={color_result['tone_vector_rgb']['B']}")
        
        print(f"\nUndertone Classification:")
        print(f"  Type: {classification_result['undertone'].upper()}")
        print(f"  Confidence: {classification_result['confidence']:.1f}%")
        print(f"  Chroma Angle: {classification_result['chroma_angle']:.1f}°")
        print(f"  Explanation: {classification_result['explanation']}")
        
        print("\nGenerated Files:")
        print("  - module1/output/capture_[timestamp].jpg")
        print("  - module1/output/skin_mask_visualization.png")
        print("  - module1/output/kmeans_segmented.png")
        print("  - module1/output/kmeans_segmented.svg")
        print("  - module1/output/kmeans_color_distribution.png")
        print("  - module1/output/kmeans_color_distribution.svg")
        
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("Clustering performed on masked skin pixels only")
        print("Background contamination eliminated via skin masking")
        print("=" * 70)


if __name__ == "__main__":
    # Run the orchestrator
    orchestrator = Orchestrator(k_clusters=3)
    success = orchestrator.run()
    
    if not success:
        sys.exit(1)
