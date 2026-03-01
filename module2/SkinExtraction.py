import sys
import os
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

# Add Module1 to path
sys.path.append("../module1")

from module1 import CameraModule
from module1 import FaceCropping
from module1 import SkinMasking


class SkinExtraction:

    def __init__(self):

        self.ModelPath = "model/SkinTypeModel.h5"
        self.ImageSize = 224

        if not os.path.exists(self.ModelPath):
            raise FileNotFoundError("SkinTypeModel.h5 not found. Train model first.")

        self.ModelSkin = load_model(self.ModelPath)
        self.ClassNames = ["dry", "normal", "oily"]

        os.makedirs("model", exist_ok=True)

    # ==============================
    # Preprocessing
    # ==============================

    def PreprocessImage(self, Image):

        ImageResized = cv2.resize(Image, (self.ImageSize, self.ImageSize))
        ImageNormalized = ImageResized / 255.0
        ImageExpanded = np.expand_dims(ImageNormalized, axis=0)

        return ImageExpanded

    # ==============================
    # Main Prediction Pipeline
    # ==============================

    def PredictSkinType(self):

        print("Starting Camera Module...")

        Camera = CameraModule()
        CaptureResult = Camera.run()

        if not CaptureResult["success"]:
            print("Face capture failed.")
            return None

        Image = CaptureResult["image"]
        BBox = CaptureResult["bbox"]

        # ------------------------------
        # Face Cropping
        # ------------------------------

        Cropper = FaceCropping()
        CropResult = Cropper.crop_to_face(Image, BBox)
        CroppedFace = CropResult["cropped_image"]

        # ------------------------------
        # Skin Masking
        # ------------------------------

        Masker = SkinMasking()

        try:
            MaskResult = Masker.create_skin_mask(CroppedFace)
        except ValueError as Error:
            print("Skin masking failed:", Error)
            return None

        MaskedImage = MaskResult["masked_image"]

        # ------------------------------
        # Model Prediction
        # ------------------------------

        InputTensor = self.PreprocessImage(MaskedImage)

        Prediction = self.ModelSkin.predict(InputTensor)
        Probabilities = Prediction[0]

        PredictedIndex = np.argmax(Probabilities)
        Confidence = float(np.max(Probabilities))

        PredictedSkinType = self.ClassNames[PredictedIndex]

        print("\n=================================")
        print("Predicted Skin Type:", PredictedSkinType)
        print("Confidence:", round(Confidence, 4))
        print("=================================\n")

        self.GenerateDebugGraph(Probabilities)

        return PredictedSkinType, Confidence

    # ==============================
    # Debug Visualization
    # ==============================

    def GenerateDebugGraph(self, Probabilities):

        sns.set_theme(style="whitegrid", context="talk")

        plt.figure(figsize=(8, 6))
        sns.barplot(x=self.ClassNames, y=Probabilities)

        plt.title("Skin Type Prediction Probabilities")
        plt.ylabel("Probability")
        plt.ylim(0, 1)

        plt.savefig("model/PredictionDebug.png")
        plt.close()

        print("Saved prediction probability graph to model/PredictionDebug.png")


# ==============================
# Run
# ==============================

if __name__ == "__main__":

    Extractor = SkinExtraction()
    Extractor.PredictSkinType()