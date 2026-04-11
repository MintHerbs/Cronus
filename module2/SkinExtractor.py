"""
module2/SkinExtractor.py

Extracts skin type from a live webcam capture and produces seaborn
charts describing the skin data itself.

Pipeline:
    CameraModule → FaceCropping → SkinMasking → CNN predict → charts

Charts saved to module2/output/:
    chart_01_skin_type_prediction.png   bar chart  – dry/normal/oily probabilities
    chart_02_lab_distributions.png      KDE        – L* a* b* spread of skin pixels
    chart_03_undertone_profile.png      bar chart  – warm / cool / neutral score
    chart_04_skin_tone_scale.png        swatches   – position on Monk Skin Tone 1-10
    chart_05_dominant_colors.png        swatches   – top 5 dominant skin pixel colors

Usage (from project root):
    python module2/SkinExtractor.py
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans
from tensorflow.keras.models import load_model

# ── Path setup ────────────────────────────────────────────────────────────────
CurrentDir  = os.path.dirname(os.path.abspath(__file__))
ProjectRoot = os.path.abspath(os.path.join(CurrentDir, ".."))
sys.path.insert(0, ProjectRoot)

from module1.CameraModule import CameraModule
from module1.FaceCropping import FaceCropping
from module1.SkinMasking  import SkinMasking

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join(ProjectRoot, "model", "SkinTypeModel.h5")
OUTPUT_PATH = os.path.join(CurrentDir, "output")
IMAGE_SIZE  = 224

# Must match folder names in dataset/Oily-Dry-Skin-Types/train/ (alphabetical)
CLASS_NAMES = ["dry", "normal", "oily"]

CLASS_INFO  = {
    "dry":    "Low sebum · micro-lines visible · dull finish",
    "normal": "Balanced sebum · smooth texture · even finish",
    "oily":   "High sebum · large pores · glossy finish",
}

CLASS_COLORS = {
    "dry":    "#3498db",
    "normal": "#2ecc71",
    "oily":   "#e67e22",
}

sns.set_theme(style="whitegrid", context="talk")


# ─────────────────────────────────────────────────────────────────────────────
class SkinExtractor:
# ─────────────────────────────────────────────────────────────────────────────

    def __init__(self):
        os.makedirs(OUTPUT_PATH, exist_ok=True)

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"\n[ERROR] Model not found at:\n  {MODEL_PATH}\n\n"
                "Train it first:\n"
                "  python module2/TrainSkinType.py\n"
            )

        print(f"Loading model: {MODEL_PATH}")
        self.Model = load_model(MODEL_PATH)
        print("Model ready.\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Pipeline
    # ─────────────────────────────────────────────────────────────────────────

    def Run(self):
        print("=" * 60)
        print("MODULE 2  —  Skin Type Extraction")
        print("=" * 60)

        # 1. Capture
        print("\n[1/4] Opening camera…")
        Capture = CameraModule().run()
        if not Capture["success"]:
            print("[ERROR] Camera capture failed.")
            return None
        RawImage = Capture["image"]
        BBox     = Capture["bbox"]
        print("[OK]   Face captured.")

        # 2. Crop
        print("[2/4] Cropping face…")
        CroppedFace = FaceCropping().crop_to_face(RawImage, BBox)["cropped_image"]
        print("[OK]   Face cropped.")

        # 3. Mask
        print("[3/4] Applying skin mask…")
        try:
            MaskResult = SkinMasking().create_skin_mask(CroppedFace)
        except ValueError as Err:
            print(f"[ERROR] Skin masking failed: {Err}")
            return None

        MaskedImage   = MaskResult["masked_image"]
        SkinPixelsRGB = MaskResult["skin_pixels_rgb"]   # (N, 3) uint8
        print("[OK]   Skin mask applied.")

        # 4. Predict skin type
        print("[4/4] Classifying skin type…")
        Result = self._Predict(MaskedImage)

        # 5. Derive skin analytics from extracted pixels
        SkinLabDF       = self._PixelsToLabDF(SkinPixelsRGB)
        UndertoneScores = self._ComputeUndertone(SkinLabDF)
        MedianL         = float(SkinLabDF["L"].median())
        MSTClass        = self._LToMST(MedianL)
        DominantColors  = self._DominantColors(SkinPixelsRGB, k=5)

        print("\n" + "=" * 60)
        print(f"  Skin type  :  {Result['skin_type'].upper()}")
        print(f"  Confidence :  {Result['confidence'] * 100:.1f}%")
        print(f"  Median L*  :  {MedianL:.1f}  →  MST class {MSTClass}")
        print(f"  Undertone  :  {max(UndertoneScores, key=UndertoneScores.get).upper()}")
        print("=" * 60)

        # 6. Generate all charts
        print("\nGenerating skin charts…")
        self._ChartSkinTypePrediction(Result)
        self._ChartLabDistributions(SkinLabDF)
        self._ChartUndertoneProfile(UndertoneScores)
        self._ChartSkinToneScale(MedianL, MSTClass)
        self._ChartDominantColors(DominantColors)
        print(f"\nAll charts saved → {OUTPUT_PATH}/")

        return {
            **Result,
            "median_L":         MedianL,
            "mst_class":        MSTClass,
            "undertone":        max(UndertoneScores, key=UndertoneScores.get),
            "undertone_scores": UndertoneScores,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _Predict(self, ImageBGR):
        Rgb    = cv2.cvtColor(ImageBGR, cv2.COLOR_BGR2RGB)
        Tensor = np.expand_dims(
            cv2.resize(Rgb, (IMAGE_SIZE, IMAGE_SIZE)) / 255.0, 0
        )
        Probs = self.Model.predict(Tensor, verbose=0)[0]
        Idx   = int(np.argmax(Probs))
        return {
            "skin_type":       CLASS_NAMES[Idx],
            "confidence":      float(Probs[Idx]),
            "probabilities":   Probs.tolist(),
            "predicted_index": Idx,
        }

    def _PixelsToLabDF(self, SkinPixelsRGB):
        """(N,3) uint8 RGB → DataFrame of standard CIE L* a* b* values."""
        Reshaped = SkinPixelsRGB.reshape(1, -1, 3).astype(np.uint8)
        Lab      = cv2.cvtColor(Reshaped, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(float)
        # OpenCV encodes LAB as: L=[0,255], a/b=[0,255] centred at 128
        # Convert to standard: L=[0,100], a/b=[-128,127]
        Lab[:, 0] = Lab[:, 0] * 100.0 / 255.0
        Lab[:, 1] = Lab[:, 1] - 128.0
        Lab[:, 2] = Lab[:, 2] - 128.0
        return pd.DataFrame(Lab, columns=["L", "a", "b"])

    def _ComputeUndertone(self, LabDF):
        """
        Warm  = high b* (yellow bias)
        Cool  = a* dominates over b* (pink/red bias)
        Neutral = balanced a*/b*
        Returns dict of percentage scores summing to 100.
        """
        MedianA = float(LabDF["a"].median())
        MedianB = float(LabDF["b"].median())

        WarmScore    = float(np.clip(MedianB / 20.0, 0, 1))
        CoolScore    = float(np.clip((MedianA - MedianB) / 20.0, 0, 1))
        NeutralScore = max(0.0, 1.0 - WarmScore - CoolScore)
        Total        = WarmScore + CoolScore + NeutralScore + 1e-9

        return {
            "Warm":    round(WarmScore    / Total * 100, 1),
            "Cool":    round(CoolScore    / Total * 100, 1),
            "Neutral": round(NeutralScore / Total * 100, 1),
        }

    def _LToMST(self, L):
        """Map median L* (0-100) to Monk Skin Tone class 1-10."""
        Bins = np.linspace(30, 85, 11)
        Idx  = int(np.searchsorted(Bins, L, side="right")) - 1
        return int(np.clip(Idx + 1, 1, 10))

    def _DominantColors(self, SkinPixelsRGB, k=5):
        """K-Means on skin pixels → top k dominant RGB colors sorted by prevalence."""
        Sample = SkinPixelsRGB
        if len(Sample) > 5000:
            Idx    = np.random.choice(len(Sample), 5000, replace=False)
            Sample = Sample[Idx]
        KM = KMeans(n_clusters=k, random_state=42, n_init=10)
        KM.fit(Sample)
        _, Counts = np.unique(KM.labels_, return_counts=True)
        Order    = np.argsort(-Counts)
        Centers  = KM.cluster_centers_[Order].astype(int)
        Percents = Counts[Order] / Counts.sum() * 100
        return list(zip(Centers, Percents))

    # ─────────────────────────────────────────────────────────────────────────
    # Charts  —  all about the SKIN DATA
    # ─────────────────────────────────────────────────────────────────────────

    def _ChartSkinTypePrediction(self, Result):
        """
        Chart 1 — Horizontal bar chart of dry/normal/oily probabilities.
        Each bar coloured by class; winning class annotated in bold.
        """
        Probs  = Result["probabilities"]
        Pi     = Result["predicted_index"]
        Colors = [CLASS_COLORS[C] for C in CLASS_NAMES]

        Fig, Ax = plt.subplots(figsize=(10, 4), dpi=150)
        Bars    = Ax.barh(CLASS_NAMES, Probs, color=Colors,
                          height=0.5, edgecolor="white", linewidth=0.8)

        for Bar, P, Cls in zip(Bars, Probs, CLASS_NAMES):
            IsWinner = CLASS_NAMES.index(Cls) == Pi
            Ax.text(
                min(P + 0.02, 0.90),
                Bar.get_y() + Bar.get_height() / 2,
                f"{P * 100:.1f}%",
                va="center", fontsize=12,
                fontweight="bold" if IsWinner else "normal",
            )
            Ax.text(
                1.02,
                Bar.get_y() + Bar.get_height() / 2,
                CLASS_INFO[Cls],
                va="center", fontsize=8.5, color="#555555",
            )

        Ax.set_xlim(0, 1.55)
        Ax.set_xlabel("Probability")
        Ax.set_title(
            f"Skin type: {Result['skin_type'].upper()}  "
            f"({Result['confidence'] * 100:.1f}% confidence)",
            fontsize=13,
        )
        plt.tight_layout()
        self._Save("chart_01_skin_type_prediction")

    def _ChartLabDistributions(self, LabDF):
        """
        Chart 2 — KDE of L* a* b* across all extracted skin pixels.
        The spread of these curves shows how consistent / varied the skin tone is.
        A sharp L* peak = even skin tone. A wide a* curve = mixed undertone.
        """
        Fig, Axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

        Channels = [
            ("L", "#c0392b", "L*  (Lightness  0–100)",   (0,   100)),
            ("a", "#27ae60", "a*  (Green ← · → Red)",    (-30,  30)),
            ("b", "#2980b9", "b*  (Blue ← · → Yellow)", (-10,  40)),
        ]

        for Ax, (Col, Color, Label, XLim) in zip(Axes, Channels):
            sns.kdeplot(
                data=LabDF, x=Col, ax=Ax,
                fill=True, color=Color, alpha=0.35, linewidth=2.5,
            )
            Median = LabDF[Col].median()
            Mean   = LabDF[Col].mean()
            Ax.axvline(Median, color=Color, linestyle="--",
                       linewidth=1.8, label=f"Median {Median:.1f}")
            Ax.axvline(Mean,   color=Color, linestyle=":",
                       linewidth=1.4, alpha=0.7, label=f"Mean {Mean:.1f}")
            Ax.set_title(Label, fontsize=12)
            Ax.set_xlim(XLim)
            Ax.set_xlabel("")
            Ax.legend(fontsize=9)

        Fig.suptitle(
            "Skin Pixel LAB Distributions  —  based on extracted skin pixels",
            fontsize=13,
        )
        plt.tight_layout()
        self._Save("chart_02_lab_distributions")

    def _ChartUndertoneProfile(self, Scores):
        """
        Chart 3 — Warm / Cool / Neutral undertone scores as a bar chart.
        Derived from the median a* and b* of the skin pixels.
        Warm = high b* (yellow). Cool = a* > b* (pink). Neutral = balanced.
        """
        UndertoneColors = {
            "Warm":    "#e8a87c",
            "Cool":    "#7eb8d4",
            "Neutral": "#b0a898",
        }

        Names   = list(Scores.keys())
        Values  = list(Scores.values())
        Colors  = [UndertoneColors[N] for N in Names]
        Dominant = max(Scores, key=Scores.get)

        Fig, Ax = plt.subplots(figsize=(7, 5), dpi=150)
        Bars    = sns.barplot(x=Names, y=Values, palette=Colors, ax=Ax)

        for Bar, V in zip(Bars.patches, Values):
            Ax.text(
                Bar.get_x() + Bar.get_width() / 2,
                Bar.get_height() + 0.8,
                f"{V:.1f}%",
                ha="center", fontsize=12, fontweight="bold",
            )

        Ax.set_title(
            f"Undertone Profile  —  Dominant: {Dominant.upper()}",
            fontsize=13,
        )
        Ax.set_xlabel("Undertone")
        Ax.set_ylabel("Score (%)")
        Ax.set_ylim(0, max(Values) * 1.25)
        plt.tight_layout()
        self._Save("chart_03_undertone_profile")

    def _ChartSkinToneScale(self, MedianL, MSTClass):
        """
        Chart 4 — Monk Skin Tone scale (1–10) with the detected class highlighted.
        Each swatch is coloured with a representative skin tone.
        An arrow marks where this person's skin falls.
        """
        # Representative RGB per MST class (1 = darkest, 10 = lightest)
        MSTSwatches = [
            (34,  22,  14),
            (53,  35,  22),
            (75,  50,  33),
            (100, 68,  46),
            (131, 91,  63),
            (160, 116, 85),
            (186, 148, 117),
            (210, 180, 153),
            (228, 207, 186),
            (243, 230, 214),
        ]

        Fig, Ax = plt.subplots(figsize=(13, 3.5), dpi=150)
        Fig.patch.set_facecolor("white")

        for i, (R, G, B) in enumerate(MSTSwatches):
            ClassNum  = i + 1
            IsActive  = (ClassNum == MSTClass)
            EdgeColor = "#111111" if IsActive else "#cccccc"
            LineWidth = 3.5       if IsActive else 0.8

            Rect = mpatches.FancyBboxPatch(
                (i * 1.1, 0.2), 0.95, 1.2,
                boxstyle="round,pad=0.05",
                facecolor=(R / 255, G / 255, B / 255),
                edgecolor=EdgeColor,
                linewidth=LineWidth,
            )
            Ax.add_patch(Rect)

            TextColor = "white" if i < 6 else "#333333"
            Ax.text(
                i * 1.1 + 0.475, 0.82,
                str(ClassNum),
                ha="center", va="center",
                fontsize=10, color=TextColor,
                fontweight="bold" if IsActive else "normal",
            )

            if IsActive:
                Ax.annotate(
                    f"Detected  (L*={MedianL:.0f})",
                    xy=(i * 1.1 + 0.475, 1.4),
                    xytext=(i * 1.1 + 0.475, 1.95),
                    ha="center", fontsize=10,
                    arrowprops=dict(arrowstyle="->",
                                   color="#111111", lw=1.8),
                    color="#111111",
                )

        Ax.set_xlim(-0.3, 11.2)
        Ax.set_ylim(0, 2.5)
        Ax.axis("off")
        Ax.set_title(
            f"Monk Skin Tone Scale  —  Class {MSTClass} detected  "
            f"(1 = darkest · 10 = lightest)",
            fontsize=13, pad=8,
        )
        plt.tight_layout()
        self._Save("chart_04_skin_tone_scale")

    def _ChartDominantColors(self, DominantColors):
        """
        Chart 5 — Top 5 dominant skin pixel colors from K-Means (k=5).
        Each swatch shows the HEX, RGB, and percentage of skin pixels it represents.
        """
        k = len(DominantColors)

        Fig, Axes = plt.subplots(1, k, figsize=(13, 4), dpi=150)
        Fig.suptitle(
            "Top 5 Dominant Skin Pixel Colors  (K-Means, k=5)",
            fontsize=13, y=1.03,
        )

        for Ax, (RGB, Pct) in zip(Axes, DominantColors):
            R, G, B   = int(RGB[0]), int(RGB[1]), int(RGB[2])
            HexColor  = f"#{R:02x}{G:02x}{B:02x}"
            Luminance = R * 0.299 + G * 0.587 + B * 0.114
            TextColor = "white" if Luminance < 128 else "#333333"

            Ax.set_facecolor(HexColor)
            Ax.text(0.5, 0.65, HexColor.upper(),
                    ha="center", va="center", fontsize=11,
                    color=TextColor, fontweight="bold",
                    transform=Ax.transAxes)
            Ax.text(0.5, 0.45, f"RGB({R}, {G}, {B})",
                    ha="center", va="center", fontsize=9,
                    color=TextColor, transform=Ax.transAxes)
            Ax.text(0.5, 0.25, f"{Pct:.1f}% of pixels",
                    ha="center", va="center", fontsize=9,
                    color=TextColor, transform=Ax.transAxes)
            Ax.set_xticks([])
            Ax.set_yticks([])
            for Spine in Ax.spines.values():
                Spine.set_edgecolor("#cccccc")
                Spine.set_linewidth(0.8)

        plt.tight_layout()
        self._Save("chart_05_dominant_colors")

    # ── Helper ────────────────────────────────────────────────────────────────

    @staticmethod
    def _Save(Name):
        Path = os.path.join(OUTPUT_PATH, f"{Name}.png")
        plt.savefig(Path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {Path}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    Extractor = SkinExtractor()
    Result    = Extractor.Run()

    if Result:
        print(f"\nReturn values for downstream modules:")
        print(f"  skin_type       = '{Result['skin_type']}'")
        print(f"  undertone       = '{Result['undertone']}'")
        print(f"  mst_class       = {Result['mst_class']}")
        print(f"  confidence      = {Result['confidence']:.4f}")