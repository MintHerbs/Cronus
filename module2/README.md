# 💄 Lipstick AI — Dataset Generation Pipeline

> **Your final training dataset is at:**
> ```
> module2/output/final_skin_tone_dataset.csv
> ```
> A Parquet version lives at the same path with `.parquet` extension. Both contain identical data.

---

## 🚀 Quick Start

Run the entire pipeline with one command from the project root:

```bash
python orchestrator.py
```

This executes all 7 modules in order and generates 8 health plots in `module2/dataset_health/`.

---

## 🗂️ Project Structure

```
project/
├── orchestrator.py
└── module2/
    ├── SkinToneGenerator.py
    ├── SkinTypesGenerator.py
    ├── ContrastCalculator.py
    ├── ShadeRule.py
    ├── LightingAugmentator.py
    ├── EdgeCaseDetector.py
    ├── DatasetAssembler.py
    ├── output/
    │   ├── final_skin_tone_dataset.csv      ← use this for training
    │   ├── final_skin_tone_dataset.parquet
    │   └── ... (intermediate files)
    └── dataset_health/
        ├── health_01_class_balance.png
        ├── health_02_lab_distributions.png
        └── ... (8 plots + reports)
```

---

## ⚙️ Pipeline Modules

Each module does one job, saves its output, and the next module picks up from there.

| # | Module | What It Does | Adds to Dataset |
|---|--------|-------------|-----------------|
| 1 | `SkinToneGenerator.py` | Samples L\*, a\*, b\* values within the documented range of human skin. Maps to Monk Skin Tone class (1–10) and derives undertone from the a\*/b\* ratio. | `L, a, b, MST_Class, Undertone` |
| 2 | `SkinTypesGenerator.py` | Assigns skin type using realistic population weights (Normal 30%, Oily 25%, Dry 25%, Combination 15%, Sensitive 5%). Assigns texture using a conditional probability table per skin type. | `Skin_Type, Texture_Descriptor` |
| 3 | `ContrastCalculator.py` | Calculates Delta E (perceptual distance) between each skin tone and a reference lip colour in LAB space. Bins the result into Low / Medium / High contrast using equal-sized quantiles. | `Delta_E, Contrast_Level` |
| 4 | `ShadeRule.py` | The rule engine. Undertone drives the shade family, L\* depth selects the specific shade within it. Skin type drives the finish recommendation. Harmony score is computed from contrast–shade intensity match. | `Primary_Shade, Sub_Shades, Recommended_Finish, Harmony_Score` |
| 5 | `LightingAugmentator.py` | Triples the dataset by creating 3 lighting variants per profile — Outdoor Daylight (baseline), Indoor Warm (+5 b\*, −3 L\*), Office Fluorescent (−4 b\*, +2 L\*) — with Gaussian noise (σ=1). | `Lighting_Context, Profile_ID` |
| 6 | `EdgeCaseDetector.py` | Injects ~15% extra profiles in difficult boundary zones — Neutral Ambiguity (a\*/b\* ≈ 1.0), Extreme Lightness (L\* < 32 or > 83), and Conflict Profiles (high a\* and b\* simultaneously). | `Profile_Type, Edge_Case_Category` |
| 7 | `DatasetAssembler.py` | Final cleanup — deduplicates, clips out-of-range LAB values, enforces types, injects metadata, shuffles rows, and exports the finished dataset. | `Generation_Method, Rule_Confidence` |

---

## 📊 Dataset Columns

### Input Features *(what goes into the model)*
| Column | Type | Description |
|--------|------|-------------|
| `L` | float | CIE LAB lightness (30–85) |
| `a` | float | CIE LAB red-green axis (5–25) |
| `b` | float | CIE LAB yellow-blue axis (8–35) |
| `MST_Class` | int | Monk Skin Tone class, 1 = darkest, 10 = lightest |
| `Undertone` | category | Warm / Cool / Neutral |
| `Skin_Type` | category | Normal / Oily / Dry / Combination / Sensitive |
| `Texture_Descriptor` | category | Smooth / Shiny / Rough / Uneven |
| `Delta_E` | float | Perceptual distance from reference lip colour |
| `Contrast_Level` | category | Low / Medium / High |
| `Lighting_Context` | category | Outdoor Daylight / Indoor Warm / Office Fluorescent |

### Output Labels *(what the model predicts)*
| Column | Type | Description |
|--------|------|-------------|
| `Primary_Shade` | category | Recommended lipstick shade (e.g. Berry, Coral, Taupe) |
| `Sub_Shades` | string | All compatible shades in the same undertone family |
| `Recommended_Finish` | category | Matte / Dewy / Cream / Satin / Natural / Long-wear |
| `Harmony_Score` | float | Strength of the skin–shade match, 0.0–1.0 |

### Metadata *(drop before training)*
| Column | Description |
|--------|-------------|
| `Profile_Type` | `standard` or `edge_case` |
| `Edge_Case_Category` | None / Neutral Ambiguity / Extreme Lightness / Conflict Profile |
| `Generation_Method` | synthetic_standard / synthetic_augmented / synthetic_edge_case |
| `Rule_Confidence` | 1.0 / 0.8 / 0.5 — how cleanly the profile fits the rules |
| `Profile_ID` | Links augmented rows back to their original pre-lighting profile |

---

## 🩺 Dataset Health Plots

All plots are saved to `module2/dataset_health/` after each orchestrator run.

| Plot | What It Shows | ✅ Healthy | ❌ Problem |
|------|--------------|-----------|-----------|
| `health_01_class_balance` | Bar charts of shade group and skin type counts | Roughly even bars | One class dominates |
| `health_02_lab_distributions` | KDE curves of L\*, a\*, b\* by undertone | Distinct but overlapping humps per undertone | Flat or collapsed lines |
| `health_03_harmony_scores` | Harmony score histogram + boxplot by shade | Most scores above 0.7, different median per shade | All shade boxes look identical |
| `health_04_mst_undertone_heatmap` | MST Class × Undertone distribution (%) | All three undertones visible in every row | Any row is 100% one colour |
| `health_05_lighting_delta` | Violin plot of b\* by lighting context | Warm > Daylight > Fluorescent on b\* axis | All three violins perfectly overlap |
| `health_06_edge_case_coverage` | Pie chart + a\*–b\* scatter of edge cases | Red dots cluster near a\*/b\* = 1 diagonal and extremes | Red dots randomly scattered |
| `health_07_schema_integrity` | Null count per column + data type summary | Every null bar is zero (green) | Any red bar present |
| `health_08_correlation_matrix` | Pearson correlation heatmap of numeric columns | L\* and MST_Class strongly correlated, no off-diagonal ±1.0 | Perfect collinearity between unexpected pairs |

A `health_report.txt` and `health_report.json` summary are also generated in the same folder.

---

## 🧠 Loading for ML Training

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

df = pd.read_csv('module2/output/final_skin_tone_dataset.csv')

# Drop metadata columns
meta_cols = ['Profile_Type', 'Edge_Case_Category', 'Generation_Method',
             'Rule_Confidence', 'Profile_ID', 'Sub_Shades']
df = df.drop(columns=meta_cols)

# Features and label
X = df.drop(columns=['Primary_Shade', 'Recommended_Finish', 'Harmony_Score'])
y = df['Primary_Shade']

# Encode categoricals and normalise LAB values before KNN
```

> **Note:** Always encode categorical columns (`Undertone`, `Skin_Type`, etc.) and normalise `L*`, `a*`, `b*` to 0–1 range before fitting KNN — otherwise lightness will dominate the distance calculation.

---

## 🗃️ Intermediate Files

The 6 intermediate CSVs in `module2/output/` are hand-off files between modules. They are safe to delete after a successful run. Keep them during development so you can rerun a single broken module without regenerating everything.

If a module crashes, fix it and rerun it directly:
```bash
python module2/ShadeRule.py
```

---

## 📦 .gitignore Recommendation

```gitignore
module2/output/
module2/dataset_health/
__pycache__/
*.pyc
.DS_Store
```

Push the **code**, not the generated files. Anyone cloning the repo can reproduce the dataset by running `python orchestrator.py`.
