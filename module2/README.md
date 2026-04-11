# Module 2: Synthetic Skin Tone Dataset Generator

Run the entire pipeline with one command from the project root:

```bash
python Orchestrator_dataset.py
```

This executes 2 modules in order and optionally generates health plots in `module2/output/dataset_health/`.

---

## Project Structure

```
project/
├── Orchestrator_dataset.py
├── output/
│   ├── synthetic_skin_tones.csv          ← intermediate file
│   ├── skin_profiles_with_contrast.csv   ← final dataset
│   └── ... (visualization files)
└── module2/
    ├── SkinToneGenerator.py
    ├── ContrastCalculator.py
    └── output/
        └── dataset_health/
            ├── health_02_lab_distributions.png
            ├── health_08_correlation_matrix.png
            └── health_report.txt
```

---

## Pipeline Modules

Each module does one job, saves its output, and the next module picks up from there.

| # | Module | What It Does | Adds to Dataset |
|---|--------|-------------|-----------------|
| 1 | `SkinToneGenerator.py` | Samples L*, a*, b* values within the documented range of human skin. Maps to Monk Skin Tone class (1–10) and derives undertone from the a*/b* ratio. | `L, a, b, MST_Class, Undertone` |
| 2 | `ContrastCalculator.py` | Calculates Delta E (perceptual distance) between each skin tone and a reference lip colour in LAB space. Bins the result into Low / Medium / High contrast using equal-sized quantiles. | `Delta_E, Contrast_Level` |

---

## Dataset Columns

### Final Dataset Features
| Column | Type | Description |
|--------|------|-------------|
| `L` | float | CIE LAB lightness (30–85) |
| `a` | float | CIE LAB red-green axis (5–25) |
| `b` | float | CIE LAB yellow-blue axis (8–35) |
| `MST_Class` | int | Monk Skin Tone class, 1 = darkest, 10 = lightest |
| `Undertone` | category | Warm / Cool / Neutral |
| `Delta_E` | float | Perceptual distance from reference lip colour |
| `Contrast_Level` | category | Low / Medium / High |

---

## Dataset Health Plots

When graphics generation is enabled, plots are saved to `module2/output/dataset_health/`.

| Plot | What It Shows | ✅ Healthy | ❌ Problem |
|------|--------------|-----------|-----------|
| `health_02_lab_distributions` | KDE curves of L*, a*, b* by undertone | Distinct but overlapping humps per undertone | Flat or collapsed lines |
| `health_08_correlation_matrix` | Pearson correlation heatmap of numeric columns | L* and MST_Class strongly correlated, no unexpected perfect correlations | Perfect collinearity between unexpected pairs |

A `health_report.txt` and `health_report.json` summary are also generated in the same folder.

---

## Loading for ML Training

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

df = pd.read_csv('output/skin_profiles_with_contrast.csv')

# Features
X = df[['L', 'a', 'b', 'MST_Class', 'Undertone', 'Delta_E', 'Contrast_Level']]

# Encode categoricals and normalise LAB values before KNN
```

> **Note:** Always encode categorical columns (`Undertone`, `Contrast_Level`) and normalise `L*`, `a*`, `b*` to 0–1 range before fitting KNN — otherwise lightness will dominate the distance calculation.

---

## Intermediate Files

The intermediate CSV `synthetic_skin_tones.csv` in `output/` is a hand-off file between modules. It is safe to delete after a successful run. Keep it during development so you can rerun ContrastCalculator without regenerating everything.

If a module crashes, fix it and rerun it directly:
```bash
python module2/ContrastCalculator.py
```
