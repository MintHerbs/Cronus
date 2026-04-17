"""
TrainShadeModel.py
Trains a Random Forest classifier on the synthetic dataset to predict
primary_group (shade family) from skin LAB + undertone + skin type.

Run from project root:
    pip install scikit-learn pandas numpy joblib
    python TrainShadeModel.py

Output:
    model/rf_shade_model.pkl   ← the trained classifier
    model/rf_label_encoder.pkl ← label encoder for primary_group
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(ROOT, "output", "final_skin_tone_dataset.csv")
MODEL_DIR = os.path.join(ROOT, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "rf_shade_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "rf_label_encoder.pkl")

# ── Undertone / skin_type → numeric mapping ────────────────────────────────────
UNDERTONE_MAP = {"warm": 0, "cool": 1, "neutral": 2, "Warm": 0, "Cool": 1, "Neutral": 2}
SKIN_TYPE_MAP = {"dry": 0, "normal": 1, "oily": 2}
CONTRAST_MAP = {"low": 0, "medium": 1, "high": 2, "Low": 0, "Medium": 1, "High": 2}


def load_dataset():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}\n"
            "Run Orchestrator_dataset.py first to generate it."
        )
    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded {len(df)} rows — columns: {df.columns.tolist()}")
    return df


def prepare_features(df):
    """
    Build feature matrix X and label vector y.

    Features used:
        skin_L, skin_a, skin_b   ← raw LAB values (continuous)
        contrast_level_enc       ← Low/Medium/High encoded as 0/1/2
        normal_pct, oily_pct, dry_pct  ← texture composition (if available)

    Target:
        primary_group  ← Coral / Nude / Red / Berry / Mauve / Pink
    """
    # Encode contrast_level
    df["contrast_enc"] = df["contrast_level"].map(CONTRAST_MAP).fillna(1)

    feature_cols = ["skin_L", "skin_a", "skin_b", "contrast_enc"]

    # Add texture columns if they exist in the dataset
    for col in ["normal_pct", "oily_pct", "dry_pct"]:
        if col in df.columns:
            feature_cols.append(col)

    X = df[feature_cols].values.astype(np.float32)
    y_raw = df["primary_group"].values

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    print(f"Features: {feature_cols}")
    print(f"Classes : {le.classes_.tolist()}")
    print(f"X shape : {X.shape}")

    return X, y, le


def train(X, y):
    """
    Train Random Forest with cross-validation.
    Returns the best fitted model.
    """
    print("\n── Training Random Forest ──────────────────────────────")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,  # grow until pure leaves
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",  # √(n_features) per split
        class_weight="balanced",  # handle any class imbalance
        random_state=42,
        n_jobs=-1,  # use all CPU cores
    )

    # 5-fold cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"5-fold CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Train on full dataset
    model.fit(X, y)

    # Also evaluate on held-out 20% for a quick sanity check
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model_check = RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"
    )
    model_check.fit(X_tr, y_tr)
    y_pred = model_check.predict(X_te)
    print(f"Hold-out accuracy : {accuracy_score(y_te, y_pred):.3f}")

    return model


def main():
    print("=" * 60)
    print("  LipMatch — Random Forest Shade Classifier Training")
    print("=" * 60)

    df = load_dataset()
    X, y, le = prepare_features(df)
    model = train(X, y)

    # Save model and encoder
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print(f"\nModel saved   → {MODEL_PATH}")
    print(f"Encoder saved → {ENCODER_PATH}")

    # Show feature importances
    feature_cols = ["skin_L", "skin_a", "skin_b", "contrast_enc"]
    for col in ["normal_pct", "oily_pct", "dry_pct"]:
        if col in df.columns:
            feature_cols.append(col)

    print("\nFeature importances:")
    for feat, imp in sorted(
        zip(feature_cols, model.feature_importances_), key=lambda x: -x[1]
    ):
        print(f"  {feat:<18} {imp:.4f}")

    print("\nDone. Run `python app.py` — the RF model will now be used.")


if __name__ == "__main__":
    main()
