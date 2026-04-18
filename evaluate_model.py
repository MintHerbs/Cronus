
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)
from sklearn.preprocessing import LabelEncoder

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
DATASET_PATH = "output/final_skin_tone_dataset.csv"
OUTPUT_DIR = "model_evaluation"
RANDOM_STATE = 42

# Seaborn theme
sns.set_theme(style="whitegrid", context="talk")

# ═══════════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════════
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("=" * 70)
print("RANDOM FOREST CLASSIFIER EVALUATION")
print("Lipstick Recommendation System")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════
# STEP 1: LOAD AND PREPARE DATA
# ═══════════════════════════════════════════════════════════════════
print("\n[STEP 1] Loading dataset...")
df = pd.read_csv(DATASET_PATH)
print(f"Loaded {len(df)} rows × {len(df.columns)} columns")

# Define features and target
feature_cols = ['skin_L', 'skin_a', 'skin_b', 'undertone', 'delta_e', 'normal_pct', 'oily_pct', 'dry_pct', 'contrast_level']
target_col = 'sub_group'

print(f"\nFeatures: {feature_cols}")
print(f"Target: {target_col}")
print(f"\nTarget distribution:")
print(df[target_col].value_counts().sort_index())

# ═══════════════════════════════════════════════════════════════════
# STEP 2: ENCODE CATEGORICAL FEATURES
# ═══════════════════════════════════════════════════════════════════
print("\n[STEP 2] Encoding categorical features...")

# One-hot encode contrast_level and undertone
df_encoded = pd.get_dummies(df[feature_cols], columns=['contrast_level', 'undertone'], prefix=['contrast', 'undertone'])
print(f"After one-hot encoding: {df_encoded.shape[1]} features")
print(f"New columns: {list(df_encoded.columns)}")

X = df_encoded.values
y = df[target_col].values

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_names = label_encoder.classes_
print(f"\nEncoded {len(class_names)} sub-groups: {list(class_names)}")

# ═══════════════════════════════════════════════════════════════════
# STEP 3: TRAIN-TEST SPLIT (80/20 STRATIFIED)
# ═══════════════════════════════════════════════════════════════════
print("\n[STEP 3] Splitting dataset (80/20 stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, 
    test_size=0.2, 
    random_state=RANDOM_STATE,
    stratify=y_encoded
)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ═══════════════════════════════════════════════════════════════════
# STEP 4: TRAIN RANDOM FOREST CLASSIFIER
# ═══════════════════════════════════════════════════════════════════
print("\n[STEP 4] Training Random Forest classifier...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
print("Training complete")

# ═══════════════════════════════════════════════════════════════════
# STEP 5: EVALUATE MODEL
# ═══════════════════════════════════════════════════════════════════
print("\n[STEP 5] Evaluating model...")

# Predictions
y_pred = rf_model.predict(X_test)

# Overall accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nOverall Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Cross-validation (5-fold stratified)
print("\nPerforming 5-fold stratified cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(rf_model, X, y_encoded, cv=cv, scoring='accuracy', n_jobs=-1)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ═══════════════════════════════════════════════════════════════════
# STEP 6: GENERATE VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════
print("\n[STEP 6] Generating visualizations...")

# ───────────────────────────────────────────────────────────────────
# VISUALIZATION 1: CONFUSION MATRIX HEATMAP
# ───────────────────────────────────────────────────────────────────
print("  → Generating confusion matrix heatmap...")
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(14, 12), dpi=300)
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names,
    cbar_kws={'label': 'Count'},
    linewidths=0.5,
    linecolor='gray',
    ax=ax
)
ax.set_xlabel('Predicted Sub-Group', fontsize=13, fontweight='bold')
ax.set_ylabel('Actual Sub-Group', fontsize=13, fontweight='bold')
ax.set_title(
    f'Confusion Matrix — Random Forest Classifier\n'
    f'Test Accuracy: {accuracy:.2%} | {len(y_test)} samples',
    fontsize=15,
    fontweight='bold',
    pad=20
)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: {cm_path}")

# ───────────────────────────────────────────────────────────────────
# VISUALIZATION 2: CLASSIFICATION REPORT AS TABLE IMAGE
# ───────────────────────────────────────────────────────────────────
print("  → Generating classification report table...")

# Get classification report as dict
report_dict = classification_report(
    y_test, 
    y_pred, 
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

# Convert to DataFrame
report_df = pd.DataFrame(report_dict).transpose()

# Separate class-wise metrics from overall metrics
class_metrics = report_df.iloc[:-3]  # All classes
overall_metrics = report_df.iloc[-3:]  # accuracy, macro avg, weighted avg

# Create figure with two subplots
fig = plt.figure(figsize=(14, 10), dpi=300)
gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)

# ── Top table: Per-class metrics ──
ax1 = fig.add_subplot(gs[0])
ax1.axis('tight')
ax1.axis('off')

# Prepare data for table
class_table_data = []
for idx, row in class_metrics.iterrows():
    class_table_data.append([
        idx,
        f"{row['precision']:.3f}",
        f"{row['recall']:.3f}",
        f"{row['f1-score']:.3f}",
        f"{int(row['support'])}"
    ])

class_table = ax1.table(
    cellText=class_table_data,
    colLabels=['Sub-Group', 'Precision', 'Recall', 'F1-Score', 'Support'],
    cellLoc='center',
    loc='center',
    colWidths=[0.35, 0.15, 0.15, 0.15, 0.15]
)
class_table.auto_set_font_size(False)
class_table.set_fontsize(9)
class_table.scale(1, 2.2)

# Style header
for i in range(5):
    cell = class_table[(0, i)]
    cell.set_facecolor('#7A2D3A')
    cell.set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(class_table_data) + 1):
    for j in range(5):
        cell = class_table[(i, j)]
        if i % 2 == 0:
            cell.set_facecolor('#F8F3EE')
        else:
            cell.set_facecolor('white')
        cell.set_edgecolor('#CCCCCC')

ax1.set_title(
    'Classification Report — Per Sub-Group Metrics',
    fontsize=14,
    fontweight='bold',
    pad=20
)

# ── Bottom table: Overall metrics ──
ax2 = fig.add_subplot(gs[1])
ax2.axis('tight')
ax2.axis('off')

overall_table_data = []
for idx, row in overall_metrics.iterrows():
    if idx == 'accuracy':
        overall_table_data.append([
            'Overall Accuracy',
            '—',
            '—',
            f"{row['precision']:.3f}",  # accuracy is stored in precision column
            f"{int(row['support'])}"
        ])
    else:
        overall_table_data.append([
            idx.replace('avg', 'Average').title(),
            f"{row['precision']:.3f}",
            f"{row['recall']:.3f}",
            f"{row['f1-score']:.3f}",
            f"{int(row['support'])}"
        ])

overall_table = ax2.table(
    cellText=overall_table_data,
    colLabels=['Metric', 'Precision', 'Recall', 'F1-Score', 'Support'],
    cellLoc='center',
    loc='center',
    colWidths=[0.35, 0.15, 0.15, 0.15, 0.15]
)
overall_table.auto_set_font_size(False)
overall_table.set_fontsize(9)
overall_table.scale(1, 2.5)

# Style header
for i in range(5):
    cell = overall_table[(0, i)]
    cell.set_facecolor('#C4717A')
    cell.set_text_props(weight='bold', color='white')

# Style rows
for i in range(1, len(overall_table_data) + 1):
    for j in range(5):
        cell = overall_table[(i, j)]
        cell.set_facecolor('#EDE6DC')
        cell.set_edgecolor('#CCCCCC')

ax2.set_title(
    'Overall Performance Metrics',
    fontsize=12,
    fontweight='bold',
    pad=15
)

report_path = os.path.join(OUTPUT_DIR, "classification_report.png")
plt.savefig(report_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: {report_path}")

# ───────────────────────────────────────────────────────────────────
# VISUALIZATION 3: FEATURE IMPORTANCE BAR CHART
# ───────────────────────────────────────────────────────────────────
print("  → Generating feature importance chart...")

# Get feature importances
feature_names = list(df_encoded.columns)
importances = rf_model.feature_importances_

# Create DataFrame and sort
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=True)

# Plot
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
colors = sns.color_palette('RdYlGn_r', len(importance_df))

bars = ax.barh(
    importance_df['Feature'],
    importance_df['Importance'],
    color=colors,
    edgecolor='white',
    linewidth=0.8
)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, importance_df['Importance'])):
    ax.text(
        val + 0.002,
        bar.get_y() + bar.get_height() / 2,
        f'{val:.4f}',
        va='center',
        fontsize=8,
        color='#2C1F27'
    )

ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
ax.set_title(
    'Random Forest Feature Importance\n'
    f'Top features driving sub-group predictions',
    fontsize=14,
    fontweight='bold',
    pad=20
)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()
importance_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
plt.savefig(importance_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: {importance_path}")

# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)
print(f"\nModel Performance:")
print(f"  • Test Accuracy:        {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  • CV Mean Accuracy:     {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  • Training samples:     {len(X_train)}")
print(f"  • Test samples:         {len(X_test)}")
print(f"  • Number of classes:    {len(class_names)}")
print(f"  • Number of features:   {X.shape[1]}")

print(f"\nVisualizations saved to: {OUTPUT_DIR}/")
print(f"  • confusion_matrix.png")
print(f"  • classification_report.png")
print(f"  • feature_importance.png")
print("=" * 70)