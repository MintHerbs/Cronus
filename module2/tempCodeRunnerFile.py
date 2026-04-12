# module2/TrainSkinType.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

sns.set_theme(style="whitegrid", context="talk")

# ==============================
# Configuration
# ==============================

CurrentDir = os.path.dirname(os.path.abspath(__file__))
ProjectRoot = os.path.abspath(os.path.join(CurrentDir, ".."))

DatasetRoot = os.path.join(ProjectRoot, "dataset", "Oily-Dry-Skin-Types")

TrainPath = os.path.join(DatasetRoot, "train")
ValidPath = os.path.join(DatasetRoot, "valid")
TestPath  = os.path.join(DatasetRoot, "test")

ModelOutputDir = os.path.join(ProjectRoot, "model")
os.makedirs(ModelOutputDir, exist_ok=True)

ImageSize = 224
BatchSize = 32
Epochs = 10

os.makedirs("model", exist_ok=True)

# ==============================
# Data Generators
# ==============================

TrainGenerator = ImageDataGenerator(
    rescale=1.0 / 255.0
).flow_from_directory(
    TrainPath,
    target_size=(ImageSize, ImageSize),
    batch_size=BatchSize,
    class_mode="categorical"
)

ValidGenerator = ImageDataGenerator(
    rescale=1.0 / 255.0
).flow_from_directory(
    ValidPath,
    target_size=(ImageSize, ImageSize),
    batch_size=BatchSize,
    class_mode="categorical"
)

TestGenerator = ImageDataGenerator(
    rescale=1.0 / 255.0
).flow_from_directory(
    TestPath,
    target_size=(ImageSize, ImageSize),
    batch_size=BatchSize,
    class_mode="categorical",
    shuffle=False
)

ClassNames = list(TrainGenerator.class_indices.keys())

print("Classes detected:", ClassNames)

# ==============================
# Model (Transfer Learning)
# ==============================

BaseModel = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(ImageSize, ImageSize, 3)
)

BaseModel.trainable = False

x = BaseModel.output
x = GlobalAveragePooling2D()(x)
OutputLayer = Dense(len(ClassNames), activation="softmax")(x)

ModelSkin = Model(inputs=BaseModel.input, outputs=OutputLayer)

ModelSkin.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ==============================
# Training
# ==============================

History = ModelSkin.fit(
    TrainGenerator,
    validation_data=ValidGenerator,
    epochs=Epochs
)

# ==============================
# Save Model
# ==============================

ModelSavePath = os.path.join(ModelOutputDir, "SkinTypeModel.h5")
ModelSkin.save(ModelSavePath)

print("Model saved to:", ModelSavePath)
# ==============================
# Evaluation
# ==============================

TestLoss, TestAccuracy = ModelSkin.evaluate(TestGenerator)
print("Test Accuracy:", TestAccuracy)

Predictions = ModelSkin.predict(TestGenerator)
PredictedClasses = np.argmax(Predictions, axis=1)
TrueClasses = TestGenerator.classes

# Classification Report
Report = classification_report(TrueClasses, PredictedClasses, target_names=ClassNames)
print(Report)

# ==============================
# Seaborn Debug Graphs
# ==============================

# Accuracy Plot
plt.figure(figsize=(8,6))
sns.lineplot(data=History.history["accuracy"], label="Train")
sns.lineplot(data=History.history["val_accuracy"], label="Validation")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("model/AccuracyPlot.png")
plt.close()

# Loss Plot
plt.figure(figsize=(8,6))
sns.lineplot(data=History.history["loss"], label="Train")
sns.lineplot(data=History.history["val_loss"], label="Validation")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("model/LossPlot.png")
plt.close()

# Confusion Matrix
ConfMatrix = confusion_matrix(TrueClasses, PredictedClasses)

plt.figure(figsize=(8,6))
sns.heatmap(
    ConfMatrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=ClassNames,
    yticklabels=ClassNames
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("model/ConfusionMatrix.png")
plt.close()

print("All debug graphs saved inside model/ folder.")