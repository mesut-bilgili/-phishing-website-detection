import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset_phishing.csv"
TARGET_CANDIDATES = ["status", "label", "target", "class"]
RANDOM_STATE = 42

print("=" * 60)
print("PHISHING WEBSITE DETECTION - SVM")
print("=" * 60)

if not DATASET_PATH.exists():
    csv_files = list(BASE_DIR.glob("*.csv"))
    if len(csv_files) == 1:
        DATASET_PATH = csv_files[0]
        print("Automatically detected dataset:", DATASET_PATH.name)
    else:
        raise FileNotFoundError(
            f"Cannot find {DATASET_PATH.name}. Put the CSV beside this script "
            "or change DATASET_PATH in the script."
        )

df = pd.read_csv(DATASET_PATH)

print("\\n========== DATASET INFORMATION ==========")
print("Dataset:", DATASET_PATH.name)
print("Shape:", df.shape)
print("Missing values:", int(df.isnull().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))


column_lookup = {str(c).strip().lower(): c for c in df.columns}
target_column = next(
    (column_lookup[name] for name in TARGET_CANDIDATES if name in column_lookup),
    None
)

if target_column is None:
    raise ValueError(
        "Target column not found. Available columns: "
        + str(df.columns.tolist())
        + ". Edit TARGET_CANDIDATES or set target_column manually."
    )

df = df.dropna(subset=[target_column]).copy()
y_raw = df[target_column].astype(str).str.strip().str.lower()
X_df = df.drop(columns=[target_column]).copy()


drop_cols = [
    c for c in X_df.columns
    if str(c).strip().lower() in {"url", "index", "unnamed: 0"}
]
X_df = X_df.drop(columns=drop_cols, errors="ignore")
X_df = X_df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)

if X_df.shape[1] == 0:
    raise ValueError("No numeric feature columns were found.")

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

if len(label_encoder.classes_) != 2:
    raise ValueError(
        f"Expected binary classification, found: {list(label_encoder.classes_)}"
    )

phishing_ids = [
    i for i, label in enumerate(label_encoder.classes_)
    if "phish" in str(label).lower()
]
if len(phishing_ids) != 1:
    raise ValueError(
        "Could not identify the phishing class automatically. "
        f"Detected labels: {list(label_encoder.classes_)}"
    )
phishing_class = phishing_ids[0]

print("\\n========== CLASS DISTRIBUTION ==========")
print(y_raw.value_counts())
print("\\nTarget column:", target_column)
print("Classes:", list(label_encoder.classes_))
print("Number of numeric features:", X_df.shape[1])


X_train, X_temp, y_train, y_temp = train_test_split(
    X_df, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50,
    random_state=RANDOM_STATE, stratify=y_temp
)

print("\\n========== DATA SPLIT ==========")
print(f"Train: {len(X_train)} ({len(X_train) / len(df) * 100:.2f}%)")
print(f"Validation: {len(X_val)} ({len(X_val) / len(df) * 100:.2f}%)")
print(f"Test: {len(X_test)} ({len(X_test) / len(df) * 100:.2f}%)")


imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train)
X_val_imp = imputer.transform(X_val)
X_test_imp = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_val_scaled = scaler.transform(X_val_imp)
X_test_scaled = scaler.transform(X_test_imp)

print("\\nPreprocessing completed.")
print("Training feature shape:", X_train_scaled.shape)
print("Validation feature shape:", X_val_scaled.shape)
print("Test feature shape:", X_test_scaled.shape)


svm_model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    random_state=RANDOM_STATE
)

print("\\n========== MODEL TRAINING ==========")
print("Algorithm: Support Vector Machine (SVM)")
print("Kernel: RBF | C: 1.0 | Gamma: scale")

start_time = time.time()
svm_model.fit(X_train_scaled, y_train)
training_time = time.time() - start_time
print(f"Training time: {training_time:.2f} seconds")


y_pred = svm_model.predict(X_test_scaled)
phishing_probability = svm_model.predict_proba(X_test_scaled)[:, phishing_class]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=phishing_class, zero_division=0)
recall = recall_score(y_test, y_pred, pos_label=phishing_class, zero_division=0)
f1 = f1_score(y_test, y_pred, pos_label=phishing_class, zero_division=0)
roc_auc = roc_auc_score(
    (y_test == phishing_class).astype(int), phishing_probability
)

print("\\n========== SVM TEST RESULTS ==========")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")
print(f"ROC-AUC   : {roc_auc * 100:.2f}%")


class_names = [str(x) for x in label_encoder.classes_]
report = classification_report(
    y_test, y_pred, labels=np.arange(len(class_names)),
    target_names=class_names, zero_division=0
)
print("\\n========== CLASSIFICATION REPORT ==========")