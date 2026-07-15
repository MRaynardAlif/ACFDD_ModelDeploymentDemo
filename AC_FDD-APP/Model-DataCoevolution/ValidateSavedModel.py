# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 23:35:53 2025

@author: Raynard
"""

# ValidateSavedModel.py

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os


def validate_saved_model(
        model_path,
        real_data_path,
        save_confusion=True,
        output_dir="ValidationOutput"
    ):
    """
    Standalone validator for any saved RandomForest model.
    Ensures feature alignment AND correct ordering using model.feature_names_in_.
    """

    # --- Load Model ---
    model = joblib.load(model_path)
    print(f"[Validator] Loaded model: {model_path}")

    # Extract expected feature order from the trained model
    expected_features = list(model.feature_names_in_)
    print(f"[Validator] Model expects {len(expected_features)} features.")

    # --- Load Real Dataset ---
    real_data = pd.read_csv(real_data_path)
    print(f"[Validator] Loaded real dataset: {real_data.shape}")

    if "Condition" not in real_data.columns:
        raise ValueError("Your real validation dataset MUST contain a 'Condition' column.")

    # --- Prepare Real Data ---
    # Select only numeric columns
    X_real = real_data.select_dtypes(include=['number']).copy()
    y_real = real_data["Condition"]

    # === Feature Alignment ===
    # Add missing columns that the model expects but are absent in real data
    for col in expected_features:
        if col not in X_real.columns:
            X_real[col] = 0.0
            
    # Filter and strictly order columns to match the trained model
    X_real = X_real[expected_features]
    print("[Validator] Real data aligned to model feature order.")

    # --- Prediction ---
    y_pred = model.predict(X_real)

    # --- Metrics ---
    print("\n=== Classification Report ===\n")
    print(classification_report(y_real, y_pred, zero_division=0))

    # --- Confusion Matrix ---
    if save_confusion:
        os.makedirs(output_dir, exist_ok=True)

        labels = sorted(list(set(y_real.unique()) | set(y_pred)))
        cm = confusion_matrix(y_real, y_pred, labels=labels)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()

        save_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"[Validator] Confusion matrix saved to: {save_path}")

    return y_pred


if __name__ == "__main__":
    # === EDIT THESE PATHS ===

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(SCRIPT_DIR, "CoevolutionResult", "best_model.pkl")
    REAL_DATA_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "RealData", "Dataset_Train_70.csv")   # your 70%/30% split

    validate_saved_model(
        model_path=MODEL_PATH,
        real_data_path=REAL_DATA_PATH,
        save_confusion=True,
        output_dir="ValidationOutput"
    )
