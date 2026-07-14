# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 21:53:36 2026

@author: Raynard
"""

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error

print("1. Loading Autoencoder and Data...")
autoencoder = load_model('ac_autoencoder.keras')
df_synth = pd.read_csv('best_synthetic_data.csv')

# 2. Filter for ONLY Normal data that the model hasn't memorized perfectly
# (Ideally, use a validation split, but we'll use the normal dataset here)
df_normal = df_synth[df_synth['Condition'] == 'NORMAL']

scaled_features = ['Wattage_scaled', 'Current_scaled', 'Voltage_scaled', 'Alpha_scaled', 'Beta_scaled', 'Delta_scaled']
X_normal = df_normal[scaled_features].values

print("2. Calculating Reconstruction Errors...")
reconstructed = autoencoder.predict(X_normal)

# Calculate MSE for every single row
mses = np.mean(np.power(X_normal - reconstructed, 2), axis=1)

print("\n--- Threshold Tuning Results ---")
print(f"Mean MSE: {np.mean(mses):.5f}")
print(f"Max Normal MSE: {np.max(mses):.5f}")

# 3. Calculate Thresholds
threshold_3sigma = np.mean(mses) + (3 * np.std(mses))
threshold_95th = np.percentile(mses, 95)
threshold_99th = np.percentile(mses, 99)

print(f"\nOption A: 3-Sigma Threshold: {threshold_3sigma:.5f}")
print(f"Option B: 95th Percentile:   {threshold_95th:.5f}")
print(f"Option C: 99th Percentile:   {threshold_99th:.5f} (Recommended)")

# Update your Gradio app with the 99th percentile result.