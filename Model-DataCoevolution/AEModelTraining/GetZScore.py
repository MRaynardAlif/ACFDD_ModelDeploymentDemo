# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 03:33:48 2026

@author: Raynard
"""

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

print("1. Loading Autoencoder and Dataset...")
autoencoder = load_model('ac_autoencoder.keras')
df_synth = pd.read_csv('best_synthetic_data.csv')

print("2. Isolating NORMAL data...")
# We only want the baseline behavior of healthy units
df_normal = df_synth[df_synth['Condition'] == 'NORMAL']

# The exact 6 features used in your app
ae_features = ['Wattage_scaled', 'Current_scaled', 'Voltage_scaled', 'Alpha_scaled', 'Beta_scaled', 'Delta_scaled']
X_normal = df_normal[ae_features].values

print("3. Generating Reconstructions...")
reconstructed = autoencoder.predict(X_normal)

# 4. Calculate the absolute error for every single row and feature
residuals = np.abs(X_normal - reconstructed)

# 5. Calculate Mean and Standard Deviation vertically across the columns (axis=0)
residual_means = np.mean(residuals, axis=0)
residual_stds = np.std(residuals, axis=0)

print("\n✅ Calculation Complete. Copy and paste these lines directly into your app.py:\n")
print(f"RESIDUAL_MEANS = np.array([{', '.join([f'{x:.5f}' for x in residual_means])}])")
print(f"RESIDUAL_STDS = np.array([{', '.join([f'{x:.5f}' for x in residual_stds])}])")