# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 04:32:03 2026

@author: Raynard
"""

import os
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

scaled_features = ['Wattage_scaled', 'Current_scaled', 'Voltage_scaled', 'Alpha_scaled', 'Beta_scaled', 'Delta_scaled']

print("1. Loading datasets...")
df_synth = pd.read_csv(os.path.join(os.path.dirname(SCRIPT_DIR), "Model-DataCoevolution", "CoevolutionResult", "best_synthetic_data.csv"))

df_normal = df_synth[(df_synth['Condition'] == 'NORMAL')]
X_train = df_normal[scaled_features].values
input_dim = X_train.shape[1]

print("2. Training Autoencoder...")
# Encoder
input_layer = Input(shape=(input_dim,), name='ae_input')
encoded = Dense(7, activation='relu', name='encoder_layer_1')(input_layer)
encoded = Dense(3, activation='relu', name='encoder_bottleneck')(encoded) # Bottleneck

# Decoder
decoded = Dense(7, activation='relu', name='decoder_layer_1')(encoded)
decoded = Dense(input_dim, activation='linear', name='decoder_output')(decoded)

autoencoder = Model(input_layer, decoded, name='ac_fault_autoencoder')
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train, X_train, epochs=20, batch_size=16, verbose=1)

#%%

print("4. Saving Models to Disk...")
# Save the Keras Autoencoder
autoencoder.save('ac_autoencoder.h5')

print("✅ Training complete. 'ac_autoencoder.h5' saved.")