# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 04:32:03 2026

@author: Raynard
"""

import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

scaled_features = ['Wattage_scaled', 'Current_scaled', 'Voltage_scaled', 'Alpha_scaled', 'Beta_scaled', 'Delta_scaled']

print("1. Loading datasets...")
df_synth = pd.read_csv('best_synthetic_data.csv')

df_normal = df_synth[(df_synth['Condition'] == 'NORMAL')]
X_train = df_normal[scaled_features].values
input_dim = X_train.shape[1]

print("2. Training Autoencoder...")
# Encoder
input_layer = Input(shape=(input_dim,))
encoded = Dense(6, activation='relu')(input_layer)
encoded = Dense(3, activation='relu')(encoded) # Bottleneck

# Decoder
decoded = Dense(6, activation='relu')(encoded)
decoded = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train, X_train, epochs=20, batch_size=16, verbose=1)

#%%

print("4. Saving Models to Disk...")
# Save the Keras Autoencoder
autoencoder.save('ac_autoencoder.keras')

print("✅ Training complete. 'ac_autoencoder.keras' saved.")