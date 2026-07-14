# -*- coding: utf-8 -*-
"""
app.py - Gradio demo for the AC Fault Detection & Diagnosis model.

Deploy this on a Hugging Face Space (SDK: gradio).

Instead of a form of 17 raw feature boxes, this asks for the physical
sensor readings a person would actually plug in (Unit type, Current,
Voltage, Indoor/Outdoor Temp, Set Point, Supply, Return) and derives
every engineered/scaled feature the exact same way the training
pipeline does (TieredScaling.py), then aligns the result 
to whatever model.feature_names_in_ expects.

Required files in the Space repo:
    app.py              (this file)
    requirements.txt
    best_model.pkl (Random Forest for Fault Detection)      
    ac_autoencoder.keras (Autoencoder for Diagnosis)
"""

import joblib
import pandas as pd
import numpy as np
import gradio as gr
#import spaces
from tensorflow.keras.models import load_model

MODEL_PATH = "best_model.pkl"  
AE_MODEL_PATH = "ac_autoencoder.keras"

# --- Load model once at startup ---
model = joblib.load(MODEL_PATH)
autoencoder = load_model(AE_MODEL_PATH)

FEATURES = list(model.feature_names_in_)
CLASSES = [str(c) for c in model.classes_]
print(f"[app] Loaded RF model. Expects {len(FEATURES)} features: {FEATURES}")
print(f"[app] Classes: {CLASSES}")
print("[app] Loaded AE model for Diagnosis.")

# --- Same per-unit rule tables as TieredScaling.py / StaticDataGenerator.py ---
NORMAL_RULES = {
    'NON INV 0.5PK': {'AMPERE MIN': 1.26, 'AMPERE MAX': 2.35, 'WATT MIN': 289.8, 'WATT MAX': 540.5, 'BETA MIN': -1, 'BETA MAX': 17, 'DELTA MIN': -5, 'DELTA MAX': 12, 'ALPHA MIN': -3, 'ALPHA MAX': 7},
    'NON INV 0.75PK': {'AMPERE MIN': 2.01, 'AMPERE MAX': 3.73, 'WATT MIN': 462.3, 'WATT MAX': 857.9, 'BETA MIN': -1, 'BETA MAX': 17, 'DELTA MIN': -5, 'DELTA MAX': 12, 'ALPHA MIN': -3, 'ALPHA MAX': 7},
    'NON INV 1PK': {'AMPERE MIN': 2.57, 'AMPERE MAX': 4.78, 'WATT MIN': 591.1, 'WATT MAX': 1099.4, 'BETA MIN': -1, 'BETA MAX': 17, 'DELTA MIN': -5, 'DELTA MAX': 12, 'ALPHA MIN': -3, 'ALPHA MAX': 7},
    'NON INV 1.5PK': {'AMPERE MIN': 3.37, 'AMPERE MAX': 6.27, 'WATT MIN': 775.1, 'WATT MAX': 1442.1, 'BETA MIN': -1, 'BETA MAX': 17, 'DELTA MIN': -5, 'DELTA MAX': 12, 'ALPHA MIN': -3, 'ALPHA MAX': 7},
    'NON INV 2PK': {'AMPERE MIN': 5.29, 'AMPERE MAX': 9.85, 'WATT MIN': 1216.7, 'WATT MAX': 2265.5, 'BETA MIN': -1, 'BETA MAX': 17, 'DELTA MIN': -5, 'DELTA MAX': 12, 'ALPHA MIN': -3, 'ALPHA MAX': 7},
    'NON INV 2.5PK': {'AMPERE MIN': 6.89, 'AMPERE MAX': 12.81, 'WATT MIN': 1584.7, 'WATT MAX': 2946.3, 'BETA MIN': -1, 'BETA MAX': 17, 'DELTA MIN': -5, 'DELTA MAX': 12, 'ALPHA MIN': -3, 'ALPHA MAX': 7},
    'INV 0.5PK': {'AMPERE MIN': 1.48, 'AMPERE MAX': 2.74, 'WATT MIN': 340.4, 'WATT MAX': 630.2, 'BETA MIN': -1, 'BETA MAX': 17, 'DELTA MIN': -5, 'DELTA MAX': 12, 'ALPHA MIN': -3, 'ALPHA MAX': 7},
    'INV 0.75PK': {'AMPERE MIN': 1.75, 'AMPERE MAX': 3.25, 'WATT MIN': 402.5, 'WATT MAX': 747.5, 'BETA MIN': -1, 'BETA MAX': 17, 'DELTA MIN': -5, 'DELTA MAX': 12, 'ALPHA MIN': -3, 'ALPHA MAX': 7},
    'INV 1PK': {'AMPERE MIN': 2.49, 'AMPERE MAX': 4.62, 'WATT MIN': 572.7, 'WATT MAX': 1062.6, 'BETA MIN': -1, 'BETA MAX': 17, 'DELTA MIN': -5, 'DELTA MAX': 12, 'ALPHA MIN': -3, 'ALPHA MAX': 7},
    'INV 1.5PK': {'AMPERE MIN': 3.35, 'AMPERE MAX': 6.22, 'WATT MIN': 770.5, 'WATT MAX': 1430.6, 'BETA MIN': -1, 'BETA MAX': 17, 'DELTA MIN': -5, 'DELTA MAX': 12, 'ALPHA MIN': -3, 'ALPHA MAX': 7},
    'INV 2PK': {'AMPERE MIN': 4.94, 'AMPERE MAX': 9.17, 'WATT MIN': 1136.2, 'WATT MAX': 2109.1, 'BETA MIN': -1, 'BETA MAX': 17, 'DELTA MIN': -5, 'DELTA MAX': 12, 'ALPHA MIN': -3, 'ALPHA MAX': 7},
    'INV 2.5PK': {'AMPERE MIN': 7.03, 'AMPERE MAX': 13.05, 'WATT MIN': 1616.9, 'WATT MAX': 3001.5, 'BETA MIN': -1, 'BETA MAX': 17, 'DELTA MIN': -5, 'DELTA MAX': 12, 'ALPHA MIN': -3, 'ALPHA MAX': 7},
}
MAINTENANCE_RULES = {
    'NON INV 0.5PK': {'AMPERE MIN': 1.11, 'AMPERE MAX': 3.42, 'WATT MIN': 255.3, 'WATT MAX': 786.6, 'BETA MIN': -3, 'BETA MAX': 25, 'DELTA MIN': -17, 'DELTA MAX': 22, 'ALPHA MIN': -10, 'ALPHA MAX': 10},
    'NON INV 0.75PK': {'AMPERE MIN': 1.78, 'AMPERE MAX': 4.95, 'WATT MIN': 409.4, 'WATT MAX': 1138.5, 'BETA MIN': -3, 'BETA MAX': 25, 'DELTA MIN': -17, 'DELTA MAX': 22, 'ALPHA MIN': -10, 'ALPHA MAX': 10},
    'NON INV 1PK': {'AMPERE MIN': 2.39, 'AMPERE MAX': 6.32, 'WATT MIN': 549.7, 'WATT MAX': 1453.6, 'BETA MIN': -3, 'BETA MAX': 25, 'DELTA MIN': -17, 'DELTA MAX': 22, 'ALPHA MIN': -10, 'ALPHA MAX': 10},
    'NON INV 1.5PK': {'AMPERE MIN': 3.15, 'AMPERE MAX': 7.94, 'WATT MIN': 724.5, 'WATT MAX': 1826.2, 'BETA MIN': -3, 'BETA MAX': 25, 'DELTA MIN': -17, 'DELTA MAX': 22, 'ALPHA MIN': -10, 'ALPHA MAX': 10},
    'NON INV 2PK': {'AMPERE MIN': 5.06, 'AMPERE MAX': 12.15, 'WATT MIN': 1163.8, 'WATT MAX': 2794.5, 'BETA MIN': -3, 'BETA MAX': 25, 'DELTA MIN': -17, 'DELTA MAX': 22, 'ALPHA MIN': -10, 'ALPHA MAX': 10},
    'NON INV 2.5PK': {'AMPERE MIN': 6.75, 'AMPERE MAX': 15.45, 'WATT MIN': 1552.5, 'WATT MAX': 3553.5, 'BETA MIN': -3, 'BETA MAX': 25, 'DELTA MIN': -17, 'DELTA MAX': 22, 'ALPHA MIN': -10, 'ALPHA MAX': 10},
    'INV 0.5PK': {'AMPERE MIN': 1.15, 'AMPERE MAX': 3.96, 'WATT MIN': 264.5, 'WATT MAX': 910.8, 'BETA MIN': -3, 'BETA MAX': 25, 'DELTA MIN': -17, 'DELTA MAX': 22, 'ALPHA MIN': -10, 'ALPHA MAX': 10},
    'INV 0.75PK': {'AMPERE MIN': 1.75, 'AMPERE MAX': 3.75, 'WATT MIN': 402.5, 'WATT MAX': 862.5, 'BETA MIN': -3, 'BETA MAX': 25, 'DELTA MIN': -17, 'DELTA MAX': 22, 'ALPHA MIN': -10, 'ALPHA MAX': 10},
    'INV 1PK': {'AMPERE MIN': 1.18, 'AMPERE MAX': 6.28, 'WATT MIN': 271.4, 'WATT MAX': 1444.4, 'BETA MIN': -3, 'BETA MAX': 25, 'DELTA MIN': -17, 'DELTA MAX': 22, 'ALPHA MIN': -10, 'ALPHA MAX': 10},
    'INV 1.5PK': {'AMPERE MIN': 2.04, 'AMPERE MAX': 9.55, 'WATT MIN': 469.2, 'WATT MAX': 2196.5, 'BETA MIN': -3, 'BETA MAX': 25, 'DELTA MIN': -17, 'DELTA MAX': 22, 'ALPHA MIN': -10, 'ALPHA MAX': 10},
    'INV 2PK': {'AMPERE MIN': 2.48, 'AMPERE MAX': 12.62, 'WATT MIN': 570.4, 'WATT MAX': 2902.6, 'BETA MIN': -3, 'BETA MAX': 25, 'DELTA MIN': -17, 'DELTA MAX': 22, 'ALPHA MIN': -10, 'ALPHA MAX': 10},
    'INV 2.5PK': {'AMPERE MIN': 5.60, 'AMPERE MAX': 17.73, 'WATT MIN': 1288, 'WATT MAX': 4077.9, 'BETA MIN': -3, 'BETA MAX': 25, 'DELTA MIN': -17, 'DELTA MAX': 22, 'ALPHA MIN': -10, 'ALPHA MAX': 10},
}
UNIT_NAMES = list(NORMAL_RULES.keys())
VOLT_MIN, VOLT_MAX = 196, 265


def tiered_scale(value, unit, prefix):
    """Same piecewise logic as TieredScaling.py's tiered_scaler()."""
    n_min, n_max = NORMAL_RULES[unit][f'{prefix} MIN'], NORMAL_RULES[unit][f'{prefix} MAX']
    m_min, m_max = MAINTENANCE_RULES[unit][f'{prefix} MIN'], MAINTENANCE_RULES[unit][f'{prefix} MAX']
    n_range = n_max - n_min if n_max > n_min else 1
    m_range_low = n_min - m_min if n_min > m_min else 1
    m_range_high = m_max - n_max if m_max > n_max else 1

    if n_min <= value <= n_max:
        return (value - n_min) / n_range
    elif value < n_min:
        return -1 + (value - m_min) / m_range_low
    else:  # value > n_max
        return 1 + (value - n_max) / m_range_high


def build_feature_row(unit, current, voltage, temp_indoor, temp_outdoor, set_point, supply, ret):
    """Derives every engineered feature exactly as the training pipeline does."""
    wattage = voltage * current
    alpha = temp_indoor - set_point   # StaticDataGenerator.py definition
    beta = temp_outdoor - set_point   
    delta = ret - supply              

    row = {
        "Current (A)": current,
        "Voltage (V)": voltage,
        "Wattage (W)": wattage,
        "Temperature Indoor (°C)": temp_indoor,
        "Temperature Outdoor (°C)": temp_outdoor,
        "Set Point (°C)": set_point,
        "Alpha (°C)": alpha,
        "Supply (°C)": supply,
        "Return (°C)": ret,
        "Delta (°C)": delta,
        "Beta (°C)": beta,
        "Voltage_scaled": (voltage - VOLT_MIN) / (VOLT_MAX - VOLT_MIN),
        "Current_scaled": tiered_scale(current, unit, "AMPERE"),
        "Wattage_scaled": tiered_scale(wattage, unit, "WATT"),
        "Alpha_scaled": tiered_scale(alpha, unit, "ALPHA"),
        "Beta_scaled": tiered_scale(beta, unit, "BETA"),
        "Delta_scaled": tiered_scale(delta, unit, "DELTA"),
    }
    return row

#@spaces.GPU
def predict_condition(unit, current, voltage, temp_indoor, temp_outdoor, set_point, supply, ret):
    row = build_feature_row(unit, current, voltage, temp_indoor, temp_outdoor, set_point, supply, ret)
    df_input = pd.DataFrame([row])

    # --- 1. FAULT DETECTION (Random Forest) ---
    rf_input = df_input.copy()
    
    # Align exactly to what the model was trained on
    for col in FEATURES:
        if col not in rf_input.columns:
            rf_input[col] = 0.0
    rf_input = rf_input[FEATURES]

    prediction = model.predict(rf_input)[0]
    probabilities = model.predict_proba(rf_input)[0]
    prob_dict = {str(c): float(p) for c, p in zip(model.classes_, probabilities)}

    # --- 2. FAULT DIAGNOSIS (Autoencoder) ---
    ae_features = ['Wattage_scaled', 'Current_scaled', 'Voltage_scaled', 'Alpha_scaled', 'Beta_scaled', 'Delta_scaled']
    ae_input_data = df_input[ae_features].values
    
    # Reconstruct the input
    reconstructed = autoencoder.predict(ae_input_data, verbose=0)[0]
    
    # Calculate Absolute Residuals and MSE
    residuals = np.abs(ae_input_data[0] - reconstructed)
    mse = np.mean(residuals**2)
    
    # Map residuals back to their feature names to find the culprit
    feature_errors = dict(zip(ae_features, residuals))
    worst_feature = max(feature_errors, key=feature_errors.get)
    worst_feature_clean = worst_feature.replace('_scaled', '')
    
    # Formulate the Diagnosis string
    if prediction == "NORMAL" and mse < 0.08116: 
        diagnosis_msg = f"✅ System operating optimally. (MSE: {mse:.4f})"
    else:
        diagnosis_msg = f"⚠️ {worst_feature_clean} Anomaly occurred. (Reconstruction MSE: {mse:.4f})"

    return str(prediction), prob_dict, diagnosis_msg


# --- Gradio UI ---
interface = gr.Interface(
    fn=predict_condition,
    inputs=[
        gr.Dropdown(choices=UNIT_NAMES, value=UNIT_NAMES[0], label="Unit Type"),
        gr.Slider(minimum=0, maximum=20, value=3.0, label="Current (A)"),
        gr.Slider(minimum=0, maximum=500, value=230, label="Voltage (V)"),
        gr.Slider(minimum=0, maximum=50, value=24, label="Temperature Indoor (°C)"),
        gr.Slider(minimum=0, maximum=50, value=30, label="Temperature Outdoor (°C)"),
        gr.Slider(minimum=16, maximum=28, value=22, label="Set Point (°C)"),
        gr.Slider(minimum=0, maximum=50, value=18, label="Supply (°C)"),
        gr.Slider(minimum=0, maximum=50, value=24, label="Return (°C)"),
    ],
    outputs=[
        gr.Textbox(label="Detected Condition"),
        gr.Label(label="Classification Confidence Breakdown"),
        gr.Textbox(label="Diagnostic Assessment")
    ],
    title="AC Unit Fault Detection & Diagnosis",
    description="Enter sensor readings to classify AC operating condition and diagnose underlying parameter faults.",
)

if __name__ == "__main__":
    interface.launch(ssr_mode=False)