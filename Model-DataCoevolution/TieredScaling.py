# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 11:58:34 2025

@author: Raynard
"""

import pandas as pd
import numpy as np
import os

def preprocess_ac_data_tiered_scaling(
    input_filepath=r'R:\Mine\AllOfMe\Big Data\Model-DataCoevolution2\SyntheticStatic\SyntheticTrainingDataset.csv',
    output_filepath=r'R:\Mine\AllOfMe\Big Data\Model-DataCoevolution2\SyntheticStatic\ScaledSyntheticTrainingDataset.csv'
):
    """
    Preprocesses the synthetic AC dataset using a custom scaling method.

    The scaling logic is now two-fold:
    1.  For Voltage: A simple linear scaling is applied based on the raw voltage
        value. The logic ensures that when the 'Condition' is 'Trouble 3', the
        raw voltage is outside the normal range, resulting in a scaled value
        of < 0 or > 1. For all other conditions, the scaled value is [0, 1].
    2.  For other parameters (Current, Wattage, Alpha, Beta, Delta): A tiered
        scaling method is used.
        - Normal Range is scaled to [0, 1].
        - Maintenance Ranges are scaled to [-1, 0) and (1, 2].
        - Trouble Ranges are scaled to be < -1 or > 2.

    Args:
        input_filepath (str): The path to the generated CSV file.
        output_filepath (str): The path to save the preprocessed CSV file.

    Returns:
        pandas.DataFrame: The preprocessed and scaled dataset.
    """
    # --- 1. Define the same Tiered Expert Rules from the generator ---
    # These are crucial for the scaling logic to work correctly.

    # Tier 1: Normal Operating Parameters
    normal_rules = {
        'NON INV 1/2 PK': {'VOLT MIN': 196, 'VOLT MAX': 265, 'AMPERE MIN': 1.26, 'AMPERE MAX': 2.35, 'WATT MIN': 289.8, 'WATT MAX': 540.5, 'BETA MIN': -1, 'BETA MAX': 17, 'DELTA MIN': -5, 'DELTA MAX': 12, 'ALPHA MIN': -3, 'ALPHA MAX': 7},
        'NON INV 3/4 PK': {'VOLT MIN': 196, 'VOLT MAX': 265, 'AMPERE MIN': 2.01, 'AMPERE MAX': 3.73, 'WATT MIN': 462.3, 'WATT MAX': 857.9, 'BETA MIN': -1, 'BETA MAX': 17, 'DELTA MIN': -5, 'DELTA MAX': 12, 'ALPHA MIN': -3, 'ALPHA MAX': 7},
        'NON INV 1 PK': {'VOLT MIN': 196, 'VOLT MAX': 265, 'AMPERE MIN': 2.57, 'AMPERE MAX': 4.78, 'WATT MIN': 591.1, 'WATT MAX': 1099.4, 'BETA MIN': -1, 'BETA MAX': 17, 'DELTA MIN': -5, 'DELTA MAX': 12, 'ALPHA MIN': -3, 'ALPHA MAX': 7},
        'NON INV 1.5 PK': {'VOLT MIN': 196, 'VOLT MAX': 265, 'AMPERE MIN': 3.37, 'AMPERE MAX': 6.27, 'WATT MIN': 775.1, 'WATT MAX': 1442.1, 'BETA MIN': -1, 'BETA MAX': 17, 'DELTA MIN': -5, 'DELTA MAX': 12, 'ALPHA MIN': -3, 'ALPHA MAX': 7},
        'NON INV 2 PK': {'VOLT MIN': 196, 'VOLT MAX': 265, 'AMPERE MIN': 5.29, 'AMPERE MAX': 9.85, 'WATT MIN': 1216.7, 'WATT MAX': 2265.5, 'BETA MIN': -1, 'BETA MAX': 17, 'DELTA MIN': -5, 'DELTA MAX': 12, 'ALPHA MIN': -3, 'ALPHA MAX': 7},
        'NON INV 2.5 PK': {'VOLT MIN': 196, 'VOLT MAX': 265, 'AMPERE MIN': 6.89, 'AMPERE MAX': 12.81, 'WATT MIN': 1584.7, 'WATT MAX': 2946.3, 'BETA MIN': -1, 'BETA MAX': 17, 'DELTA MIN': -5, 'DELTA MAX': 12, 'ALPHA MIN': -3, 'ALPHA MAX': 7},
        'INV 1/2 PK': {'VOLT MIN': 196, 'VOLT MAX': 265, 'AMPERE MIN': 1.48, 'AMPERE MAX': 2.74, 'WATT MIN': 340.4, 'WATT MAX': 630.2, 'BETA MIN': -1, 'BETA MAX': 17, 'DELTA MIN': -5, 'DELTA MAX': 12, 'ALPHA MIN': -3, 'ALPHA MAX': 7},
        'INV 3/4 PK': {'VOLT MIN': 196, 'VOLT MAX': 265, 'AMPERE MIN': 1.75, 'AMPERE MAX': 3.25, 'WATT MIN': 402.5, 'WATT MAX': 747.5, 'BETA MIN': -1, 'BETA MAX': 17, 'DELTA MIN': -5, 'DELTA MAX': 12, 'ALPHA MIN': -3, 'ALPHA MAX': 7},
        'INV 1 PK': {'VOLT MIN': 196, 'VOLT MAX': 265, 'AMPERE MIN': 2.49, 'AMPERE MAX': 4.62, 'WATT MIN': 572.7, 'WATT MAX': 1062.6, 'BETA MIN': -1, 'BETA MAX': 17, 'DELTA MIN': -5, 'DELTA MAX': 12, 'ALPHA MIN': -3, 'ALPHA MAX': 7},
        'INV 1.5 PK': {'VOLT MIN': 196, 'VOLT MAX': 265, 'AMPERE MIN': 3.35, 'AMPERE MAX': 6.22, 'WATT MIN': 770.5, 'WATT MAX': 1430.6, 'BETA MIN': -1, 'BETA MAX': 17, 'DELTA MIN': -5, 'DELTA MAX': 12, 'ALPHA MIN': -3, 'ALPHA MAX': 7},
        'INV 2 PK': {'VOLT MIN': 196, 'VOLT MAX': 265, 'AMPERE MIN': 4.94, 'AMPERE MAX': 9.17, 'WATT MIN': 1136.2, 'WATT MAX': 2109.1, 'BETA MIN': -1, 'BETA MAX': 17, 'DELTA MIN': -5, 'DELTA MAX': 12, 'ALPHA MIN': -3, 'ALPHA MAX': 7},
        'INV 2.5 PK': {'VOLT MIN': 196, 'VOLT MAX': 265, 'AMPERE MIN': 7.03, 'AMPERE MAX': 13.05, 'WATT MIN': 1616.9, 'WATT MAX': 3001.5, 'BETA MIN': -1, 'BETA MAX': 17, 'DELTA MIN': -5, 'DELTA MAX': 12, 'ALPHA MIN': -3, 'ALPHA MAX': 7},
    }

    # Tier 2: Maintenance Parameter Ranges
    maintenance_rules = {
        'NON INV 1/2 PK': {'VOLT MIN': 196, 'VOLT MAX': 265, 'AMPERE MIN': 1.11, 'AMPERE MAX': 3.42, 'WATT MIN': 255.3, 'WATT MAX': 786.6, 'BETA MIN': -3, 'BETA MAX': 25, 'DELTA MIN': -17, 'DELTA MAX': 22, 'ALPHA MIN': -10, 'ALPHA MAX': 10},
        'NON INV 3/4 PK': {'VOLT MIN': 196, 'VOLT MAX': 265, 'AMPERE MIN': 1.78, 'AMPERE MAX': 4.95, 'WATT MIN': 409.4, 'WATT MAX': 1138.5, 'BETA MIN': -3, 'BETA MAX': 25, 'DELTA MIN': -17, 'DELTA MAX': 22, 'ALPHA MIN': -10, 'ALPHA MAX': 10},
        'NON INV 1 PK': {'VOLT MIN': 196, 'VOLT MAX': 265, 'AMPERE MIN': 2.39, 'AMPERE MAX': 6.32, 'WATT MIN': 549.7, 'WATT MAX': 1453.6, 'BETA MIN': -3, 'BETA MAX': 25, 'DELTA MIN': -17, 'DELTA MAX': 22, 'ALPHA MIN': -10, 'ALPHA MAX': 10},
        'NON INV 1.5 PK': {'VOLT MIN': 196, 'VOLT MAX': 265, 'AMPERE MIN': 3.15, 'AMPERE MAX': 7.94, 'WATT MIN': 724.5, 'WATT MAX': 1826.2, 'BETA MIN': -3, 'BETA MAX': 25, 'DELTA MIN': -17, 'DELTA MAX': 22, 'ALPHA MIN': -10, 'ALPHA MAX': 10},
        'NON INV 2 PK': {'VOLT MIN': 196, 'VOLT MAX': 265, 'AMPERE MIN': 5.06, 'AMPERE MAX': 12.15, 'WATT MIN': 1163.8, 'WATT MAX': 2794.5, 'BETA MIN': -3, 'BETA MAX': 25, 'DELTA MIN': -17, 'DELTA MAX': 22, 'ALPHA MIN': -10, 'ALPHA MAX': 10},
        'NON INV 2.5 PK': {'VOLT MIN': 196, 'VOLT MAX': 265, 'AMPERE MIN': 6.75, 'AMPERE MAX': 15.45, 'WATT MIN': 1552.5, 'WATT MAX': 3553.5, 'BETA MIN': -3, 'BETA MAX': 25, 'DELTA MIN': -17, 'DELTA MAX': 22, 'ALPHA MIN': -10, 'ALPHA MAX': 10},
        'INV 1/2 PK': {'VOLT MIN': 196, 'VOLT MAX': 265, 'AMPERE MIN': 1.15, 'AMPERE MAX': 3.96, 'WATT MIN': 264.5, 'WATT MAX': 910.8, 'BETA MIN': -3, 'BETA MAX': 25, 'DELTA MIN': -17, 'DELTA MAX': 22, 'ALPHA MIN': -10, 'ALPHA MAX': 10},
        'INV 3/4 PK': {'VOLT MIN': 196, 'VOLT MAX': 265, 'AMPERE MIN': 1.75, 'AMPERE MAX': 3.75, 'WATT MIN': 402.5, 'WATT MAX': 862.5, 'BETA MIN': -3, 'BETA MAX': 25, 'DELTA MIN': -17, 'DELTA MAX': 22, 'ALPHA MIN': -10, 'ALPHA MAX': 10},
        'INV 1 PK': {'VOLT MIN': 196, 'VOLT MAX': 265, 'AMPERE MIN': 1.18, 'AMPERE MAX': 6.28, 'WATT MIN': 271.4, 'WATT MAX': 1444.4, 'BETA MIN': -3, 'BETA MAX': 25, 'DELTA MIN': -17, 'DELTA MAX': 22, 'ALPHA MIN': -10, 'ALPHA MAX': 10},
        'INV 1.5 PK': {'VOLT MIN': 196, 'VOLT MAX': 265, 'AMPERE MIN': 2.04, 'AMPERE MAX': 9.55, 'WATT MIN': 469.2, 'WATT MAX': 2196.5, 'BETA MIN': -3, 'BETA MAX': 25, 'DELTA MIN': -17, 'DELTA MAX': 22, 'ALPHA MIN': -10, 'ALPHA MAX': 10},
        'INV 2 PK': {'VOLT MIN': 196, 'VOLT MAX': 265, 'AMPERE MIN': 2.48, 'AMPERE MAX': 12.62, 'WATT MIN': 570.4, 'WATT MAX': 2902.6, 'BETA MIN': -3, 'BETA MAX': 25, 'DELTA MIN': -17, 'DELTA MAX': 22, 'ALPHA MIN': -10, 'ALPHA MAX': 10},
        'INV 2.5 PK': {'VOLT MIN': 196, 'VOLT MAX': 265, 'AMPERE MIN': 5.60, 'AMPERE MAX': 17.73, 'WATT MIN': 1288, 'WATT MAX': 4077.9, 'BETA MIN': -3, 'BETA MAX': 25, 'DELTA MIN': -17, 'DELTA MAX': 22, 'ALPHA MIN': -10, 'ALPHA MAX': 10},
    }

    # Helper mapping to get the correct rule key from the 'Unit' column in the CSV
    unit_name_to_rule_key_map = {
        'NON INV 0.5PK': 'NON INV 1/2 PK', 'NON INV 0.75PK': 'NON INV 3/4 PK', 'NON INV 1PK': 'NON INV 1 PK',
        'NON INV 1.5PK': 'NON INV 1.5 PK', 'NON INV 2PK': 'NON INV 2 PK', 'NON INV 2.5PK': 'NON INV 2.5 PK',
        'INV 0.5PK': 'INV 1/2 PK', 'INV 0.75PK': 'INV 3/4 PK', 'INV 1PK': 'INV 1 PK', 'INV 1.5PK': 'INV 1.5 PK',
        'INV 2PK': 'INV 2 PK', 'INV 2.5PK': 'INV 2.5 PK'
    }

    # --- 2. Load the Dataset ---
    try:
        df = pd.read_csv(input_filepath)
        print(f"Successfully loaded dataset from '{input_filepath}'. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_filepath}'.")
        print("Please run the data generation script first or check the file path.")
        return None

    # --- 3. Define the Custom Scaling Functions ---
    def voltage_scaler_explicit(row):
        """
        Applies linear scaling to Voltage based on its raw value.
        
        This function's logic relies on the data generator:
        - If the generator created a 'Trouble 3' row, the raw 'Voltage (V)'
          is already outside the normal 196-265V range.
        - This calculation simply translates that raw value to the scaled space.
        - A raw value < 196 will result in a scaled value < 0.
        - A raw value > 265 will result in a scaled value > 1.
        - A raw value within [196, 265] results in a scaled value in [0, 1].
        """
        v_min = 196
        v_max = 265
        v_range = v_max - v_min if v_max > v_min else 1
        return (row['Voltage (V)'] - v_min) / v_range

    def tiered_scaler(value, unit_name, param_prefix):
        """Applies the piecewise scaling based on the unit's rules."""
        rule_key = unit_name_to_rule_key_map.get(unit_name, unit_name)
        
        n_min = normal_rules[rule_key][f'{param_prefix} MIN']
        n_max = normal_rules[rule_key][f'{param_prefix} MAX']
        m_min = maintenance_rules[rule_key][f'{param_prefix} MIN']
        m_max = maintenance_rules[rule_key][f'{param_prefix} MAX']
        
        n_range = n_max - n_min if n_max > n_min else 1
        m_range_low = n_min - m_min if n_min > m_min else 1
        m_range_high = m_max - n_max if m_max > n_max else 1
        
        if n_min <= value <= n_max:
            return (value - n_min) / n_range
        elif m_min <= value < n_min:
            return -1 + (value - m_min) / m_range_low
        elif n_max < value <= m_max:
            return 1 + (value - n_max) / m_range_high
        elif value < m_min:
            return -1 + (value - m_min) / m_range_low
        elif value > m_max:
            return 1 + (value - n_max) / m_range_high
        else:
            return np.nan

    # --- 4. Apply the Scalers to Relevant Columns ---
    print("Applying scaling to the dataset...")
    
    df_scaled = df.copy()

    # Apply special simple scaler for Voltage
    print("  - Applying explicit logic scaling to 'Voltage (V)' -> 'Voltage_scaled'")
    df_scaled['Voltage_scaled'] = df_scaled.apply(
        voltage_scaler_explicit,
        axis=1
    )

    # Identify columns for tiered scaling
    params_to_scale_tiered = {
        'Current (A)': 'AMPERE',
        'Wattage (W)': 'WATT',
        'Alpha (°C)': 'ALPHA',
        'Beta (°C)': 'BETA',
        'Delta (°C)': 'DELTA'
    }

    for col, prefix in params_to_scale_tiered.items():
        new_col_name = f"{col.split(' ')[0]}_scaled"
        print(f"  - Applying tiered scaling to '{col}' -> '{new_col_name}'")
        df_scaled[new_col_name] = df_scaled.apply(
            lambda row: tiered_scaler(row[col], row['Unit'], prefix),
            axis=1
        )
    
    # --- 5. Final Assembly and Output ---
    print("Finalizing the preprocessed DataFrame...")
    
    # Reordering columns to place scaled values next to originals for easy comparison.
    all_scaled_cols = ['Voltage_scaled'] + [f"{c.split(' ')[0]}_scaled" for c in params_to_scale_tiered.keys()]
    original_cols = list(df.columns)
    
    final_col_order = []
    for col in original_cols:
        final_col_order.append(col)
        scaled_col_name = f"{col.split(' ')[0]}_scaled"
        if scaled_col_name in all_scaled_cols:
            final_col_order.append(scaled_col_name)
            
    final_col_order = list(dict.fromkeys(final_col_order))
    df_processed = df_scaled[final_col_order]

    try:
        output_dir = os.path.dirname(output_filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        df_processed.to_csv(output_filepath, index=False)
        print(f"\nSuccessfully preprocessed and saved dataset. File saved to: '{output_filepath}'")
    except Exception as e:
        print(f"Error saving file: {e}")

    return df_processed

#%%

if __name__ == '__main__':
    processed_df = preprocess_ac_data_tiered_scaling()
    if processed_df is not None:
        print("✅ ScaledSyntheticTrainingDataset.csv generated successfully.")



