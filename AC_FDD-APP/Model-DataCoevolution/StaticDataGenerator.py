# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 23:15:58 2025

@author: Raynard
"""

import pandas as pd
import numpy as np
import datetime
import os
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_ac_dataset(num_samples_per_unit=3500, output_filepath=os.path.join(SCRIPT_DIR,"SyntheticStatic", "SyntheticTrainingDataset.csv")):
    """
    Generates a balanced, synthetic dataset for AC unit fault detection
    based on a tiered set of expert-defined rules for 7 distinct conditions.

    The logic is now tiered:
    - NORMAL: Within NORMAL parameter ranges.
    - MAINTENANCE/ABNORMAL: Outside NORMAL, but inside MAINTENANCE ranges.
    - TROUBLE: Outside MAINTENANCE ranges.
    - Wattage is calculated from Voltage * Current.
    - Inverter (INV) unit behavior is now a stateful simulation based on Alpha.
    - Non-inverter cycles are determined by a stateful indoor temperature simulation.
    - Supply Temperature is now determined by dynamic rules for each unit type.
    - Outdoor Temperature follows a realistic, stepped 24-hour cycle and is independent of fault conditions.
    - Set points are deterministically assigned and sorted for each condition block.
    - All numeric values are rounded to 2 decimal places.
    - The final dataset is sorted by Unit, then by Condition.
    - Fault conditions now use complex AND/OR logic for realism.

    Args:
        num_samples_per_unit (int): The total number of data points to generate
                                    for each type of AC unit. Should be divisible by 7.
        output_filepath (str): The path to save the generated CSV file.

    Returns:
        pandas.DataFrame: The generated synthetic dataset.
    """
    # --- Helper function for time interval calculation ---
    def _calculate_next_change(current_time, base_minutes, tolerance_minutes):
        """Calculates the next event time based on a base and tolerance."""
        random_minutes = base_minutes + np.random.uniform(-tolerance_minutes, tolerance_minutes)
        # Ensure the interval is at least one minute
        interval_minutes = max(1, random_minutes)
        return current_time + datetime.timedelta(minutes=interval_minutes)

    # --- 1. Define Tiered Expert Rules ---

    # Tier 1: NORMAL Operating Parameters
    NORMAL_rules = {
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

    # Tier 2: MAINTENANCE Parameter Ranges
    MAINTENANCE_rules = {
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

    unit_name_mapping_inv = {v: k for k, v in {
        'NON INV 0.5PK': 'NON INV 1/2 PK', 'NON INV 0.75PK': 'NON INV 3/4 PK', 'NON INV 1PK': 'NON INV 1 PK',
        'NON INV 1.5PK': 'NON INV 1.5 PK', 'NON INV 2PK': 'NON INV 2 PK', 'NON INV 2.5PK': 'NON INV 2.5 PK',
        'INV 0.5PK': 'INV 1/2 PK', 'INV 0.75PK': 'INV 3/4 PK', 'INV 1PK': 'INV 1 PK', 'INV 1.5PK': 'INV 1.5 PK',
        'INV 2PK': 'INV 2 PK', 'INV 2.5PK': 'INV 2.5 PK'
    }.items()}
    
    # --- 2. Data Generation Setup ---
    conditions = ['NORMAL', 'MAINTENANCE 1', 'MAINTENANCE 2', 'ABNORMAL', 'TROUBLE 1', 'TROUBLE 2', 'TROUBLE 3']
    
    total_rows = len(NORMAL_rules) * num_samples_per_unit
    start_time = datetime.datetime(2025, 5, 1)
    time_series = pd.to_datetime(pd.date_range(start=start_time, periods=total_rows, freq='T'))
    
    if num_samples_per_unit % len(conditions) != 0:
        raise ValueError(f"'num_samples_per_unit' must be divisible by the number of conditions ({len(conditions)}).")
    
    samples_per_condition = num_samples_per_unit // len(conditions)

    # --- Pre-generate Outdoor Temperature Timeseries with Stepping Logic ---
    print("Pre-generating realistic outdoor temperature curve...")
    outdoor_temps = []
    current_temp = 26.0
    next_change_time = _calculate_next_change(start_time, 14, 1)

    for t in time_series:
        hour = t.hour + t.minute / 60.0
        if t >= next_change_time:
            direction, interval_base, tolerance = ('none', 0, 0)
            if 0 <= hour < 6: direction, interval_base, tolerance = 'decrease', 14, 1
            elif 6 <= hour < 9.75: direction, interval_base, tolerance = 'increase', 4, 1
            elif 9.75 <= hour < 13.75: direction, interval_base, tolerance = 'increase', 9, 1
            elif 13.75 <= hour < 18: direction, interval_base, tolerance = 'decrease', 7, 1
            else: direction, interval_base, tolerance = 'decrease', 15, 1
            
            if direction == 'increase': current_temp += 0.1
            elif direction == 'decrease': current_temp -= 0.1
            next_change_time = _calculate_next_change(t, interval_base, tolerance)
        outdoor_temps.append(current_temp)
    
    outdoor_df = pd.DataFrame({'Time': time_series, 'base_temp_outdoor': outdoor_temps})

    # --- Setup for Simulation ---
    print("Setting up simulation framework...")
    setup_data = []
    unit_names = [unit_name_mapping_inv.get(k, k) for k in NORMAL_rules.keys()]
    for unit_name in unit_names:
        for condition in conditions:
            # Create the sorted list of set points for this block
            set_point_values = sorted(list(range(18, 25)), reverse=True) # [24, 23, ..., 18]
            num_repeats = samples_per_condition // len(set_point_values)
            remainder = samples_per_condition % len(set_point_values)
            set_point_block = sorted(np.repeat(set_point_values, num_repeats).tolist() + set_point_values[:remainder], reverse=True)
            
            for i in range(samples_per_condition):
                setup_data.append({
                    'Unit': unit_name, 
                    'Condition': condition,
                    'Set Point (°C)': set_point_block[i]
                })
    
    setup_df = pd.DataFrame(setup_data)
    setup_df['Time'] = time_series
    setup_df = setup_df.sort_values(by='Time').reset_index(drop=True)

    # Initialize state for each individual AC unit
    unit_states = {}
    for unit_name in unit_names:
        unit_states[unit_name] = {
            'indoor_temp': 25.0,
            'cycle': 'IDLE',
            'next_change': start_time
        }

    # --- Main Simulation Loop ---
    print("Starting main dataset generation with simulation logic...")
    all_data = []
    for idx, row_setup in setup_df.iterrows():
        row = row_setup.to_dict()
        unit_name = row['Unit']
        condition = row['Condition']
        current_time = row['Time']
        set_point = row['Set Point (°C)']
        
        unit_NORMAL_rules = NORMAL_rules[next(k for k, v in unit_name_mapping_inv.items() if v == unit_name)]
        unit_maint_rules = MAINTENANCE_rules[next(k for k, v in unit_name_mapping_inv.items() if v == unit_name)]
        state = unit_states[unit_name]
        
        # --- State Machine for Indoor Temp and Cycle ---
        if 'NON INV' in unit_name:
            if current_time >= state['next_change']:
                if state['cycle'] == 'COOLING':
                    state['indoor_temp'] -= 0.1
                    base, tolerance = (6, 2) if set_point <= 21 else (4, 2)
                    if state['indoor_temp'] <= set_point - 1.0:
                        state['cycle'] = 'IDLE'
                else: # cycle == 'IDLE'
                    state['indoor_temp'] += 0.1
                    base, tolerance = (3, 1)
                    if state['indoor_temp'] >= set_point + 0.4:
                        state['cycle'] = 'COOLING'
                state['next_change'] = _calculate_next_change(current_time, base, tolerance)
            row['Cycle'] = state['cycle']
        
        elif 'INV' in unit_name:
            state['cycle'] = 'COOLING'
            row['Cycle'] = state['cycle']
            
            current_alpha = state['indoor_temp'] - set_point
            
            if current_alpha < 0:
                state['indoor_temp'] = set_point - 1.0
            elif current_time >= state['next_change']:
                state['indoor_temp'] -= 0.1
                
                base, tolerance = 0, 0
                if current_alpha >= 4 and set_point <= 21: base, tolerance = 6, 2
                elif current_alpha >= 0 and set_point <= 21: base, tolerance = 4, 2
                elif current_alpha >= 4 and set_point > 21: base, tolerance = 5, 2
                elif current_alpha >= 0 and set_point > 21: base, tolerance = 3, 2
                
                if base > 0:
                    state['next_change'] = _calculate_next_change(current_time, base, tolerance)
        
        # --- Generate Sensor Values based on State ---
        # New complex fault rules with AND/OR logic
        complex_fault_rules = {
            'MAINTENANCE 1': [[('BETA', '>MAX'), ('DELTA', '<MIN'), ('ALPHA', '>MAX')], [('AMPERE', '<MIN')]],
            'MAINTENANCE 2': [[('BETA', '>MAX'), ('DELTA', '>MAX'), ('ALPHA', '>MAX')], [('AMPERE', '>MAX')]],
            'ABNORMAL': [[('DELTA', '>MAX')], [('ALPHA', '>MAX')]],
            'TROUBLE 1': [[('BETA', '>MAX'), ('DELTA', '<MIN'), ('ALPHA', '>MAX')], [('AMPERE', '<MIN')]],
            'TROUBLE 2': [[('BETA', '>MAX'), ('DELTA', '>MAX'), ('ALPHA', '>MAX')], [('AMPERE', '>MAX')]],
            'TROUBLE 3': [[('VOLT', '<MIN'), ('VOLT', '>MAX')]]
        }
        
        active_faults = {}
        if condition in complex_fault_rules:
            rule_set = complex_fault_rules[condition]
            for or_group in rule_set:
                param, direction = random.choice(or_group)
                active_faults[param] = direction

        severity = 'NORMAL'
        if condition in ['MAINTENANCE 1', 'MAINTENANCE 2', 'ABNORMAL']: severity = 'MAINTENANCE'
        elif condition in ['TROUBLE 1', 'TROUBLE 2', 'TROUBLE 3']: severity = 'TROUBLE'

        base_temp_outdoor = outdoor_df.loc[idx, 'base_temp_outdoor']
        row['Temperature Outdoor (°C)'] = base_temp_outdoor

        if row['Cycle'] == 'IDLE':
            row['Current (A)'] = 0.075 * np.random.uniform(0.9, 1.1)
            row['Voltage (V)'] = np.random.uniform(unit_NORMAL_rules['VOLT MIN'], unit_NORMAL_rules['VOLT MAX'])
            row['Temperature Indoor (°C)'] = state['indoor_temp']
            row['Supply (°C)'] = state['indoor_temp']
            row['Return (°C)'] = state['indoor_temp'] + np.random.uniform(-0.5, 0.5)

        else: # cycle == 'COOLING'
            def generate_tiered_value(param_prefix, fault_direction, base_val_func):
                n_min, n_max = unit_NORMAL_rules[f'{param_prefix} MIN'], unit_NORMAL_rules[f'{param_prefix} MAX']
                m_min, m_max = unit_maint_rules[f'{param_prefix} MIN'], unit_maint_rules[f'{param_prefix} MAX']
                if severity == 'NORMAL' or not fault_direction: return base_val_func()
                if fault_direction == '<MIN':
                    return np.random.uniform(m_min, n_min) if severity == 'MAINTENANCE' else np.random.uniform(m_min - abs(m_min)*0.2, m_min)
                elif fault_direction == '>MAX':
                    return np.random.uniform(n_max, m_max) if severity == 'MAINTENANCE' else np.random.uniform(m_max, m_max * 1.2)
                return base_val_func()

            row['Voltage (V)'] = generate_tiered_value('VOLT', active_faults.get('VOLT'), lambda: np.random.uniform(unit_NORMAL_rules['VOLT MIN'], unit_NORMAL_rules['VOLT MAX']))
            row['Current (A)'] = generate_tiered_value('AMPERE', active_faults.get('AMPERE'), lambda: np.random.uniform(unit_NORMAL_rules['AMPERE MIN'], unit_NORMAL_rules['AMPERE MAX']))
            
            row['Temperature Indoor (°C)'] = state['indoor_temp']
            
            current_alpha = state['indoor_temp'] - set_point
            if 'NON INV' in unit_name:
                if state['indoor_temp'] >= set_point - 1:
                    supply = set_point - 6.0
                else: 
                    supply = set_point - np.random.uniform(2, 5) 
            else: # Inverter Unit
                if current_alpha >= 4: supply = set_point - 6.0
                elif current_alpha >= 0: supply = set_point - 3.0
                else: supply = set_point
            row['Supply (°C)'] = supply

            delta_val = generate_tiered_value('DELTA', active_faults.get('DELTA'), lambda: np.random.uniform(unit_NORMAL_rules['DELTA MIN'], unit_NORMAL_rules['DELTA MAX']))
            row['Return (°C)'] = row['Supply (°C)'] + delta_val

        row['Wattage (W)'] = row['Voltage (V)'] * row['Current (A)']
        row['Alpha (°C)'] = row['Temperature Indoor (°C)'] - row['Set Point (°C)']
        row['Beta (°C)'] = row['Temperature Outdoor (°C)'] - row['Set Point (°C)']
        row['Delta (°C)'] = row['Return (°C)'] - row['Supply (°C)']
        
        all_data.append(row)
        unit_states[unit_name] = state

    # --- 7. Final Assembly and Output ---
    print("Assembling final DataFrame...")
    final_columns = [
        'Time', 'Current (A)', 'Voltage (V)', 'Wattage (W)', 'Temperature Indoor (°C)', 'Temperature Outdoor (°C)', 
        'Set Point (°C)', 'Alpha (°C)', 'Supply (°C)', 'Return (°C)', 'Delta (°C)', 'Beta (°C)', 'Cycle', 'Unit', 'Condition'
    ]
    df = pd.DataFrame(all_data)
    
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    df[numeric_cols] = df[numeric_cols].round(2)
    
    df = df[final_columns]
    
    unit_order = [unit_name_mapping_inv.get(k, k) for k in NORMAL_rules.keys()]
    condition_order = conditions

    df['Unit'] = pd.Categorical(df['Unit'], categories=unit_order, ordered=True)
    df['Condition'] = pd.Categorical(df['Condition'], categories=condition_order, ordered=True)

    df = df.sort_values(by=['Unit', 'Condition', 'Time']).reset_index(drop=True)

    try:
        output_dir = os.path.dirname(output_filepath)
        if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)
        df.to_csv(output_filepath, index=False)
        print(f"\nSuccessfully generated dataset with {len(df)} rows. File saved to: '{output_filepath}'")
    except Exception as e:
        print(f"Error saving file: {e}")

    return df

#%%

if __name__ == '__main__':
    generated_df = generate_ac_dataset()
    if generated_df is not None:
        print("\n--- Value Counts for 'Condition' ---")
        print(generated_df['Condition'].value_counts().sort_index())
        print(generated_df['Unit'].value_counts().sort_index())
        print(generated_df['Cycle'].value_counts().sort_index())


