# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 23:15:58 2025

@author: Raynard
"""

"""
SyntheticDataGenerator_v6.py
Rule-based, per-condition evolutionary synthetic data generator.
Outputs columns:
Time,Current (A),Voltage (V),Wattage (W),Temperature Indoor (°C),
Temperature Outdoor (°C),Set Point (°C),Alpha (°C),Supply (°C),Return (°C),
Delta (°C),Beta (°C),Cycle,Unit,Condition
"""

import pandas as pd
import numpy as np
import os
import random

class SyntheticDataGenerator:
    def __init__(self, baseline_path):
        self.baseline = pd.read_csv(baseline_path)
        print(f"[SyntheticDataGenerator] Baseline loaded: {self.baseline.shape}")

    # Columns touched by every mutation (multiplicative rule_exp + additive
    # gaussian noise scaled by fault_int).
    PERTURB_COLS = [
        "Current_scaled", "Voltage_scaled", "Wattage_scaled",
        "Alpha_scaled", "Beta_scaled", "Delta_scaled",
        'Current (A)', 'Voltage (V)', 'Wattage (W)',
        'Alpha (°C)', 'Beta (°C)', 'Delta (°C)', 'Supply (°C)', 'Return (°C)',
    ]

    # Subset of PERTURB_COLS that also receive the additive `thermal` bias.
    THERMAL_COLS = {
        "Alpha_scaled", "Beta_scaled", "Delta_scaled",
        'Alpha (°C)', 'Beta (°C)', 'Delta (°C)', 'Supply (°C)', 'Return (°C)',
    }

    # Safety clamp for *_scaled columns only (raw sensor columns don't have
    # a fixed cross-unit range, so they aren't clamped here). Normal tier is
    # [0, 1], maintenance extends to roughly [-1, 2]; this bound is a
    # generous margin beyond "trouble" that still allows exploration while
    # preventing rule_exp from compounding into runaway values across many
    # accepted generations (see `generator.baseline = evolved_df` in
    # CoevolutionLoop.py). Tune if it's clipping legitimate exploration.
    SCALED_SAFETY_RANGE = (-3.0, 4.0)

    def evolve_dataset(self, params, output_path):
        """Generate new dataset variant based on evolution parameters."""
        df = self.baseline.copy()

        # Apply evolution transformations
        rule_exp = params.get("rule_exp", 1.0)     # multiplicative scale
        noise_amp = params.get("fault_int", 1.0)   # noise amplitude
        # rel_freq: fraction of ROWS this proposal actually perturbs. Rows
        # not selected keep their baseline values. This lets the controller
        # search "how often" a rule-shift shows up, independent of how
        # strong it is (rule_exp/fault_int) - previously rel_freq was
        # proposed/scored by the controller but never used here.
        rel_freq = float(np.clip(params.get("rel_freq", 1.0), 0.0, 1.0))
        # thermal: additive bias applied only to temperature-related columns
        # - previously proposed/scored by the controller but never used here.
        thermal = params.get("thermal", 0.0)

        n_rows = len(df)
        mutate_mask = np.random.rand(n_rows) < rel_freq

        for col in self.PERTURB_COLS:
            if col not in df.columns:
                continue
            noise = np.random.normal(0, 0.02 * noise_amp, size=n_rows)
            new_values = df[col] * rule_exp + noise
            if col in self.THERMAL_COLS:
                new_values = new_values + thermal
            if col.endswith("_scaled"):
                new_values = new_values.clip(*self.SCALED_SAFETY_RANGE)
            df.loc[mutate_mask, col] = new_values[mutate_mask]

        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        df.to_csv(output_path, index=False)
        print(f"[SyntheticDataGenerator] Saved evolved dataset ({len(df)} rows, "
              f"{mutate_mask.sum()} mutated) at: {output_path}")
        return df

