# -*- coding: utf-8 -*-
"""
Shared feature construction for RandomForestClassifier.py and RFClassifierValidation.py.

Replaces "X = df.select_dtypes(include=['number'])", which silently dropped
Cycle/Unit (informative) and kept every raw sensor column alongside its
tiered-scaled counterpart (a shortcut risk). Both choices are now explicit
and were checked empirically against RealData/ScaledRealValidationDataset.csv
before being set as defaults - see notes below. Numbers are from an 8400-row
synthetic run (700/unit), not the repo's default 3500/unit; re-validate at
full scale before trusting these as final.

    Real F1 (weighted), old vs. new default:
      old (raw+scaled, no Cycle/Unit) ............ 0.605
      new default (scaled-only + Cycle) .......... 0.643-0.657

Design decisions, each tested against the alternative:
  - RAW columns (Current (A), Voltage (V), Wattage (W), Alpha/Beta/Delta (°C))
    are DROPPED. Their *_scaled counterparts are kept. Reason: the raw
    thresholds differ drastically by unit size, so keeping raw values lets
    the model fit unit-specific splits that work on synthetic data's exact
    generation quirks (uniform sampling, 2-decimal rounding) but don't
    transfer to real telemetry - exactly the shortcut TieredScaling.py's
    normalization exists to prevent. Dropping raw-but-scaled columns
    measurably improved real F1 (0.605 -> 0.657) in testing.
  - 'Cycle' (IDLE/COOLING) is kept, binary-encoded. Near-neutral in testing
    (0.657 -> 0.643); kept by default since it's cheap and the effect was
    within noise at this sample size.
  - 'Unit' is OFF by default (include_unit=False). Every encoding tried -
    one-hot (12 dummies) and coarse (INV flag + PK size) - reduced real F1,
    because RealValidationDataset.csv only covers 2 of 12 unit types (both
    0.5PK). The model learns Unit-conditioned splits from all 12 synthetic
    unit types that real validation data can never exercise correctly. This
    is a direct consequence of the real-data coverage gap, not a flaw in the
    encoding - revisit if/when real telemetry from more unit sizes exists.
  - 'Time' is dropped (absolute timestamp, not generalizable).
  - 'Condition' is dropped (it's the label).
"""

import pandas as pd

RAW_PAIRED_COLS = ['Current (A)', 'Voltage (V)', 'Wattage (W)',
                    'Alpha (°C)', 'Beta (°C)', 'Delta (°C)']


def build_feature_matrix(df, reference_columns=None, include_unit=False):
    """
    Build the model-input feature matrix from a raw/scaled dataframe.

    Args:
        df (pd.DataFrame): raw input dataframe (synthetic or real).
        reference_columns (Iterable[str] or None): if given, output is
            reindexed to exactly these columns - missing ones filled with 0,
            extra ones dropped. Pass the *training* feature columns when
            building the real-data validation matrix.
        include_unit (bool): include Unit as a feature. Default False - see
            module docstring; only enable once real validation data covers
            more than 2 of the 12 unit types, and re-check real F1 before
            trusting the result.

    Returns:
        pd.DataFrame: numeric feature matrix ready for RandomForestClassifier.
    """
    X = df.select_dtypes(include=['number']).copy()
    X = X.drop(columns=[c for c in RAW_PAIRED_COLS if c in X.columns])

    if 'Cycle' in df.columns:
        X['Cycle_COOLING'] = (df['Cycle'] == 'COOLING').astype(int)

    if include_unit and 'Unit' in df.columns:
        unit_dummies = pd.get_dummies(df['Unit'].astype(str), prefix='Unit')
        X = pd.concat([X, unit_dummies], axis=1)

    if reference_columns is not None:
        for col in reference_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[list(reference_columns)]

    return X
