# -*- coding: utf-8 -*-

import os
import json
import joblib
import pandas as pd
import numpy as np

from SyntheticDataGenerator import SyntheticDataGenerator
from RandomForestClassifier import RFModel
from RFClassifierValidation import RFValidator
from CoevolutionController import CoevolutionController
from StaticDataGenerator import generate_ac_dataset
from TieredScaling import preprocess_ac_data_tiered_scaling

generate_ac_dataset(num_samples_per_unit=3500, 
                    output_filepath=r'R:/Mine/AllOfMe/T.A. Penelitian/ML_Model/Experiments/UpdatedExperiments/Model-DataCoevolution/SyntheticStatic/SyntheticTrainingDataset.csv')

preprocess_ac_data_tiered_scaling(input_filepath='R:/Mine/AllOfMe/T.A. Penelitian/ML_Model/Experiments/UpdatedExperiments/Model-DataCoevolution/SyntheticStatic/SyntheticTrainingDataset.csv',
                                  output_filepath='R:/Mine/AllOfMe/T.A. Penelitian/ML_Model/Experiments/UpdatedExperiments/Model-DataCoevolution/SyntheticStatic/ScaledSyntheticTrainingDataset.csv')

# --- Feature Divergence Function (Simplified) ---
def feature_divergence_score(real_df, synth_df):
    """
    Calculates a simple divergence score.

    Prefers the *_scaled columns (what RandomForestClassifier/FeatureUtils
    actually train on) over raw sensor columns. Raw columns are deliberately
    dropped from the model's feature matrix (see FeatureUtils.py docstring)
    because their thresholds aren't comparable across unit sizes - scoring
    divergence on them would penalize/reward drift the model never sees.
    """
    real_numeric = real_df.select_dtypes(include=np.number)
    synth_numeric = synth_df.select_dtypes(include=np.number)

    scaled_cols = [c for c in real_numeric.columns if c.endswith("_scaled")]
    common_cols = [c for c in scaled_cols if c in synth_numeric.columns]

    if not common_cols:
        # Fall back to raw numeric columns, but flag it - this changes what
        # the divergence term in the hybrid score is actually measuring.
        common_cols = list(set(real_numeric.columns) & set(synth_numeric.columns))
        if common_cols:
            print("[Divergence] WARNING: no shared '_scaled' columns between "
                  "real and synthetic data - falling back to raw numeric "
                  "columns, which the model does not train on.")

    if not common_cols:
        return 1.0  # Max divergence if no columns match

    real_means = real_numeric[common_cols].mean()
    synth_means = synth_numeric[common_cols].mean()

    abs_diff = (real_means - synth_means).abs()
    real_scale = real_means.abs().mean() + 1e-6  # Avoid division by zero

    normalized_divergence = (abs_diff / real_scale).mean()
    return normalized_divergence


# --- Pareto Front Manager ---
class ParetoManager:
    """Tracks the set of best non-dominated solutions (the Pareto front)."""
    def __init__(self, save_path):
        self.save_path = save_path
        self.front = []  # List of solution dictionaries

    def dominates(self, a, b):
        """Checks if solution 'a' dominates solution 'b'."""
        better_or_equal = (a["f1"] >= b["f1"] and a["precision"] >= b["precision"] and a["recall"] >= b["recall"]
        )
        strictly_better = (a["f1"] > b["f1"] and (a["precision"] > b["precision"] or a["recall"] > b["recall"])
        )
        return better_or_equal and strictly_better

    def update(self, candidate):
        """
        Adds a new candidate solution to the Pareto front, removing any
        solutions it now dominates.
        """
        new_front = [c for c in self.front if not self.dominates(candidate, c)]
        dominated_by_existing = any(self.dominates(c, candidate) for c in self.front)

        if not dominated_by_existing:
            new_front.append(candidate)

        self.front = new_front

        try:
            with open(self.save_path, "w") as f:
                json.dump(self.front, f, indent=4)
        except Exception as e:
            print(f"[Pareto] Error saving front: {e}")


# --- Composite + Hybrid Score ---
def calculate_hybrid_score(report, divergence):
    """
    Calculates the composite and hybrid scores based on your defined weights.
    """
    f1 = report["weighted avg"]["f1-score"]
    precision = report["weighted avg"]["precision"]
    recall = report["weighted avg"]["recall"]
    composite = 0.2 * f1 + 0.3 * precision + 0.5 * recall
    hybrid = 0.9 * composite + 0.1 * (1 - divergence)
    return hybrid, composite, f1, precision, recall


# --- Configuration ---
# Each path can be overridden with an environment variable (e.g. for running
# on a different machine or in CI) without editing this file; the string
# literals remain the defaults you had before.
BASELINE_SYNTH_PATH = os.environ.get(
    "COEVO_BASELINE_SYNTH_PATH",
    r"R:\Mine\AllOfMe\T.A. Penelitian\ML_Model\Experiments\UpdatedExperiments\Model-DataCoevolution\SyntheticStatic\ScaledSyntheticTrainingDataset.csv",
)
REAL_VALID_PATH = os.environ.get(
    "COEVO_REAL_VALID_PATH",
    r"R:\Mine\AllOfMe\T.A. Penelitian\ML_Model\RealData\Dataset_Test_30.csv",
)
OUTPUT_DIR = os.environ.get(
    "COEVO_OUTPUT_DIR",
    r"R:\Mine\AllOfMe\T.A. Penelitian\ML_Model\Experiments\UpdatedExperiments\Model-DataCoevolution\CoevolutionResult",
)

MAX_ITER = int(os.environ.get("COEVO_MAX_ITER", 210))
EARLY_STOP_PATIENCE = int(os.environ.get("COEVO_EARLY_STOP_PATIENCE", 70))

# --- Create output directory ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pkl")
BEST_SYNTH_PATH = os.path.join(OUTPUT_DIR, "best_synthetic_data.csv")
BEST_PARAMS_PATH = os.path.join(OUTPUT_DIR, "best_controller_params.json")
PARETO_FRONT_PATH = os.path.join(OUTPUT_DIR, "pareto_front_solutions.json")
HISTORY_PATH = os.path.join(OUTPUT_DIR, "coevolution_history.json")

# --- Initialize Components ---
print("=== Starting Model–Data Coevolution (Simulated Annealing) ===")
generator = SyntheticDataGenerator(BASELINE_SYNTH_PATH)
controller = CoevolutionController(
    initial_step_size=0.1,
    start_temperature=0.2,
    cooling_rate=0.98,
    random_state=42,
)
rf_trainer = RFModel()
validator = RFValidator(REAL_VALID_PATH, output_dir=OUTPUT_DIR)
pareto = ParetoManager(PARETO_FRONT_PATH)

try:
    real_data = pd.read_csv(REAL_VALID_PATH)
    print(f"Loaded real validation data for divergence calculation: {real_data.shape}")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load real validation data from {REAL_VALID_PATH}. {e}")
    exit()


# --- Main Loop ---
history = []

for iteration in range(1, MAX_ITER + 1):
    print(f"\n--- Iteration {iteration}/{MAX_ITER} ---")

    # 1. Controller PROPOSES new parameters
    params = controller.propose_new_params()

    synth_path = os.path.join(OUTPUT_DIR, f"synthetic_iter_{iteration}.csv")

    # 2. Generate, Train, and Validate
    try:
        evolved_df = generator.evolve_dataset(params, synth_path)
        model, f1_train = rf_trainer.train(evolved_df)
        report = validator.validate(model, feature_reference_df=evolved_df)
    except Exception as e:
        print(f"[Loop] ERROR during train/validate step: {e}. Skipping iteration.")
        controller.accept_or_reject_proposal(-np.inf)
        continue

    # 3. Calculate Scores
    divergence = feature_divergence_score(real_data, evolved_df)
    hybrid, composite, f1, precision, recall = calculate_hybrid_score(report, divergence)

    print(
        f"[Loop] Score: Hybrid={hybrid:.4f} | Composite={composite:.4f} | Divergence={divergence:.4f}"
    )

    # 4. Controller DECIDES to accept or reject the proposal
    controller.accept_or_reject_proposal(hybrid)

    # 5. Update Pareto Front
    # NOTE: use `params` (the proposal actually tested this iteration), not
    # controller.current_params - on a rejected iteration current_params
    # reverts to the *previous* accepted state, which would otherwise mean
    # the params stored here don't match the f1/precision/recall next to them.
    candidate = {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "iteration": iteration,
        "hybrid_score": hybrid,
        "params": params,
    }
    pareto.update(candidate)

    # 6. Check if the ALL-TIME best score was updated this iteration
    if controller.best_score == hybrid:
        print(f"[Loop] 🔥 New best Hybrid Score found={hybrid:.4f} (F1={f1:.4f})")

        joblib.dump(model, BEST_MODEL_PATH)
        evolved_df.to_csv(BEST_SYNTH_PATH, index=False)
        with open(BEST_PARAMS_PATH, "w") as f: json.dump(controller.best_params, f, indent=4)

        validator.validate(
            model,
            feature_reference_df=evolved_df,
            save_best=True,
            normalize=False  
        )

        generator.baseline = evolved_df

    # 7. Log History
    history.append(
        {
            "iteration": iteration,
            "f1_train": f1_train,
            "f1_real": f1,
            "precision": precision,
            "recall": recall,
            "divergence": divergence,
            "hybrid_score_tested": hybrid,
            "current_accepted_score": controller.current_score,
            "best_all_time_score": controller.best_score,
            "temperature": controller.temperature,
            "params_tested": params,
            "classification_report": report,
        }
    )
    try:
        with open(HISTORY_PATH, "w") as f:
            json.dump(history, f, indent=4)
    except Exception as e:
        print(f"[History] Error saving history: {e}")

    # 8. Check for Early Stopping
    if iteration > EARLY_STOP_PATIENCE:
        recent_best_scores = [h["best_all_time_score"] for h in history[-EARLY_STOP_PATIENCE:]]
        if len(set(recent_best_scores)) == 1:
            print(
                f"[Loop] Early stopping triggered: Best score has not changed for {EARLY_STOP_PATIENCE} iterations."
            )
            break

print("\n=== Coevolution Complete (Simulated Annealing) ===")
print(f"Best Hybrid Score Found: {controller.best_score:.4f}")
print(f"Best Model Path: {BEST_MODEL_PATH}")
print(f"Best Synthetic Data: {BEST_SYNTH_PATH}")
print(f"Best Parameters saved to: {BEST_PARAMS_PATH}")
print(f"Pareto Front saved to: {PARETO_FRONT_PATH}")