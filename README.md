# ACFDD_ModelDeploymentDemo
Source code and experiments applying Random Forest algorithms to the Air Conditioner FDD problem space.

## 1. Dataset Description
The synthetic data source provided for training and demonstration in this repository is **"best_synthetic_data.csv"**. This dataset is rule-based, fully synthetic data that has been engineered to simulate the complex, real-world thermodynamic and electrical patterns inherent to AC operations and faults. It serves as a highly representative proxy for real-world sensor logs, allowing users to train, test, and evaluate the fault diagnostic pipeline right out of the box.

## 2. Feature Input
The `best_synthetic_data.csv` file contains the following sequential time-series columns, which include raw sensor readings, engineered thermodynamic features, and categorical metadata:

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| **Time** | `Datetime` | The chronological timestamp of the sensor reading (1-minute intervals). |
| **Current (A)** | `Float` | The electrical current drawn by the air conditioning unit. |
| **Voltage (V)** | `Float` | The input supply voltage to the unit. |
| **Wattage (W)** | `Float` | The total power consumption of the unit. |
| **Temperature Indoor (°C)** | `Float` | The ambient room temperature measured by the indoor unit. |
| **Temperature Outdoor (°C)** | `Float` | The ambient environmental temperature measured at the outdoor condenser. |
| **Set Point (°C)** | `Float` | The target cooling temperature configured on the remote control. |
| **Alpha (°C)** | `Float` | *Engineered Feature:* Defined strictly as `Temperature Indoor - Set Point`. Indicates how far the room is from the target temperature. |
| **Supply (°C)** | `Float` | The temperature of the chilled air exiting the indoor unit's louvers. |
| **Return (°C)** | `Float` | The temperature of the room air being pulled back into the indoor unit. |
| **Delta (°C)** | `Float` | *Engineered Feature:* The temperature split across the evaporator coil, calculated as `Return - Supply`. |
| **Beta (°C)** | `Float` | *Engineered Feature:* Defined strictly as `Temperature Outdoor - Temperature Indoor`. Indicates the thermal load differential between the inside and outside environments. |
| **Cycle** | `String` | The operational mode of the AC (e.g., `COOLING`). |
| **Unit** | `String` | The specific hardware identifier or capacity configuration (e.g., `INV 2.5PK`). |
| **Condition** | `String` | **Target Variable:** The true operational state or specific fault class (e.g., `NORMAL`, `TROUBLE 1`, `MAINTENANCE 1`). |

## 3. Prediction Target
This model is designed for a **Multiclass Classification** task. The prediction target is the specific operational state or fault condition of the air conditioning unit. The tracked target classes are:
*   `NORMAL`
*   `ABNORMAL`      = OPENING, OVERCAPACITY
*   `MAINTENANCE 1` = INSUFFICIENT FREON, LEAKAGE, FREEZE
*   `MAINTENANCE 2` = FILTER DUST, FIN OUTDOOR DUST, BLOCKAGE , ELECTRICAL
*   `TROUBLE 1`     = INSUFFICIENT FREON, LEAKAGE, FREEZE
*   `TROUBLE 2`     = COMPRESSOR OVERLOAD, BLOCKAGE, ELECTRICAL
*   `TROUBLE 3`     = UNSTABLE VOLTAGE, GENERATOR POWER SUPPLY

## 4. Preprocessing
Preprocessing here happens in four distinct stages, spread across three files, and it's worth separating them because they're preprocessing very different things (synthetic generation, real data, and model input).
### Stage 1: Raw feature engineering (StaticDataGenerator.py)
This isn't preprocessing an existing dataset, it's generating one. But it does the raw-feature engineering that would normally happen in a cleaning step:
*  Sensor values are drawn per-unit from tiered rule ranges (Normal/Maintenance/Trouble), based on a stateful simulation of indoor temperature and compressor cycling.
*  Four features are derived deterministically from the primitives, not sampled:

row['Wattage (W)'] = row['Voltage (V)'] * row['Current (A)']

row['Alpha (°C)'] = row['Temperature Indoor (°C)'] - row['Set Point (°C)']

row['Beta (°C)']  = row['Temperature Outdoor (°C)'] - row['Set Point (°C)']

row['Delta (°C)'] = row['Return (°C)'] - row['Supply (°C)']

*  All numeric columns rounded to 2 decimals, Unit/Condition cast to ordered categoricals, then sorted by Unit → Condition → Time before writing out.

### Stage 2: Tiered scaling (TieredScaling.py)
This is the closest thing to a classic "preprocessing" step, and it's a custom piecewise normalization rather than anything like z-score/min-max from sklearn:
*  For Current, Wattage, Alpha, Beta, Delta: each unit's own Normal-range min/max maps to [0,1]; below/above that maps into [-1,0) or (1,2] using the Maintenance-range bounds as the new min/max for that segment; beyond the Maintenance bounds it keeps extrapolating past -1/2 unbounded (no clipping):
if n_min <= value <= n_max:      return (value - n_min) / n_range
elif m_min <= value < n_min:     return -1 + (value - m_min) / m_range_low
elif n_max < value <= m_max:     return 1 + (value - n_max) / m_range_high
*  Voltage gets a different, simpler treatment. Plain linear scaling against a single global 196–265V band, not the per-unit tiered logic the other five features get.
*  Output: the scaled columns (Current_scaled, Voltage_scaled, etc.) are inserted next to their raw counterparts, not replacing them, so the model ultimately sees both.
*  No standardization, no outlier clipping, no missing-value imputation anywhere in this step.

### Stage 3: SFD noise injection (SyntheticDataGenerator.py)
Covered in detail earlier, independent Gaussian jitter and multiplicative drift applied to 6 raw/6 scaled column pairs, functioning as a data-augmentation preprocessing pass between Stage 2 and model training.

### Stage 4: Model-input prep (RandomForestClassifier.py / RFClassifierValidation.py)
*  Feature selection is just df.select_dtypes(include=['number']). This is the only "feature selection" happening. No manual column drops, no correlation filtering.
*  Stratified 70/30 train_test_split on Condition, random_state=42. No scaling step here, since RF is scale-invariant, and the data's already been through the Stage 2 tiered transform anyway.
*  No standardization/normalization applied at this stage. Whatever scale Stage 2 produced is what the RF trains on directly.
*  Real-data alignment (RFClassifierValidation.py): numeric columns selected from the real CSV, then reconciled against the training feature list. Any column the model was trained on but missing from real data gets filled with a constant 0.0, extras get dropped, and columns get reordered to match training order exactly.

## 5. Model Insights & Evaluation
### Why using Random Forest?
The Random Forest algorithm is exceptionally well-suited for Air Conditioner Fault Detection and Diagnosis because:
*   **Resilience to Sensor Noise:** AC sensor data is inherently noisy. Random Forest's ensemble approach smooths out individual decision tree errors, making it highly robust.
*   **Handles Non-linear Relationships:** Thermodynamic processes are highly non-linear. Random Forest captures these complex interactions natively without requiring heavy mathematical transformations.
*   **Interpretability:** Unlike deep neural networks, Random Forest provides transparent feature importance. If the model predicts a specific failure, we can verify exactly which sensor readings triggered that diagnosis.

### How is the model performance?
When evaluated against the real-world ACFDD datasets, the model achieved the following diagnostic metrics:
*   **Precision:** `0.9713` (97.13%)
*   **F1-Score:** `0.9271` (92.71%)
*   **Recall:** `0.8939` (89.39%)

### What factors affect the model's result?
While the model catches true faults flawlessly, the confusion matrix reveals a sim-to-real gap regarding the `NORMAL` baseline. Several factors influence this:
* **High Sensitivity / Cautious Prediction:** The overall ~89% recall is primarily pulled down by the `NORMAL` class. Out of 1,680 truly normal instances, the model occasionally flagged them as faults (e.g., 105 as `TROUBLE 1`, 66 as `ABNORMAL`). The model prioritizes catching every possible issue, resulting in a conservative approach that generates some false alarms during normal operations.
* **Sensor Noise on Real Hardware:** Environmental variability and stochastic noise in real sensors can make perfectly normal operations occasionally mimic the synthetic signatures of a fault state. 
* **Feature Engineering:** Refining the threshold calculations is critical to helping the model better differentiate between a genuine anomaly and a temporary, harmless environmental fluctuation.

