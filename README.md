# ACFDD_ModelDeploymentDemo
Source code and experiments applying combined Random Forest and Autoencoder algorithms to the Air Conditioner FDD problem space.

## 1. Training Dataset Description
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
*  For Current, Wattage, Alpha, Beta, Delta: each unit's own Normal-range min/max maps to [0,1]; below/above that maps into [-1,0) or (1,2] using the Maintenance-range bounds as the new min/max for that segment; beyond the Maintenance bounds it keeps extrapolating past -1/2 unbounded (no clipping for Trouble Condition). This guarantees that when the Autoencoder sees a value like 1.5, it instantly knows the parameter is in a degraded state, regardless of whether that number represents 6 Amps on a 0.5PK unit or 12 Amps on a 2.5PK unit:

if n_min <= value <= n_max:      return (value - n_min) / n_range

elif m_min <= value < n_min:     return -1 + (value - m_min) / m_range_low

elif n_max < value <= m_max:     return 1 + (value - n_max) / m_range_high

*  Voltage gets a different, simpler treatment. Plain linear scaling against a single global 196–265V band, not the per-unit tiered logic the other five features get.
*  Output: the scaled columns (Current_scaled, Voltage_scaled, etc.) are inserted next to their raw counterparts, not replacing them, so the model ultimately sees both.
*  No standardization, no outlier clipping, no missing-value imputation anywhere in this step.

### Stage 3: SFD noise injection (SyntheticDataGenerator.py)
Covered in detail earlier, independent Gaussian jitter and multiplicative drift applied to 6 raw/6 scaled column pairs, functioning as a data-augmentation preprocessing pass between Stage 2 and model training.

### Stage 4: Model-input prep (RandomForestClassifier.py / RFClassifierValidation.py)
There are two script for two distinct sides of the AC-FDD machine learning architecture. It separated the training logic for unsupervised diagnostic model (the Autoencoder) and supervised detection model (the Random Forest). Here is a breakdown of both training pipelines:

#### RandomForestClassifier.py (Random Forest):
This script builds the frontline guard that classifies the severity of the system's state. Wrapped in an object-oriented RFModel class, which makes it highly reusable.
*  Feature selection is just df.select_dtypes(include=['number']). This is the only "feature selection" happening. No manual column drops, no correlation filtering.
*  Stratified 70/30 train_test_split on Condition, random_state=42. No scaling step here, since RF is scale-invariant, and the data's already been through the Stage 2 tiered transform anyway.
*  No standardization/normalization applied at this stage. Whatever scale Stage 2 produced is what the RF trains on directly.
*  Real-data alignment (RFClassifierValidation.py): numeric columns selected from the real CSV, then reconciled against the training feature list. Any column the model was trained on but missing from real data gets filled with a constant 0.0, extras get dropped, and columns get reordered to match training order exactly.

#### train_and_save.py (Autoencoder):
This script is responsible for building the baseline understanding of a healthy air conditioner.
*  The crucial filter, df_synth[(df_synth['Condition'] == 'NORMAL')]. This is the most important line in the script. Intentionally blinding the model to any faults, forcing it to only learn the mathematical signature of perfect thermodynamic operation.
*  The Bottleneck Architecture, fully connected neural network using the Keras Functional API. With 6 scaled features input, Compression (Encoder) that squeezes down to a 3-neuron bottleneck to force the network to learn the physical relationships rather than just memorizing the data, Reconstruction (Decoder), and finally, outputs the reconstructed 6 features using a linear activation (to predict continuous numerical values).

## 5. Model Insights & Evaluation
### The Random Forest:
This model does not understand thermodynamics. It understands statistical thresholds and non-linear boundaries between discrete severity states (NORMAL, MAINTENANCE, TROUBLE). Here is why the Random Forest algorithm is exceptionally well-suited for Air Conditioner Fault Detection because:
*   **Resilience to Sensor Noise:** AC sensor data is inherently noisy. Random Forest's ensemble approach smooths out individual decision tree errors, making it highly robust.
*   **Handles Non-linear Relationships:** Thermodynamic processes are highly non-linear. Random Forest captures these complex interactions natively without requiring heavy mathematical transformations.

#### How is the model performance?
When evaluated against the real-world ACFDD datasets, the model achieved the following diagnostic metrics:

**On 30% Split Real Data (in the Coevolution Loop):**

<img width="497" height="264" alt="image" src="https://github.com/user-attachments/assets/ee8d9f87-d776-428f-a71f-3b9bd45fece4" />


*   **Accuracy:** `0.9302` (93.02%)
*   **Precision:** `0.9795` (97.95%)
*   **F1-Score:** `0.9530` (95.30%)
*   **Recall:** `0.9302` (93.02%)

This "best" number is worth pausing on, support=0 for ABNORMAL, TROUBLE 1, TROUBLE 2, and TROUBLE 3 means real validation set (Dataset_Test_30.csv) has zero true examples of those conditions. It only actually contains NORMAL, MAINTENANCE 1, and MAINTENANCE 2. Since support is 0, those rows only appear because the model did predict ABNORMAL/TROUBLE 1/TROUBLE 2/TROUBLE 3 for some real rows, just with zero true instances to match against and the model is producing false positives for fault types that real validation data can't confirm or deny it on.

<img width="2000" height="1600" alt="confusion_matrix_best_model" src="https://github.com/user-attachments/assets/6ebf5ece-cf43-4af0-83af-420b7cadf785" />

4 out of 7 fault classes (ABNORMAL, TROUBLE 1, TROUBLE 2, TROUBLE 3) have zero real-world validation. Where the real errors actually are:
*   **MAINTENANCE 2** (348 real examples): perfect, 348/348 on the diagonal. Genuinely solid.
*   **MAINTENANCE 1** (724 examples): 702 correct, 22 leak into TROUBLE 1 (~3%). A nearby-severity confusion, not alarming.
*   **NORMAL** (1680 examples): only 1510 correct. The other 170 healthy units (10.1%) get flagged as something else. Specifically 77 as ABNORMAL, 32 as MAINTENANCE 1, 28 as TROUBLE 1, 27 as MAINTENANCE 2, and 6 as TROUBLE 3.

**On 70% Split Real Data (held-out):**

<img width="501" height="289" alt="image" src="https://github.com/user-attachments/assets/87ed10c5-9689-4b3a-90a5-a9674e2147f8" />

The NORMAL false-alarm rate is real and stable, not sampling noise. 89.9% → 89.4% recall is about as close as two independent samples get. And the shape of the errors replicates too. Proportionally, healthy units still get misclassified as ABNORMAL most often (~42-45% of the errors both times), then TROUBLE 1, then MAINTENANCE 1/2, then rarely TROUBLE 3.That consistency across two disjoint samples means this isn't noiseit's a genuine decision-boundary issue where the model's learned NORMAL region has real, reproducible overlap with several fault regions in feature space, most of all ABNORMAL.

<img width="2400" height="1800" alt="confusion_matrix" src="https://github.com/user-attachments/assets/12f32605-7c94-4b24-b448-20eea7b27616" />

The genuinely good news, made more credible by this is MAINTENANCE 2 at 100% recall on both an in-search split and a fully held-out split of more than double the size is a real, well-supported result, not luck. Same for the overall ~95-96% weighted F1 holding steady. This is exactly the kind of double-validation that makes a sim-to-real transfer claim defensible: This isn't reporting a number from the split that happened to influence the search, but reporting one that replicates on data the model never touched.

### The Autoencoder:
This model acts as a mathematical proxy for the laws of physics. By forcing 6 features through a 3-neuron bottleneck, it learns the baseline relationships. The insight comes from **Z-Score Residuals**. When a physical relationship breaks (e.g., the compressor draws high power, but Delta drops), the Autoencoder fails to reconstruct it. The specific feature it fails to reconstruct provides the root-cause diagnosis.

#### How is the model performance?
This is evaluated by observing the Mean Squared Error (MSE) distribution. Demonstrated with the reconstruction error on healthy data is tightly grouped (e.g., strictly under the 99th percentile threshold of 0.07172). Then, tested when anomalous data is introduced, the MSE drastically spikes above this threshold, proving the diagnostic trigger is mathematically sound.

### What factors affect the model's result?

#### For The Random Forest:
While the model catches true faults flawlessly, the confusion matrix reveals a sim-to-real gap regarding the `NORMAL` baseline. Several factors influence this:
* **High Sensitivity / Cautious Prediction:** The overall ~93% recall is primarily pulled down by the `NORMAL` class. Out of 1,680 truly normal instances, the model occasionally flagged them as faults (e.g., 28 as `TROUBLE 1`, 77 as `ABNORMAL`). The model prioritizes catching every possible issue, resulting in a conservative approach that generates some false alarms during normal operations.
* **Sensor Noise on Real Hardware:** Environmental variability and stochastic noise in real sensors can make perfectly normal operations occasionally mimic the synthetic signatures of a fault state. 
* **Feature Engineering:** Refining the threshold calculations is critical to helping the model better differentiate between a genuine anomaly and a temporary, harmless environmental fluctuation.
* **The Rule-Based Thresholds (from AdjustedRule.xlsx):** Since the whole synthetic dataset is generated from these, they set the ceiling on what's achievable before the coevolution loop even starts.

#### For The Autoencoder:
The accuracy of the diagnosis depends entirely on the mathematical relationships of the engineered features. For instance, the diagnostic parameter *β* strictly represents the environmental thermal gradient (*β* = Temperature Outdoor - Temperature Indoor). If this calculation were mistakenly altered (e.g., subtracting the set point instead), the Autoencoder's understanding of the thermal load would collapse, ruining the reconstruction accuracy. Since the Autoencoder maps a "normal" baseline, its results can be affected by extreme, unprecedented environmental conditions. If a heatwave pushes the ambient temperature far beyond the synthetic training distribution, the AC unit might operate in a highly stressed (but mechanically healthy) state that the Autoencoder has never seen, leading to an artificially high reconstruction error.

## 6. What's real here:
Those two model never saw a single row of real data during training. Every parameter it learned came from a rule-based synthetic generator. That's a legitimate sim-to-real transfer result. Zero-shot generalization from purely synthetic sensor data to a physical AC unit, with no domain adaptation beyond the tiered scaling, landing in the 90-100% range on 3 of 3 testable classes. This is the kind of result that justifies the whole "model-data coevolution" approach as a concept. It's not a trivial outcome, a naively-generated synthetic dataset frequently transfers far worse than this. Still, the strong result on MAINTENANCE 1/2/NORMAL doesn't tell anything about ABNORMAL/TROUBLE 1/2/3. Those are just untested, not implicitly validated by the other classes doing well. Sim-to-real transfer quality isn't guaranteed to be uniform across classes, especially fault classes that rarer/more extreme in the parameter space. The point is, it's a solid result on what's been tested, and it's an incomplete validation of the full 7-class system.

## 7. App Deployment Link:
### 1.  Hugging Face: 
https://huggingface.co/spaces/Raynrd/AC-Fault-Detection

### 2. Gradio Modal:
https://mraynardalif--ac-fault-detection-ui.modal.run/


## 8. ACFDD Repository Presentation:
https://youtu.be/tEJdUDRJ7_s
