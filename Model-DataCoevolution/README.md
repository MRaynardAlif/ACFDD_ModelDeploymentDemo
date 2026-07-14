# The Model-Data Coevolution Framework
The core methodology relies on a closed-loop feedback system, termed Model-Data Coevolution, which iteratively optimizes the synthetic data generation process. The system consists of four main functional modules, the Controller, the Evolver, the Trainer, and the Validator.

<img width="7903" height="4613" alt="image" src="https://github.com/user-attachments/assets/f95d2a52-19ad-4a64-9de7-d4246d734713" />

## 1. The Controller: Simulated Annealing Optimization
The Controller serves as the optimization engine that optimizes the augmentation hyperparameters. The main task is to find the optimal set of augmentation parameters and fed it to the evolver to evolve the rule-based baseline dataset. The SA algorithm operates by proposing a new set of augmentation parameters at each iteration. If the new parameters result in a better fitness score, they are accepted. If they result in a worse score, they are accepted with a probability *P* governed by the Boltzmann distribution:

<img width="549" height="87" alt="image" src="https://github.com/user-attachments/assets/7fced2f4-caae-4209-a6b1-c8257c7418ab" />

Where *T* is the system temperature. The system was initialized with a start temperature of 0.2 and a cooling rate of 0.98, decaying *T* over time to transition from exploration (high randomness) to exploitation (local refinement). The Controller manage a parameter set *θ={α,β,γ,δ}*, corresponding to Rule Exponent (*α* (global scaling factor)), Fault Intensity (*β* (stochastic noise injection)), Relative Frequency (*γ* (sampling probability, adjust the class prior probabilities *P(Y)*)), and Thermal Drift (*δ* (additive bias)).

## 2.	The Evolver: Augmentation Logic
The Evolver module applies the augmentation parameters proposed by the Controller to the rule-based baseline synthetic data. While *γ* governs the sampling distribution *P(Y)*, the feature transformation logic relies on *α*, *β*, and *δ*. 

## 3. The Trainer
The Trainer module trains a machine learning model on the evolved dataset. The classifier's hyperparameters were kept constant throughout all experiments.

## 4. The Validator: Hybrid Fitness Function
The Validator module evaluates the trained model on the 30% of real-world validation set. To explicitly address the problem of "alarm fatigue," the fitness function was engineered to weigh Recall (sensitivity) higher than precision or accuracy. This choice was informed by the baseline model's tendency to miss normal states (low Normal Recall), causing excessive false alarms. The fitness score is calculated as a hybrid metric:

<img width="1074" height="144" alt="image" src="https://github.com/user-attachments/assets/24b28ddd-a100-4cb8-aacd-7cea9a41e49d" />

Where *F1*, *P*, and *R* are the weighted average F1-score, Precision, and Recall from the validation report. *Div* is the feature divergence score, a simple metric (mean normalized difference between the real and synthetic feature means) used to penalize evolved datasets that drift too far from the real-world data's distribution. To prevent the evolutionary controller from overfitting to the specific distribution of the real-world data ("meta-overfitting"), a strict split were employed. The validation set (30% of real-world validation dataset) were used inside the closed-loop optimization to calculate the hybrid score and guide the controller. The holdout test set (70% of real-world validation dataset) were completely hidden from the optimization process. This 70% set is used strictly for the final generalization verification of the evolved policies.


