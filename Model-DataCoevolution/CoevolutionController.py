# -*- coding: utf-8 -*-

"""
CoevolutionController.py
A "smarter" brain that uses Simulated Annealing to avoid local optima.

It's a Hill-Climber that can "jump" out of valleys.
"""

import numpy as np
import json
import os
import copy

class CoevolutionController:
    """
    Implements a Simulated Annealing algorithm.
    This controller can accept "worse" solutions with a probability
    that decreases over time, allowing it to escape local optima.
    """

    # Floor on temperature so score_difference / temperature can never hit
    # 0/0 (nan) or blow up once cooling_rate has driven temperature very
    # close to zero over a long run.
    MIN_TEMPERATURE = 1e-6

    def __init__(self, initial_step_size=0.1, 
                 start_temperature=1.0, cooling_rate=0.99,
                 random_state=42):
        """
        Initializes the controller.

        Args:
            initial_step_size (float): How "big" of a random nudge to apply.
            start_temperature (float): The initial probability of accepting
                                     a "bad" move. Higher = more exploration.
            cooling_rate (float): How fast the temperature drops (e.g., 0.99
                                = 1% cooler each iteration).
            random_state (int): Seed for reproducibility.
        """
        self.step_size = initial_step_size
        self.temperature = start_temperature
        self.cooling_rate = cooling_rate
        self.random_state = random_state
        np.random.seed(random_state)
        
        # We now track two things:
        # 1. The parameters for the *current* step
        self.current_params = {
            "rule_exp": 1.0, "fault_int": 1.0, "rel_freq": 1.0, "thermal": 0.0,
        }
        # 2. The *best parameters ever found*
        self.best_params = copy.deepcopy(self.current_params)
        
        # We also track the scores associated with them
        self.current_score = -np.inf
        self.best_score = -np.inf
        
        self.proposed_params = copy.deepcopy(self.current_params)
        
        print(f"[Controller] Initialized. T_start={start_temperature}, Cooling={cooling_rate}")

    def propose_new_params(self):
        """
        Creates a "proposal" by nudging the *current* params.
        """
        # Start from the current solution, not the "best" one
        self.proposed_params = copy.deepcopy(self.current_params)
        
        # Nudge each parameter by a small, random amount
        self.proposed_params["rule_exp"] *= (1 + np.random.uniform(-self.step_size, self.step_size))
        self.proposed_params["fault_int"] *= (1 + np.random.uniform(-self.step_size, self.step_size))
        self.proposed_params["rel_freq"] *= (1 + np.random.uniform(-self.step_size, self.step_size))
        self.proposed_params["thermal"] += np.random.uniform(-self.step_size * 0.1, self.step_size * 0.1)

        # Clamp parameters to safe ranges
        self.proposed_params["rule_exp"] = float(np.clip(self.proposed_params["rule_exp"], 0.5, 2.0))
        self.proposed_params["fault_int"] = float(np.clip(self.proposed_params["fault_int"], 0.5, 2.0))
        # rel_freq is the FRACTION OF ROWS a proposal mutates (see
        # SyntheticDataGenerator.evolve_dataset), so it needs [0, 1] bounds,
        # not the multiplicative-scale bounds used for rule_exp/fault_int.
        # Floored at 0.05 rather than 0 so a proposal always touches at
        # least some rows.
        self.proposed_params["rel_freq"] = float(np.clip(self.proposed_params["rel_freq"], 0.05, 1.0))
        self.proposed_params["thermal"] = float(np.clip(self.proposed_params["thermal"], -0.5, 0.5))

        print(f"[Controller] Proposing new params...")
        return self.proposed_params

    def accept_or_reject_proposal(self, new_score):
        """
        The core Simulated Annealing "brain".
        Decides whether to accept the new `proposed_params` (and score).
        """
        
        if self.current_score == -np.inf: # Handle the very first iteration
            self.current_score = new_score
            self.best_score = new_score
            self.current_params = copy.deepcopy(self.proposed_params)
            self.best_params = copy.deepcopy(self.proposed_params)
            print("[Controller] First iteration. Set as baseline.")
            return

        # 1. Is the new score better than the current one?
        if new_score > self.current_score:
            print(f"[Controller] ✅ Proposal ACCEPTED (Better score).")
            self.current_params = copy.deepcopy(self.proposed_params)
            self.current_score = new_score
            
            # Is it also the best *ever*?
            if new_score > self.best_score:
                self.best_score = new_score
                self.best_params = copy.deepcopy(self.proposed_params)
                print(f"[Controller] 🔥 New all-time best score.")
        
        # 2. The new score is *worse*. Should we accept it anyway?
        else:
            # Calculate how much worse it is
            score_difference = new_score - self.current_score  # This is a negative number
            
            # Calculate the probability of accepting this "bad" move
            # This is the core of the algorithm.
            # max(..., MIN_TEMPERATURE) guards against temperature having
            # decayed to (near) zero over a long run, which would otherwise
            # produce inf/nan when score_difference is also 0.
            safe_temperature = max(self.temperature, self.MIN_TEMPERATURE)
            acceptance_probability = np.exp(score_difference / safe_temperature)
            
            if np.random.rand() < acceptance_probability:
                print(f"[Controller] ⚠️ Proposal ACCEPTED (Bad move, but exploring). Prob={acceptance_probability:.3f}")
                # We accept the "bad" move to explore
                self.current_params = copy.deepcopy(self.proposed_params)
                self.current_score = new_score
            else:
                print(f"[Controller] ❌ Proposal REJECTED (Bad move). Prob={acceptance_probability:.3f}")
                # We reject the proposal. The self.current_params remains
                # unchanged, so we'll try a new proposal from that
                # *previous* (better) location.

        # 3. "Cool down" the temperature for the next iteration
        self.temperature = max(self.temperature * self.cooling_rate, self.MIN_TEMPERATURE)
        print(f"[Controller] New Temperature: {self.temperature:.6f}")

    def get_best_params(self):
        """Returns the best parameters *ever* found during the run."""
        return self.best_params