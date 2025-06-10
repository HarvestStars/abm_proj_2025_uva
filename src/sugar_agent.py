from mesa import Agent, Model
import numpy as np
import random

# --- Agent Class ---
class SugarAgent(Agent):
    def __init__(self, model):
        super().__init__(model)
        self.sugar_level = 0  # Initial sugar level

    def compute_utility(self, pos):
        x, y = pos
        return self.model.grid_sugar[x, y]  # Use numpy sugar grid directly

    def choose_move(self):
        # alpha = 1 ; U = alpha * 0.3
        # probs = exp(lambda * U) / sum(exp(lambda * U_i))

        # to be replaced with a more complex utility function
        neighbors = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        utilities = np.array([self.compute_utility(pos) for pos in neighbors], dtype=float)
        exp_utilities = np.exp(utilities)
        probs = exp_utilities / exp_utilities.sum()
        selected_pos = self.random.choices(neighbors, weights=probs)[0]
        return selected_pos

    def step(self):
        new_pos = self.choose_move()
        self.model.grid.move_agent(self, new_pos)
        x, y = new_pos
        if self.model.grid_sugar[x, y] > 0:
            self.sugar_level += 1 # maybe grid sugar * 10%
        self.model.grid_sugar[x, y] = max(0, self.model.grid_sugar[x, y] - 1)
        self.model.sugar_layer.modify_cell((x, y), lambda v: max(0, v - 1))
