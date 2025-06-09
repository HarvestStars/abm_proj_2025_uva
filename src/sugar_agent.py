from mesa import Agent, Model
import numpy as np
import random

class BaseWalker(Agent):
    def __init__(self, model : Model, pos):
        super().__init__(model)

        self.pos = pos

    def move(self):
        ''' 
        This method should get the neighbouring cells (Moore's neighbourhood), select one, and move the agent to this cell.
        '''
        print(f"Agent {self.unique_id} at {self.pos} is moving.")
        neighbours_cells = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False, radius=1)

        # If there are no neighbours, we cannot move
        if not neighbours_cells:
            print(f"Agent {self.unique_id} at {self.pos} has no neighbours to move to.")
            return
        
        # Randomly select a neighbour to move to
        new_pos = random.choice(neighbours_cells)
        
        # Move the agent to the new position
        self.model.grid.move_agent(self, new_pos)
        self.pos = new_pos
        print(f"Agent {self.unique_id} moved to {self.pos}")
        

# --- Agent Class ---
class SugarAgent(Agent):
    def __init__(self, model):
        super().__init__(model)

    def compute_utility(self, pos):
        x, y = pos
        return self.model.grid_sugar[x, y]  # Use numpy sugar grid directly

    def choose_move(self):
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
        self.pos = (x, y)
        self.model.grid_sugar[x, y] = max(0, self.model.grid_sugar[x, y] - 1)
        self.model.sugar_layer.modify_cell((x, y), lambda v: max(0, v - 1))
