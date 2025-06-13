from mesa import Agent
import numpy as np

class SugarAgent(Agent):
    def __init__(self, model, agent_type="neutral", alpha=0.0, is_cooperator=False):
        super().__init__(model)
        self.sugar_level = 0
        self.agent_type = agent_type  # Added risk types
        self.alpha = alpha  # Added risk parameter for utility calculations
        self.is_cooperator = is_cooperator  # Added cooperation behavior
        
        # Set default alpha values for each risk type
        if alpha == 0.0:
            if agent_type == "risk_averse":
                self.alpha = -1.0
            elif agent_type == "risk_seeking":
                self.alpha = 1.0
            else:
                self.alpha = 0.0

    def compute_utility(self, pos):
        # Risk setting of sugar concentration × 10% as base utility
        x, y = pos
        c = self.model.grid_sugar[x, y]
        base_utility = c * 0.1
        
        # Apply risk preference transformation based on agent type
        if self.alpha == 0:
            utility = base_utility  # Risk-neutral: linear utility
        elif self.alpha > 0:
            # Risk-seeking: increasing returns to sugar
            if base_utility > 0:
                utility = np.power(base_utility, 1 - self.alpha)
            else:
                utility = 0
        else:
            # Risk-averse: diminishing returns to sugar
            utility = 1 - np.exp(self.alpha * base_utility)
            
        return max(utility, 0)

    def choose_move(self):
        # Logit model for movement decisions
        neighbors = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=True
        )
        
        # Only consider unoccupied positions (except current position)
        available_positions = []
        for pos in neighbors:
            cell_contents = self.model.grid.get_cell_list_contents([pos])
            if len(cell_contents) == 0 or pos == self.pos:
                available_positions.append(pos)
        
        if not available_positions:
            return self.pos
        
        # Calculate utilities for all available positions
        utilities = np.array([self.compute_utility(pos) for pos in available_positions])
        
        lambda_param = getattr(self.model, 'lambda_param', 1.0)
        exp_utilities = np.exp(lambda_param * utilities)
        
        if np.any(np.isinf(exp_utilities)) or np.sum(exp_utilities) == 0:
            max_idx = np.argmax(utilities)
            return available_positions[max_idx]
        
        probabilities = exp_utilities / np.sum(exp_utilities)
        
        try:
            selected_idx = self.random.choices(range(len(available_positions)), 
                                             weights=probabilities)[0]
            return available_positions[selected_idx]
        except (ValueError, IndexError):
            return self.random.choice(available_positions)

    def cooperate(self):
        # Cooperation phase - agents grow sugar in empty neighboring cells
        if not self.is_cooperator:
            return
        
        neighbors = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        
        # Find empty neighboring cells
        empty_spots = []
        for pos in neighbors:
            cell_contents = self.model.grid.get_cell_list_contents([pos])
            if len(cell_contents) == 0:
                empty_spots.append(pos)
        
        # Grow sugar in one random empty spot
        if empty_spots:
            cooperation_spot = self.random.choice(empty_spots)
            x, y = cooperation_spot
            self.model.grid_sugar[x, y] = min(self.model.grid_sugar[x, y] + 1, 4)
            self.model.sugar_layer.modify_cell(cooperation_spot, lambda v: min(v + 1, 4))

    def step(self):
        # Enhanced agent movement with consumption and cooperation
        # Choose new position using logit model
        new_pos = self.choose_move()

        if new_pos != self.pos:
            self.model.grid.move_agent(self, new_pos)

        x, y = new_pos
        if self.model.grid_sugar[x, y] > 0:
            sugar_consumed = min(1, self.model.grid_sugar[x, y])
            self.sugar_level += sugar_consumed
            self.model.grid_sugar[x, y] = max(0, self.model.grid_sugar[x, y] - sugar_consumed)
            self.model.sugar_layer.modify_cell((x, y), lambda v: max(0, v - sugar_consumed))

        self.cooperate()

    @classmethod
    def create_agents(cls, model, num_agents):
        # Create heterogeneous agent population with different risk types and cooperation
        agents = []
        cooperation_rate = getattr(model, 'cooperation_rate', 0.3)
        
        # Distribute agents equally across risk types
        num_risk_averse = num_agents // 3
        num_neutral = num_agents // 3
        num_risk_seeking = num_agents - num_risk_averse - num_neutral
        num_cooperators = int(num_agents * cooperation_rate)
        
        # Create risk-averse agents
        for i in range(num_risk_averse):
            is_coop = i < (num_cooperators // 3)
            agent = cls(model, agent_type="risk_averse", alpha=-1.0, is_cooperator=is_coop)
            agents.append(agent)
        
        # Create neutral agents
        for i in range(num_neutral):
            is_coop = i < (num_cooperators // 3)
            agent = cls(model, agent_type="neutral", alpha=0.0, is_cooperator=is_coop)
            agents.append(agent)
        
        # Create risk-seeking agents
        for i in range(num_risk_seeking):
            is_coop = i < (num_cooperators - 2 * (num_cooperators // 3))
            agent = cls(model, agent_type="risk_seeking", alpha=1.0, is_cooperator=is_coop)
            agents.append(agent)
        
        # Add all agents to model
        for agent in agents:
            model.agents.add(agent)
        
        return agents