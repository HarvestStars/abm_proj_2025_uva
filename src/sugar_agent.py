from mesa import Agent
import numpy as np
import random

class SugarAgent(Agent):
    def __init__(self, model, agent_type="neutral", alpha=0.0, is_cooperator=False):
        super().__init__(model)
        self.sugar_level = 0  # Initial sugar level
        self.agent_type = agent_type  # "risk_averse", "neutral", "risk_seeking"
        self.alpha = alpha  # Risk parameter: α < 0 (averse), α = 0 (neutral), α > 0 (seeking)
        self.is_cooperator = is_cooperator  # Whether agent cooperates by growing sugar
        
        # Set alpha based on agent type if not explicitly provided
        if alpha == 0.0:
            if agent_type == "risk_averse":
                self.alpha = -1.0
            elif agent_type == "risk_seeking":
                self.alpha = 1.0
            else:  # neutral
                self.alpha = 0.0

    def compute_utility(self, pos):
        """
        Improvement 1: Risk setting
        Calculate utility U = c × 10% where c is sugar concentration
        Apply risk preference transformation based on agent type
        """
        x, y = pos
        c = self.model.grid_sugar[x, y]  # Sugar concentration at position
        
        # Base utility: sugar concentration × 10%
        base_utility = c * 0.1
        
        # Apply risk preference transformation
        if self.alpha == 0:
            # Risk-neutral: U = base_utility
            utility = base_utility
        elif self.alpha > 0:
            # Risk-seeking: U = base_utility^(1-α) - more sensitive to high values
            if base_utility > 0:
                utility = np.power(base_utility, 1 - self.alpha)
            else:
                utility = 0
        else:  # self.alpha < 0
            # Risk-averse: U = 1 - exp(-α × base_utility) - diminishing returns
            utility = 1 - np.exp(self.alpha * base_utility)
            
        return max(utility, 0)  # Ensure non-negative utility

    def choose_move(self):
        """
        Improvement 2: Logit model
        Apply logit model using utility U to obtain probability
        P_ij = exp(λ × V_ij) / Σ exp(λ × V_ij)
        """
        # Get all neighboring positions including current position
        neighbors = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=True
        )
        
        # Remove occupied positions (except current position)
        available_positions = []
        for pos in neighbors:
            cell_contents = self.model.grid.get_cell_list_contents([pos])
            if len(cell_contents) == 0 or pos == self.pos:
                available_positions.append(pos)
        
        if not available_positions:
            return self.pos  # Stay in place if no available moves
        
        # Calculate utilities for all available positions
        utilities = np.array([self.compute_utility(pos) for pos in available_positions])
        
        # Apply logit model with lambda (noise parameter)
        lambda_param = getattr(self.model, 'lambda_param', 1.0)
        exp_utilities = np.exp(lambda_param * utilities)
        
        # Handle numerical overflow
        if np.any(np.isinf(exp_utilities)) or np.sum(exp_utilities) == 0:
            # If overflow or all zeros, choose position with highest utility
            max_idx = np.argmax(utilities)
            return available_positions[max_idx]
        
        # Calculate probabilities
        probabilities = exp_utilities / np.sum(exp_utilities)
        
        # Choose position based on probabilities
        try:
            selected_idx = self.random.choices(range(len(available_positions)), 
                                             weights=probabilities)[0]
            return available_positions[selected_idx]
        except (ValueError, IndexError):
            # Fallback to random choice if weights are invalid
            return self.random.choice(available_positions)

    def cooperate(self):
        """
        Improvement 4: Cooperation phase
        If agent is cooperator, grow sugar in empty neighboring spots
        """
        if not self.is_cooperator:
            return
        
        # Get neighboring positions
        neighbors = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        
        # Find empty spots
        empty_spots = []
        for pos in neighbors:
            cell_contents = self.model.grid.get_cell_list_contents([pos])
            if len(cell_contents) == 0:  # Empty cell
                empty_spots.append(pos)
        
        # Cooperate by growing sugar in one random empty spot
        if empty_spots:
            cooperation_spot = self.random.choice(empty_spots)
            x, y = cooperation_spot
            
            # Grow sugar (add 1 unit)
            self.model.grid_sugar[x, y] = min(self.model.grid_sugar[x, y] + 1, 4)
            self.model.sugar_layer.modify_cell(cooperation_spot, lambda v: min(v + 1, 4))

    def step(self):
        """
        Improvement 3: Move agents
        Complete agent step with movement, consumption, and cooperation
        """
        # 1. Choose new position using logit model
        new_pos = self.choose_move()
        
        # 2. Move to new position
        if new_pos != self.pos:
            self.model.grid.move_agent(self, new_pos)
        
        # 3. Consume sugar at new position
        x, y = new_pos
        if self.model.grid_sugar[x, y] > 0:
            sugar_consumed = min(1, self.model.grid_sugar[x, y])
            self.sugar_level += sugar_consumed
            self.model.grid_sugar[x, y] = max(0, self.model.grid_sugar[x, y] - sugar_consumed)
            self.model.sugar_layer.modify_cell((x, y), lambda v: max(0, v - sugar_consumed))
        
        # 4. Cooperation phase
        self.cooperate()

    @classmethod
    def create_agents(cls, model, num_agents):
        """
        Create agents with different types and cooperation status
        FIXED: Handle case where model doesn't have cooperation_rate attribute
        """
        agents = []
        
        # Get cooperation rate from model, with fallback
        cooperation_rate = getattr(model, 'cooperation_rate', 0.3)
        
        # Calculate number of each type (roughly equal distribution)
        num_risk_averse = num_agents // 3
        num_neutral = num_agents // 3
        num_risk_seeking = num_agents - num_risk_averse - num_neutral
        
        # Calculate number of cooperators
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
        
        # Add agents to model
        for agent in agents:
            model.agents.add(agent)
        
        return agents