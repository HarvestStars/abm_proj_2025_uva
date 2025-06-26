from mesa import Agent
import numpy as np

class SugarAgent_Neutral(Agent):
    """Risk-neutral agents with linear utility function"""

    def __init__(self, model, alpha=0.0, consume_per_step=1, consume_proportion=0.1, consume_prop_mode=False, is_cooperator=False):
        super().__init__(model)
        self.sugar_level = 0
        self.agent_type = "neutral"
        self.alpha = alpha
        self.is_cooperator = is_cooperator
        self.consume_per_step = consume_per_step
        self.consume_proportion = consume_proportion
        self.consume_prop_mode = consume_prop_mode
        assert isinstance(consume_per_step, (int)), "consume_per_step must be a number (int or float)"
        print(f"Creating neutral agent with alpha={self.alpha}, consume_per_step={self.consume_per_step}, consume_proportion={self.consume_proportion}, consume_prop_mode={self.consume_prop_mode}, is_cooperator={self.is_cooperator}")

    def compute_utility(self, pos):
        """Risk-neutral utility: U = sugar concentration × 10%"""
        x, y = pos
        c = self.model.grid_sugar[x, y]
        base_utility = c
        return base_utility  # Linear utility

    # whether move or not if the self cell is richer?
    def choose_move(self):
        """Logit model for movement decisions"""
        neighbors = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=True
        )
        
        available_positions = []
        for pos in neighbors:
            cell_contents = self.model.grid.get_cell_list_contents([pos])
            if len(cell_contents) == 0 or pos == self.pos:
                available_positions.append(pos)
        
        if not available_positions:
            return self.pos
        
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
        """Cooperation phase"""
        if not self.is_cooperator:
            return
        
        neighbors = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        
        empty_spots = []
        for pos in neighbors:
            cell_contents = self.model.grid.get_cell_list_contents([pos])
            if len(cell_contents) == 0:
                empty_spots.append(pos)
        
        if empty_spots:
            cooperation_spot = self.random.choice(empty_spots)
            x, y = cooperation_spot
            self.model.grid_sugar[x, y] = min(self.model.grid_sugar[x, y] + 1, 4)
            self.model.sugar_layer.modify_cell(cooperation_spot, lambda v: min(v + 1, 4))

    def step(self):
        """Agent step: move, consume, cooperate"""
        new_pos = self.choose_move()

        if new_pos != self.pos:
            self.model.grid.move_agent(self, new_pos)

        x, y = new_pos
        if self.model.grid_sugar[x, y] > 0:
            if self.consume_prop_mode:
                # Consume a proportion of the sugar in the cell
                sugar_consumed = self.model.grid_sugar[x, y] * self.consume_proportion
            else:
                # Consume a fixed amount of sugar
                sugar_consumed = min(self.consume_per_step, self.model.grid_sugar[x, y])
            self.sugar_level += sugar_consumed
            self.model.grid_sugar[x, y] = max(0, self.model.grid_sugar[x, y] - sugar_consumed)
            self.model.sugar_layer.modify_cell((x, y), lambda v: max(0, v - sugar_consumed))

        # self.cooperate()

    @classmethod
    def create_agents(cls, model, num_agents, consume_per_step=1, consume_proportion=0.1, consume_prop_mode=True, fixed_alpha=None):
        """
        Create neutral agents
        
        Parameters:
        -----------
        model : Model
            Mesa model instance
        num_agents : int
            Number of agents to create
        consume_per_step : int
            Amount of sugar consumed per step
        fixed_alpha : float or None
            If provided, all neutral agents will use this fixed alpha value
            If None, default value 0.0 will be used
        """
        agents = []
        cooperation_rate = getattr(model, 'cooperation_rate', 0.3)
        print(f"Creating {num_agents} neutral agents with cooperation rate {cooperation_rate}")
        num_cooperators = int(num_agents * cooperation_rate)
        
        # Neutral agents typically have alpha fixed at 0
        alpha_val = fixed_alpha if fixed_alpha is not None else 0.0
        
        for i in range(num_agents):
            is_coop = i < num_cooperators
            agent = cls(model, alpha=alpha_val, consume_per_step=consume_per_step, consume_proportion=consume_proportion, consume_prop_mode=consume_prop_mode, is_cooperator=is_coop)
            agents.append(agent)
            model.agents.add(agent)
        
        return agents


class SugarAgent_Riskseeking(Agent):
    """Risk-seeking agents with increasing returns utility function"""

    def __init__(self, model, alpha=-1.0, consume_per_step=1, consume_proportion=0.1, consume_prop_mode=False, is_cooperator=False):
        super().__init__(model)
        self.sugar_level = 0
        self.agent_type = "risk_seeking"
        self.alpha = alpha
        self.is_cooperator = is_cooperator
        self.consume_per_step = consume_per_step
        self.consume_proportion = consume_proportion
        self.consume_prop_mode = consume_prop_mode

    def compute_utility(self, pos):
        """
            Risk-seeking utility:         
            U(c) = (1 - exp(-a * c)) / a   if a != 0
            = c                      if a == 0

            a < 0 means risk-seeking behavior
        """
        x, y = pos
        c = self.model.grid_sugar[x, y]
        if self.alpha != 0:
            utility = (1 - np.exp(-self.alpha * c)) / self.alpha
        else:
            utility = c

        return utility

    def choose_move(self):
        """Logit model for movement decisions"""
        neighbors = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=True
        )
        
        available_positions = []
        for pos in neighbors:
            cell_contents = self.model.grid.get_cell_list_contents([pos])
            if len(cell_contents) < 4 or pos == self.pos:  # multigrid means they can overlap on same cell
                available_positions.append(pos)
        
        if not available_positions:
            return self.pos
        
        utilities = np.array([self.compute_utility(pos) for pos in available_positions])
        lambda_param = getattr(self.model, 'lambda_param', 1.0)

        # exp_utilities = np.exp(lambda_param * utilities)
        scaled_utilities = lambda_param * utilities
        scaled_utilities -= np.max(scaled_utilities)  # prevent overflow, tricky!
        exp_utilities = np.exp(scaled_utilities)
        
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
        """Cooperation phase"""
        if not self.is_cooperator:
            return
        
        neighbors = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        
        empty_spots = []
        for pos in neighbors:
            cell_contents = self.model.grid.get_cell_list_contents([pos])
            if len(cell_contents) == 0:
                empty_spots.append(pos)
        
        if empty_spots:
            cooperation_spot = self.random.choice(empty_spots)
            x, y = cooperation_spot
            self.model.grid_sugar[x, y] = min(self.model.grid_sugar[x, y] + 1, 4)
            self.model.sugar_layer.modify_cell(cooperation_spot, lambda v: min(v + 1, 4))

    def step(self):
        """Agent step: move, consume, cooperate"""
        new_pos = self.choose_move()

        if new_pos != self.pos:
            self.model.grid.move_agent(self, new_pos)

        x, y = new_pos
        if self.model.grid_sugar[x, y] > 0:
            if self.consume_prop_mode:
                # Consume a proportion of the sugar in the cell
                sugar_consumed = self.model.grid_sugar[x, y] * self.consume_proportion
            else:
                # Consume a fixed amount of sugar
                sugar_consumed = min(self.consume_per_step, self.model.grid_sugar[x, y])
            self.sugar_level += sugar_consumed
            self.model.grid_sugar[x, y] = max(0, self.model.grid_sugar[x, y] - sugar_consumed)
            self.model.sugar_layer.modify_cell((x, y), lambda v: max(0, v - sugar_consumed))

        # self.cooperate()

    @classmethod
    def create_agents(cls, model, num_agents, alpha_values=None, consume_per_step=1, consume_proportion=0.1, consume_prop_mode=False, random_alpha=False):
        """
        Create risk-seeking agents
        
        Parameters:
        -----------
        model : Model
            Mesa model instance
        num_agents : int
            Number of agents to create
        alpha_values : float, list of floats, or None
            - If float: all agents use this fixed value
            - If list: values are assigned to agents in a cyclic manner
            - If None: use default value 1.0 or random generation (if random_alpha=True)
        consume_per_step : int
            Amount of sugar consumed per step
        random_alpha : bool
            If True and alpha_values is None, generate random values in [0.1, 10.0]
        """
        agents = []
        cooperation_rate = getattr(model, 'cooperation_rate', 0.3)
        num_cooperators = int(num_agents * cooperation_rate)
        
        for i in range(num_agents):
            is_coop = i < num_cooperators
            
            # Determine alpha value
            if alpha_values is not None:
                if isinstance(alpha_values, (int, float)):
                    # Fixed value
                    alpha_val = float(alpha_values)
                elif isinstance(alpha_values, list):
                    # Cycle through list values
                    alpha_val = alpha_values[i % len(alpha_values)]
                else:
                    raise ValueError("alpha_values must be float or list")
            elif random_alpha:
                # Random generation
                alpha_val = model.random.uniform(0.1, 10.0)
            else:
                # Default value
                alpha_val = 1.0
            
            # Ensure alpha is within valid range
            assert -10.0 <= alpha_val <= -0.1, f"Risk-seeking alpha must be in [-10.0, -0.1], got: {alpha_val}"

            agent = cls(model, alpha=alpha_val, consume_per_step=consume_per_step, consume_proportion=consume_proportion, consume_prop_mode=consume_prop_mode, is_cooperator=is_coop)
            agents.append(agent)
            model.agents.add(agent)
        
        return agents


class SugarAgent_Aversion(Agent):
    """Risk-averse agents with diminishing returns utility function"""

    def __init__(self, model, alpha=1.0, consume_per_step=1, consume_proportion=0.1, consume_prop_mode=False, is_cooperator=False):
        super().__init__(model)
        self.sugar_level = 0
        self.agent_type = "risk_averse"
        self.alpha = alpha
        self.is_cooperator = is_cooperator
        self.consume_per_step = consume_per_step
        self.consume_proportion = consume_proportion
        self.consume_prop_mode = consume_prop_mode

    def compute_utility(self, pos):
        """
            Risk-adversion utility:         
            U(c) = (1 - exp(-a * c)) / a   if a != 0
            = c                      if a == 0

            a > 0 means risk-averse behavior
        """
        x, y = pos
        c = self.model.grid_sugar[x, y]
        if self.alpha != 0:
            utility = (1 - np.exp(-self.alpha * c)) / self.alpha
        else:
            utility = c

        return utility
    
    def choose_move(self):
        """Logit model for movement decisions"""
        neighbors = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=True
        )
        
        available_positions = []
        for pos in neighbors:
            cell_contents = self.model.grid.get_cell_list_contents([pos])
            if len(cell_contents) == 0 or pos == self.pos:
                available_positions.append(pos)
        
        if not available_positions:
            return self.pos
        
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
        """Cooperation phase"""
        if not self.is_cooperator:
            return
        
        neighbors = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        
        empty_spots = []
        for pos in neighbors:
            cell_contents = self.model.grid.get_cell_list_contents([pos])
            if len(cell_contents) == 0:
                empty_spots.append(pos)
        
        if empty_spots:
            cooperation_spot = self.random.choice(empty_spots)
            x, y = cooperation_spot
            self.model.grid_sugar[x, y] = min(self.model.grid_sugar[x, y] + 1, 4)
            self.model.sugar_layer.modify_cell(cooperation_spot, lambda v: min(v + 1, 4))

    def step(self):
        """Agent step: move, consume, cooperate"""
        new_pos = self.choose_move()

        if new_pos != self.pos:
            self.model.grid.move_agent(self, new_pos)

        x, y = new_pos
        if self.model.grid_sugar[x, y] > 0:
            if self.consume_prop_mode:
                # Consume a proportion of the sugar in the cell
                sugar_consumed = self.model.grid_sugar[x, y] * self.consume_proportion
            else:
                # Consume a fixed amount of sugar
                sugar_consumed = min(self.consume_per_step, self.model.grid_sugar[x, y])
            self.sugar_level += sugar_consumed
            self.model.grid_sugar[x, y] = max(0, self.model.grid_sugar[x, y] - sugar_consumed)
            self.model.sugar_layer.modify_cell((x, y), lambda v: max(0, v - sugar_consumed))

        # self.cooperate()

    @classmethod
    def create_agents(cls, model, num_agents, alpha_values=None, consume_per_step=1, consume_proportion=0.1, consume_prop_mode=False, random_alpha=False):
        """
        Create risk-averse agents
        
        Parameters:
        -----------
        model : Model
            Mesa model instance
        num_agents : int
            Number of agents to create
        alpha_values : float, list of floats, or None
            - If float: all agents use this fixed value
            - If list: values are assigned to agents in a cyclic manner
            - If None: use default value -1.0 or random generation (if random_alpha=True)
        consume_per_step : int
            Amount of sugar consumed per step
        random_alpha : bool
            If True and alpha_values is None, generate random values in [-10.0, -0.1]
        """
        agents = []
        cooperation_rate = getattr(model, 'cooperation_rate', 0.3)
        num_cooperators = int(num_agents * cooperation_rate)
        
        for i in range(num_agents):
            is_coop = i < num_cooperators
            
            # Determine alpha value
            if alpha_values is not None:
                if isinstance(alpha_values, (int, float)):
                    # Fixed value
                    alpha_val = float(alpha_values)
                elif isinstance(alpha_values, list):
                    # Cycle through list values
                    alpha_val = alpha_values[i % len(alpha_values)]
                else:
                    raise ValueError("alpha_values must be float or list")
            elif random_alpha:
                # Random generation
                alpha_val = model.random.uniform(-10.0, -0.1)
            else:
                # Default value
                alpha_val = -1.0
            
            # Ensure alpha is within valid range
            assert 0.1 <= alpha_val <= 10.0, f"Risk-averse alpha must be in [0.1, 10.0], got: {alpha_val}"

            agent = cls(model, alpha=alpha_val, is_cooperator=is_coop, consume_per_step=consume_per_step, consume_proportion=consume_proportion, consume_prop_mode=consume_prop_mode)
            agents.append(agent)
            model.agents.add(agent)
        
        return agents
