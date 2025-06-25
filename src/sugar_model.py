from mesa import Model
from mesa.space import MultiGrid, PropertyLayer
from mesa.datacollection import DataCollector
import numpy as np
import sugar_agent as sa
from pathlib import Path

def read_sugar_map():
    """Read the sugar map from the file"""
    sugar_map_path = Path(__file__).parent / "sugar-map.txt"
    sugar_distribution = np.genfromtxt(sugar_map_path, dtype=int)
    sugar_distribution = np.flip(sugar_distribution, axis=0)
    return sugar_distribution

class SugarModel(Model):
    def __init__(self, width=None, height=None, 
                 num_agents=100, 
                 lambda_param=1.0, 

                 cooperation_rate=1.0,# for agent cooperation
                 cooperation_threshold=2,# for agent cooperation
                 feedback_per_step_D=3, # for agent feedback and sugar growth 
                 max_sugar_per_cell=4,  # for cell sugar maximum
                 alpha_range=(-2, 2),

                 consume_per_step=1,# for agent per step consumption
                 consume_proportion=0.1, # for agent consumption proportion
                 consume_proportion_mode=False, # true for proportion, false for fixed amount

                 research_mode="balanced", 
                 research_alpha=None, 
                 **kwargs):
        """
        Initialize the sugar model with the given parameters.
        """
        # For select research_mode, convert to string if necessary
        try:
            research_mode = research_mode.value
            print("Extracted .value from research_mode")
        except Exception:
            print("Used research_mode as-is (already a string)")

        super().__init__()
        
        # Model parameters
        self.lambda_param = lambda_param
        self.cooperation_rate = cooperation_rate
        self.cooperation_threshold = cooperation_threshold
        self.alpha_range = alpha_range
        self.research_mode = research_mode
        self.research_alpha = research_alpha
        self.feedback_per_step_D = feedback_per_step_D
        self.max_sugar_per_cell = max_sugar_per_cell
        if consume_proportion_mode:
            self.consume_proportion = consume_proportion
            self.consume_per_step = None  # Use proportion instead of fixed amount
        else:
            self.consume_proportion = None
            self.consume_per_step = consume_per_step
        
        # CRITICAL FIX: Always read sugar map first and use ITS dimensions
        self.grid_sugar = read_sugar_map()
        actual_height, actual_width = self.grid_sugar.shape  # numpy gives (rows, cols)
        
        print(f"Sugar map loaded: {actual_height} rows x {actual_width} cols")
        print(f"Ignoring any width/height parameters: width={width}, height={height}")
        
        # Create grid with ACTUAL dimensions from sugar map
        self.grid = MultiGrid(actual_width, actual_height, torus=False)
        
        # Store the actual dimensions for reference
        self.actual_width = actual_width
        self.actual_height = actual_height
        
        # Add property layer for sugar visualization with CORRECT dimensions
        self.sugar_layer = PropertyLayer(
            "sugar",
            width=actual_width,
            height=actual_height,
            default_value=0.0,
            dtype=float,
        )
        self.sugar_layer.set_cells(self.grid_sugar)
        self.grid.add_property_layer(self.sugar_layer)
        
        # Create agents based on research mode
        max_possible_agents = actual_width * actual_height
        safe_total_agents = min(num_agents, max_possible_agents // 3)
        
        print(f"Creating {safe_total_agents} agents (requested: {num_agents})")
        print(f"Research mode: {research_mode}")


        if research_mode == "balanced":
            # Equal distribution of all types with default alphas
            agents_per_type = safe_total_agents // 3
            remaining = safe_total_agents - (agents_per_type * 3)
            
            sa.SugarAgent_Neutral.create_agents(self, agents_per_type, consume_per_step=consume_per_step, consume_proportion=consume_proportion, consume_prop_mode=consume_proportion_mode)
            sa.SugarAgent_Riskseeking.create_agents(self, agents_per_type, consume_per_step=consume_per_step, consume_proportion=consume_proportion, consume_prop_mode=consume_proportion_mode)
            sa.SugarAgent_Aversion.create_agents(self, agents_per_type + remaining, consume_per_step=consume_per_step, consume_proportion=consume_proportion, consume_prop_mode=consume_proportion_mode)

        elif research_mode == "risk_seeking":
            # Study risk-seeking agents with variable alpha
            agents_per_type = safe_total_agents // 3
            remaining = safe_total_agents - (agents_per_type * 3)
            
            # Fixed types
            sa.SugarAgent_Neutral.create_agents(self, agents_per_type, consume_per_step=consume_per_step, consume_proportion=consume_proportion, consume_prop_mode=consume_proportion_mode)
            sa.SugarAgent_Aversion.create_agents(self, agents_per_type, consume_per_step=consume_per_step, consume_proportion=consume_proportion, consume_prop_mode=consume_proportion_mode)

            # Variable risk-seeking
            sa.SugarAgent_Riskseeking.create_agents(
                self, 
                agents_per_type + remaining,
                alpha_values=research_alpha,
                consume_per_step=consume_per_step,
                consume_proportion=consume_proportion,
                consume_prop_mode=consume_proportion_mode
            )
            
        elif research_mode == "risk_averse":
            # Study risk-averse agents with variable alpha
            agents_per_type = safe_total_agents // 3
            remaining = safe_total_agents - (agents_per_type * 3)
            
            # Fixed types
            sa.SugarAgent_Neutral.create_agents(self, agents_per_type, consume_per_step=consume_per_step, consume_proportion=consume_proportion, consume_prop_mode=consume_proportion_mode)
            sa.SugarAgent_Riskseeking.create_agents(self, agents_per_type, consume_per_step=consume_per_step, consume_proportion=consume_proportion, consume_prop_mode=consume_proportion_mode)

            # Variable risk-averse
            sa.SugarAgent_Aversion.create_agents(
                self, 
                agents_per_type + remaining,
                alpha_values=research_alpha,
                consume_per_step=consume_per_step,
                consume_proportion=consume_proportion,
                consume_prop_mode=consume_proportion_mode
            )
        else:
            raise ValueError(f"Unknown research_mode: {research_mode}")

        # Place agents randomly
        for agent in self.agents:
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Data collection
        self.datacollector = DataCollector(
            model_reporters={
                "TotalSugar": lambda m: np.sum(m.grid_sugar),
                "AvgSugarLevel": lambda m: np.mean([a.sugar_level for a in m.agents]) if m.agents else 0,
                "SugarGini": self.calculate_gini,
                "NumCooperators": lambda m: sum(1 for a in m.agents if hasattr(a, 'is_cooperator') and a.is_cooperator),

                # add coop or not column
                "RiskAverseAvgSugar": lambda m: np.mean([a.sugar_level for a in m.agents if hasattr(a, 'agent_type') and a.agent_type == "risk_averse"]) if any(hasattr(a, 'agent_type') and a.agent_type == "risk_averse" for a in m.agents) else 0,
                "NeutralAvgSugar": lambda m: np.mean([a.sugar_level for a in m.agents if hasattr(a, 'agent_type') and a.agent_type == "neutral"]) if any(hasattr(a, 'agent_type') and a.agent_type == "neutral" for a in m.agents) else 0,
                "RiskSeekingAvgSugar": lambda m: np.mean([a.sugar_level for a in m.agents if hasattr(a, 'agent_type') and a.agent_type == "risk_seeking"]) if any(hasattr(a, 'agent_type') and a.agent_type == "risk_seeking" for a in m.agents) else 0,
            },
            agent_reporters={
                "SugarLevel": lambda a: a.sugar_level,
                "AgentType": lambda a: getattr(a, 'agent_type', 'unknown'),
                "IsCooperator": lambda a: getattr(a, 'is_cooperator', False),
                "Alpha": lambda a: getattr(a, 'alpha', 0.0),
            },
        )
        
        print(f"Model initialized successfully with {len(self.agents)} agents")
        print(f"Grid size: {self.grid.width} x {self.grid.height}")
        print(f"Sugar layer size: {self.sugar_layer.width} x {self.sugar_layer.height}")
        
        self.datacollector.collect(self)

    def calculate_gini(self):
        """Calculate Gini coefficient for wealth inequality"""
        if not self.agents:
            return 0
            
        sugar_levels = [agent.sugar_level for agent in self.agents]
        if len(sugar_levels) == 0 or all(level == 0 for level in sugar_levels):
            return 0
        
        # Sort sugar levels
        sorted_levels = sorted(sugar_levels)
        n = len(sorted_levels)
        
        # Calculate Gini coefficient
        cumsum = np.cumsum(sorted_levels)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0

    def step(self):
        """Execute one step of the model"""
        # Agent actions (movement, consumption, cooperation)
        self.agents.shuffle_do("step")
        
        # Environmental sugar regeneration (reduced since cooperators now grow sugar)
        self.update_environment_sugar()
        
        # Collect data
        self.datacollector.collect(self)

    def update_environment_sugar(self):
        """
        Reduced environmental sugar growth since cooperators now handle growth
        Only minimal natural regeneration
        """
        for x in range(1, self.grid.width - 1):
            for y in range(1, self.grid.height - 1):
                # # Natural regeneration is much slower now
                # if self.random.random() < 0.01:  # 1% chance per cell per step
                #     if self.grid_sugar[x, y] < 4:  # Don't exceed maximum
                #         self.grid_sugar[x, y] += 1
                #         self.sugar_layer.modify_cell((x, y), lambda v: min(v + 1, 4))

                # check neighbors for sugar growth
                neighbors_list = self.grid.get_neighbors((x, y), moore=True, include_center=False)
                cooperators = [agent for agent in neighbors_list if hasattr(agent, 'is_cooperator') and agent.is_cooperator]

                if len(cooperators) > self.cooperation_threshold:
                    # Calculate average sugar level of neighbors
                    if self.grid_sugar[x, y] < self.max_sugar_per_cell:  # Don't exceed maximum
                        self.grid_sugar[x, y] += self.feedback_per_step_D
                        self.sugar_layer.modify_cell((x, y), lambda v: min(v + 1, self.max_sugar_per_cell))

    def get_agent_type_counts(self):
        """Get counts of each agent type"""
        counts = {"risk_averse": 0, "neutral": 0, "risk_seeking": 0}
        for agent in self.agents:
            if hasattr(agent, 'agent_type'):
                agent_type = agent.agent_type
                if agent_type in counts:
                    counts[agent_type] += 1
        return counts

    def get_cooperation_stats(self):
        """Get cooperation statistics by agent type"""
        stats = {
            "risk_averse_cooperators": 0,
            "neutral_cooperators": 0,
            "risk_seeking_cooperators": 0,
            "total_cooperators": 0
        }
        
        for agent in self.agents:
            if hasattr(agent, 'is_cooperator') and agent.is_cooperator:
                stats["total_cooperators"] += 1
                if hasattr(agent, 'agent_type'):
                    agent_type = agent.agent_type
                    if agent_type in ["risk_averse", "neutral", "risk_seeking"]:
                        stats[f"{agent_type}_cooperators"] += 1
        
        return stats
    
    def clean(self):
        """Clean up references to break potential circular references and help GC"""
        print("Cleaning SugarModel...")

        # clear agents and their references
        if hasattr(self, 'agents'):
            for agent in self.agents:
                if hasattr(agent, 'model'):
                    agent.model = None  # If agent has reverse model reference
                if hasattr(agent, 'grid'):
                    agent.grid = None  # Clear agent's reference to grid
            self.agents.clear()

        # Clear grid and property layer
        if hasattr(self, 'grid'):
            del self.grid
            self.grid = None
        
        if hasattr(self, 'sugar_layer'):
            self.sugar_layer = None

        # Clear datacollector's closure-held model references
        if hasattr(self, 'datacollector'):
            self.datacollector.model_reporters.clear()
            self.datacollector.agent_reporters.clear()
            self.datacollector = None

        # Clear sugar grid
        if hasattr(self, 'grid_sugar'):
            self.grid_sugar = None

        # Clear remaining attributes
        self.schedule = None
        self.random = None
        self.running = False

        print("SugarModel cleaned.")