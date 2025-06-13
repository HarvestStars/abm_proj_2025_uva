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
    def __init__(self, width=None, height=None, num_agents=100, lambda_param=1.0, 
                 cooperation_rate=0.3, alpha_range=(-2, 2), **kwargs):
        """
        Initialize the Sugar model with risk preferences and cooperation
        
        COMPLETELY IGNORE width/height parameters - ALWAYS use sugar map dimensions
        """
        super().__init__()
        
        # Model parameters
        self.lambda_param = lambda_param
        self.cooperation_rate = cooperation_rate
        self.alpha_range = alpha_range
        
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
        
        # Create agents - ensure we don't create more agents than the grid can reasonably hold
        max_possible_agents = actual_width * actual_height
        safe_num_agents = min(num_agents, max_possible_agents // 3)  # Use at most 1/3 of the grid
        
        print(f"Creating {safe_num_agents} agents (requested: {num_agents})")
        
        sa.SugarAgent.create_agents(self, safe_num_agents)
        
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
                # Natural regeneration is much slower now
                if self.random.random() < 0.01:  # 1% chance per cell per step
                    if self.grid_sugar[x, y] < 4:  # Don't exceed maximum
                        self.grid_sugar[x, y] += 1
                        self.sugar_layer.modify_cell((x, y), lambda v: min(v + 1, 4))

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

# Test the model dimensions
if __name__ == '__main__':
    print("Testing model creation with various parameters...")
    
    # Test 1: Default parameters
    model1 = SugarModel()
    print(f"Test 1 - Grid: {model1.grid.width}x{model1.grid.height}")
    
    # Test 2: With different width/height (should be ignored)
    model2 = SugarModel(width=20, height=30)
    print(f"Test 2 - Grid: {model2.grid.width}x{model2.grid.height}")
    
    # Test 3: With Mesa parameters (should be ignored)
    model3 = SugarModel(width=10, height=10, num_agents=50)
    print(f"Test 3 - Grid: {model3.grid.width}x{model3.grid.height}")
    
    print("All tests should show the same grid dimensions (50x48)!")