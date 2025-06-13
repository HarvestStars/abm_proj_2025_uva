import solara
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mesa.visualization import Slider, SolaraViz, make_plot_component
from mesa.visualization.components.matplotlib_components import make_mpl_space_component
from mesa.visualization.solara_viz import update_counter 
from sugar_model import SugarModel
import numpy as np

# Agent portrayal with risk type visualization
def agent_portrayal(agent):
    """
    Visualize agents with different colors based on risk type and cooperation status
    """
    sugar_level = agent.sugar_level
    max_sugar = 50  # Normalize display
    sugar_norm = min(sugar_level / max_sugar, 1.0)
    
    # Base colors by agent type
    if hasattr(agent, 'agent_type'):
        if agent.agent_type == "risk_averse":
            base_color = (0.2, 0.6, 1.0)  # Blue for risk-averse
        elif agent.agent_type == "risk_seeking":
            base_color = (1.0, 0.2, 0.2)  # Red for risk-seeking
        else:  # neutral
            base_color = (0.2, 0.8, 0.2)  # Green for neutral
    else:
        base_color = (0.5, 0.5, 0.5)  # Gray for unknown type
    
    # Adjust intensity based on sugar level
    color_intensity = 0.3 + 0.7 * sugar_norm
    final_color = tuple(c * color_intensity for c in base_color)
    
    # Size based on cooperation status
    if hasattr(agent, 'is_cooperator'):
        size = 12 if agent.is_cooperator else 8
        marker = "s" if agent.is_cooperator else "o"  # Square for cooperators, circle for others
    else:
        size = 8
        marker = "o"
    
    return {
        "marker": marker, 
        "color": final_color, 
        "size": size,
        "alpha": 0.8
    }

# Property layer portrayal for sugar field
propertylayer_portrayal = {
    "sugar": {"color": "orange", "alpha": 0.6, "colorbar": True, "vmin": 0, "vmax": 4}
}

# Create space component
sugarscape_space = make_mpl_space_component(
    agent_portrayal=agent_portrayal,
    propertylayer_portrayal=propertylayer_portrayal,
    post_process=None,
    draw_grid=False,
)

# FIXED: Get actual dimensions from sugar map
def get_sugar_map_dimensions():
    """Read sugar map to get actual dimensions"""
    from pathlib import Path
    sugar_map_path = Path(__file__).parent / "sugar-map.txt"
    sugar_distribution = np.genfromtxt(sugar_map_path, dtype=int)
    height, width = sugar_distribution.shape  # Note: numpy gives (rows, cols) = (height, width)
    return width, height

# Get the actual dimensions
ACTUAL_WIDTH, ACTUAL_HEIGHT = get_sugar_map_dimensions()
print(f"Sugar map dimensions: {ACTUAL_WIDTH} x {ACTUAL_HEIGHT}")

# FIXED: Model parameters - remove width/height to prevent override
model_params = {
    # DO NOT include width/height here - they will override the sugar map dimensions
    "num_agents": Slider("Number of Agents", value=100, min=50, max=200, step=10),
    "lambda_param": Slider("Lambda (Logit Noise)", value=1.0, min=0.1, max=50.0, step=0.5),
    "cooperation_rate": Slider("Cooperation Rate", value=0.3, min=0.0, max=0.8, step=0.05),
}

# Custom histogram for sugar levels by agent type
@solara.component
def SugarLevelByTypeHistogram(model):
    update_counter.get()
    fig = Figure(figsize=(12, 8))
    
    # Create subplots for different visualizations
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    
    # Collect data by agent type
    risk_averse_sugar = [agent.sugar_level for agent in model.agents if hasattr(agent, 'agent_type') and agent.agent_type == "risk_averse"]
    neutral_sugar = [agent.sugar_level for agent in model.agents if hasattr(agent, 'agent_type') and agent.agent_type == "neutral"]
    risk_seeking_sugar = [agent.sugar_level for agent in model.agents if hasattr(agent, 'agent_type') and agent.agent_type == "risk_seeking"]
    
    # Cooperator vs non-cooperator data
    cooperator_sugar = [agent.sugar_level for agent in model.agents if hasattr(agent, 'is_cooperator') and agent.is_cooperator]
    non_cooperator_sugar = [agent.sugar_level for agent in model.agents if hasattr(agent, 'is_cooperator') and not agent.is_cooperator]
    
    # Plot 1: Sugar distribution by risk type
    all_sugar_values = risk_averse_sugar + neutral_sugar + risk_seeking_sugar
    if all_sugar_values:
        max_sugar = max(all_sugar_values)
        bins = range(0, max_sugar + 5, 2) if max_sugar > 0 else [0, 1, 2]
        
        if risk_averse_sugar:
            ax1.hist(risk_averse_sugar, bins=bins, alpha=0.7, color='blue', label='Risk Averse', density=True)
        if neutral_sugar:
            ax1.hist(neutral_sugar, bins=bins, alpha=0.7, color='green', label='Neutral', density=True)
        if risk_seeking_sugar:
            ax1.hist(risk_seeking_sugar, bins=bins, alpha=0.7, color='red', label='Risk Seeking', density=True)
    
    ax1.set_title("Sugar Distribution by Risk Type")
    ax1.set_xlabel("Sugar Level")
    ax1.set_ylabel("Density")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cooperator vs Non-cooperator
    if cooperator_sugar or non_cooperator_sugar:
        all_coop_sugar = cooperator_sugar + non_cooperator_sugar
        if all_coop_sugar:
            max_sugar_coop = max(all_coop_sugar)
            bins_coop = range(0, max_sugar_coop + 5, 2) if max_sugar_coop > 0 else [0, 1, 2]
            
            if cooperator_sugar:
                ax2.hist(cooperator_sugar, bins=bins_coop, alpha=0.7, color='purple', label='Cooperators', density=True)
            if non_cooperator_sugar:
                ax2.hist(non_cooperator_sugar, bins=bins_coop, alpha=0.7, color='orange', label='Non-cooperators', density=True)
    
    ax2.set_title("Sugar Distribution: Cooperation")
    ax2.set_xlabel("Sugar Level")
    ax2.set_ylabel("Density")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Average sugar by type over time
    if hasattr(model.datacollector, 'model_vars'):
        df = model.datacollector.get_model_vars_dataframe()
        if len(df) > 1:
            steps = df.index
            ax3.plot(steps, df['RiskAverseAvgSugar'], 'b-', label='Risk Averse', linewidth=2)
            ax3.plot(steps, df['NeutralAvgSugar'], 'g-', label='Neutral', linewidth=2)
            ax3.plot(steps, df['RiskSeekingAvgSugar'], 'r-', label='Risk Seeking', linewidth=2)
            
            ax3.set_title("Average Sugar Levels Over Time")
            ax3.set_xlabel("Time Steps")
            ax3.set_ylabel("Average Sugar Level")
            ax3.legend()
            ax3.grid(True, alpha=0.3)
    
    return solara.FigureMatplotlib(fig)

# Gini coefficient over time component
@solara.component
def GiniCoefficientPlot(model):
    update_counter.get()
    fig = Figure(figsize=(10, 6))
    ax = fig.subplots()
    
    if hasattr(model.datacollector, 'model_vars'):
        df = model.datacollector.get_model_vars_dataframe()
        if len(df) > 1:
            steps = df.index
            ax.plot(steps, df['SugarGini'], 'purple', linewidth=2, label='Gini Coefficient')
            ax.axhline(y=0.4, color='red', linestyle='--', alpha=0.7, label='High Inequality Threshold')
            ax.set_title("Wealth Inequality Over Time (Gini Coefficient)")
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Gini Coefficient")
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    return solara.FigureMatplotlib(fig)

# Cooperation statistics component
@solara.component
def CooperationStats(model):
    update_counter.get()
    fig = Figure(figsize=(10, 4))
    ax1, ax2 = fig.subplots(1, 2)
    
    # Count cooperators by type
    if hasattr(model, 'get_cooperation_stats'):
        coop_stats = model.get_cooperation_stats()
    else:
        coop_stats = {
            'risk_averse_cooperators': 0,
            'neutral_cooperators': 0,
            'risk_seeking_cooperators': 0
        }
    
    # Plot 1: Cooperators by type
    types = ['Risk Averse', 'Neutral', 'Risk Seeking']
    cooperator_counts = [
        coop_stats.get('risk_averse_cooperators', 0),
        coop_stats.get('neutral_cooperators', 0),
        coop_stats.get('risk_seeking_cooperators', 0)
    ]
    colors = ['blue', 'green', 'red']
    
    bars = ax1.bar(types, cooperator_counts, color=colors, alpha=0.7)
    ax1.set_title("Cooperators by Risk Type")
    ax1.set_ylabel("Number of Cooperators")
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars, cooperator_counts):
        if count > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom')
    
    # Plot 2: Cooperation rate over time
    if hasattr(model.datacollector, 'model_vars'):
        df = model.datacollector.get_model_vars_dataframe()
        if len(df) > 1 and 'NumCooperators' in df.columns:
            steps = df.index
            total_agents = len(model.agents)
            if total_agents > 0:
                coop_rates = df['NumCooperators'] / total_agents
                
                ax2.plot(steps, coop_rates, 'purple', linewidth=2)
                ax2.set_title("Cooperation Rate Over Time")
                ax2.set_xlabel("Time Steps")
                ax2.set_ylabel("Proportion of Cooperators")
                ax2.set_ylim(0, 1)
                ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return solara.FigureMatplotlib(fig)

# FIXED: Create model with correct dimensions
def create_model_with_fixed_dimensions(**kwargs):
    """Create model ensuring correct dimensions from sugar map"""
    # Force the correct dimensions and ignore any width/height in kwargs
    kwargs_filtered = {k: v for k, v in kwargs.items() if k not in ['width', 'height']}
    
    # Create model - it will automatically use sugar map dimensions
    model = SugarModel(**kwargs_filtered)
    
    print(f"Created model with grid dimensions: {model.grid.width} x {model.grid.height}")
    print(f"Sugar map dimensions: {model.grid_sugar.shape}")
    print(f"Number of agents: {len(model.agents)}")
    
    return model

# Override the default model creation
model = create_model_with_fixed_dimensions()

page = SolaraViz(
    model,
    components=[
        sugarscape_space,
        make_plot_component("TotalSugar"),
        SugarLevelByTypeHistogram,
        GiniCoefficientPlot,
        CooperationStats,
    ],
    model_params=model_params,
    name="Enhanced Sugarscape ABM with Risk & Cooperation",
    play_interval=200,
)

# Display the visualization
page