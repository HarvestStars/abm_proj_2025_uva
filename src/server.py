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
    
    # Set marker size and type based on sugar level
    min_size, max_size = 6, 50
    size = int(min_size + (max_size - min_size) * sugar_norm)

    if getattr(agent, "is_cooperator", False):
        marker = "s"  # square
    else:
        marker = "o"  # circle
    
    return {
        "marker": marker, 
        "color": final_color, 
        "size": size,
        "alpha": 0.8
        # "tooltip": f"ID:{agent.unique_id}, sugar:{sugar_level}",
    }

# Custom component for the grid visualization with dynamic colorbar
@solara.component
def DynamicSugarscapeSpace(model):
    update_counter.get()
    
    if hasattr(model, 'value'):
        current_model = model.value
    else:
        current_model = model
    
    # Get current max sugar value
    if hasattr(current_model, 'grid_sugar'):
        current_max = np.max(current_model.grid_sugar)
        # vmax is at least 4, but if any cell exceeds 4, use the actual max
        vmax = max(4, current_max)
    else:
        vmax = 4
    
    # Create property layer portrayal with current vmax
    propertylayer_portrayal = {
        "sugar": {
            "color": "orange", 
            "alpha": 0.6, 
            "colorbar": True, 
            "vmin": 0, 
            "vmax": vmax
        }
    }
    
    # Create and return the space component
    space_component = make_mpl_space_component(
        agent_portrayal=agent_portrayal,
        propertylayer_portrayal=propertylayer_portrayal,
        post_process=None,
        draw_grid=False,
    )
    
    return space_component(current_model)

# Get actual dimensions from sugar map
def get_sugar_map_dimensions():
    """Read sugar map to get actual dimensions"""
    from pathlib import Path
    sugar_map_path = Path(__file__).parent / "sugar-map.txt"
    sugar_distribution = np.genfromtxt(sugar_map_path, dtype=int)
    height, width = sugar_distribution.shape  # Note: numpy gives (rows, cols) = (height, width)
    return width, height

ACTUAL_WIDTH, ACTUAL_HEIGHT = get_sugar_map_dimensions()
print(f"Sugar map dimensions: {ACTUAL_WIDTH} x {ACTUAL_HEIGHT}")

# Model parameters - simplified without Select widget
model_params = {
    "num_agents": Slider("Number of Agents", value=90, min=30, max=300, step=10),
    "lambda_param": Slider("Lambda (Logit Noise)", value=1.0, min=0.1, max=20.0, step=0.5),
    "cooperation_rate": Slider("Cooperation Rate", value=0.3, min=0.0, max=1.0, step=0.05),
    "cooperation_threshold": Slider("Cooperation Threshold", value=1, min=1, max=8, step=1),
    "feedback_per_step_D": Slider("Feedback per Step (D)", value=1, min=1, max=10.0, step=0.5),
    "max_sugar_per_cell": Slider("Max Sugar per Cell", value=4, min=4, max=10, step=1),
    "consume_per_step": Slider("Consume per Step", value=1, min=1, max=4, step=1),
    "consume_proportion": Slider("Consume Proportion", value=0.1, min=0.1, max=1.0, step=0.05),
    
    "research_mode": "balanced",  # Simple string default
    "research_alpha": Slider("Research Alpha", value=1.0, min=-10.0, max=10.0, step=0.5),
}

# Custom component to show research mode info
@solara.component
def ResearchModeInfo(model):
    update_counter.get()
    
    if hasattr(model, 'value'):
        current_model = model.value
    else:
        current_model = model
    
    research_mode = getattr(current_model, 'research_mode', 'balanced')
    research_alpha = getattr(current_model, 'research_alpha', None)
    
    with solara.Card("Research Configuration"):
        solara.Markdown(f"**Mode:** {research_mode}")
        
        if research_mode == "balanced":
            solara.Markdown("All agent types use default alpha values:")
            solara.Markdown("- Risk Averse: α = 1.0")
            solara.Markdown("- Neutral: α = 0.0")
            solara.Markdown("- Risk Seeking: α = -1.0")
        elif research_mode == "risk_seeking":
            solara.Markdown("Studying Risk-Seeking agents:")
            solara.Markdown(f"- Risk Seeking: α = {research_alpha}")
            solara.Markdown("- Risk Averse: α = 1.0 (fixed)")
            solara.Markdown("- Neutral: α = 0.0 (fixed)")
        elif research_mode == "risk_averse":
            solara.Markdown("Studying Risk-Averse agents:")
            solara.Markdown(f"- Risk Averse: α = {research_alpha}")
            solara.Markdown("- Risk Seeking: α = -1.0 (fixed)")
            solara.Markdown("- Neutral: α = 0.0 (fixed)")

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
    
    # Sugar distribution by risk type
    all_sugar_values = risk_averse_sugar + neutral_sugar + risk_seeking_sugar
    if all_sugar_values:
        max_sugar = max(all_sugar_values)
        print(f"Max sugar level across all agents: {max_sugar}")
        if isinstance(max_sugar, float):
            bin_size = 2.0  # or 1.0, 0.5, etc. depending on your data
            bins = np.arange(0, max_sugar + bin_size, bin_size)
        else:
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
    
    # Cooperator vs Non-cooperator
    if cooperator_sugar or non_cooperator_sugar:
        all_coop_sugar = cooperator_sugar + non_cooperator_sugar
        if all_coop_sugar:
            max_sugar_coop = max(all_coop_sugar)
            if isinstance(max_sugar_coop, float):
                bin_size = 2.0
                bins_coop = np.arange(0, max_sugar_coop + bin_size, bin_size)
            else:
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
    
    # Average sugar by type over time
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

# Alpha distribution component
@solara.component
def AlphaDistribution(model):
    update_counter.get()
    fig = Figure(figsize=(10, 6))
    ax = fig.subplots()
    
    # Collect alpha values by agent type
    risk_averse_alphas = [agent.alpha for agent in model.agents if hasattr(agent, 'agent_type') and agent.agent_type == "risk_averse"]
    neutral_alphas = [agent.alpha for agent in model.agents if hasattr(agent, 'agent_type') and agent.agent_type == "neutral"]
    risk_seeking_alphas = [agent.alpha for agent in model.agents if hasattr(agent, 'agent_type') and agent.agent_type == "risk_seeking"]
    
    # Create histograms
    bins = np.linspace(-10, 10, 41)
    
    if risk_averse_alphas:
        ax.hist(risk_averse_alphas, bins=bins, alpha=0.7, color='blue', label=f'Risk Averse (n={len(risk_averse_alphas)})')
    if neutral_alphas:
        ax.hist(neutral_alphas, bins=bins, alpha=0.7, color='green', label=f'Neutral (n={len(neutral_alphas)})')
    if risk_seeking_alphas:
        ax.hist(risk_seeking_alphas, bins=bins, alpha=0.7, color='red', label=f'Risk Seeking (n={len(risk_seeking_alphas)})')
    
    ax.set_title("Alpha Distribution by Agent Type")
    ax.set_xlabel("Alpha Value")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add vertical lines for default values
    ax.axvline(x=1.0, color='blue', linestyle='--', alpha=0.5)
    ax.axvline(x=0.0, color='green', linestyle='--', alpha=0.5)
    ax.axvline(x=-1.0, color='red', linestyle='--', alpha=0.5)
    
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
    
    # Get total counts by type
    type_counts = model.get_agent_type_counts()
    
    # Cooperators by type with total counts
    types = ['Risk Averse', 'Neutral', 'Risk Seeking']
    cooperator_counts = [
        coop_stats.get('risk_averse_cooperators', 0),
        coop_stats.get('neutral_cooperators', 0),
        coop_stats.get('risk_seeking_cooperators', 0)
    ]
    total_counts = [
        type_counts.get('risk_averse', 0),
        type_counts.get('neutral', 0),
        type_counts.get('risk_seeking', 0)
    ]
    colors = ['blue', 'green', 'red']
    
    # Create grouped bar chart
    x = np.arange(len(types))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, cooperator_counts, width, label='Cooperators', color=colors, alpha=0.7)
    bars2 = ax1.bar(x + width/2, total_counts, width, label='Total', color=colors, alpha=0.3)
    
    ax1.set_title("Agent Counts by Risk Type")
    ax1.set_ylabel("Number of Agents")
    ax1.set_xticks(x)
    ax1.set_xticklabels(types)
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars1, cooperator_counts):
        if count > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom', fontsize=8)
    for bar, count in zip(bars2, total_counts):
        if count > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom', fontsize=8)
    
    # Cooperation rate over time
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


def create_model_with_fixed_dimensions(**kwargs):
    """Create model ensuring correct dimensions from sugar map"""
    # Force the correct dimensions and ignore any width/height in kwargs
    kwargs_filtered = {k: v for k, v in kwargs.items() if k not in ['width', 'height']}
    
    # Create model - it will automatically use sugar map dimensions
    model = SugarModel(**kwargs_filtered)
    
    print(f"Created model with grid dimensions: {model.grid.width} x {model.grid.height}")
    print(f"Sugar map dimensions: {model.grid_sugar.shape}")
    print(f"Number of agents: {len(model.agents)}")
    print(f"Research mode: {model.research_mode}")
    if model.research_mode != "balanced":
        print(f"Research alpha: {model.research_alpha}")
    
    return model

# Override the default model creation
model = create_model_with_fixed_dimensions()

page = SolaraViz(
    model,
    components=[
        ResearchModeInfo,
        DynamicSugarscapeSpace,  # Use the dynamic component
        make_plot_component("TotalSugar"),
        SugarLevelByTypeHistogram,
        AlphaDistribution,
        GiniCoefficientPlot,
        CooperationStats,
    ],
    model_params=model_params,
    name="Enhanced Sugarscape ABM with Risk Preferences",
    play_interval=200,
)

# Display the visualization
page