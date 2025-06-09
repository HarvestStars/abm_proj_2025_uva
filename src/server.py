import solara
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mesa.visualization import Slider, SolaraViz, make_plot_component
from mesa.visualization.components.matplotlib_components import make_mpl_space_component
from mesa.visualization.solara_viz import update_counter 
from sugar_model import SugarModel

# 1. Agent portrayal: simple blue circles
def agent_portrayal(agent):
    sugar_absorb = agent.sugar_level

    # max sugar level for normalization
    max_sugar = 20
    sugar_norm = min(sugar_absorb / max_sugar, 1.0)  # normalize to 0~1

    # red channel fixed at 1 (max red), other channels decrease with sugar amount
    r = 1.0
    g = 1.0 - sugar_norm  # more sugar, less green
    b = 1.0 - sugar_norm  # more sugar, less blue

    size = 5 + sugar_absorb * 1
    return {"marker": "o", "color": (r, g, b), "size": size}

# 2. Property-layer portrayal: sugar field in orange gradient
propertylayer_portrayal = {
    "sugar": {"color": "orange", "alpha": 0.8, "colorbar": True, "vmin": 0, "vmax": 10}
}

# 3. Create the space component using Matplotlib
sugarscape_space = make_mpl_space_component(
    agent_portrayal=agent_portrayal,
    propertylayer_portrayal=propertylayer_portrayal,
    post_process=None,
    draw_grid=False,
)

# 4. Model parameters for interactive sliders
model_params = {
    "width": 10,
    "height": 10,
    "num_agents": 2,
}

# 5. Hook up SolaraViz with space and plot components
@solara.component
def SugarLevelHistogram(model):
    update_counter.get()  # ensure the component updates on each step
    fig = Figure()
    ax = fig.subplots()

    sugar_levels = [agent.sugar_level for agent in model.agents]
    print("Sugar levels:", sugar_levels)  # debugging output

    ax.hist(sugar_levels, bins=range(0, 100), edgecolor="black", color="red")
    ax.set_title("Sugar Level Distribution")
    ax.set_xlabel("Sugar Level")
    ax.set_ylabel("Number of Agents")

    return solara.FigureMatplotlib(fig)

model = SugarModel()
page = SolaraViz(
    model,
    components=[
        sugarscape_space,
        make_plot_component("TotalSugar"),
        SugarLevelHistogram,
    ],
    model_params=model_params,
    name="Sugarscape ABM",
    play_interval=150,
)

page    # Display the visualization
