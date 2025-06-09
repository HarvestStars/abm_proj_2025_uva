from mesa.visualization import Slider, SolaraViz, make_plot_component
from mesa.visualization.components.matplotlib_components import make_mpl_space_component
from sugar_model import SugarModel

# 1. Agent portrayal: simple blue circles

def agent_portrayal(agent):
    return {"marker": "o", "color": "blue", "size": 10}

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

model = SugarModel()
page = SolaraViz(
    model,
    components=[
        sugarscape_space,
        make_plot_component("TotalSugar"),
    ],
    model_params=model_params,
    name="Sugarscape ABM",
    play_interval=150,
)

page    # Display the visualization
