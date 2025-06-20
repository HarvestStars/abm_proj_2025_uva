from mesa import Model
from mesa.space import MultiGrid, PropertyLayer
from mesa.datacollection import DataCollector
import numpy as np
import sugar_agent as sa
from pathlib import Path

def read_sugar_map():
    # get the path to the sugar map file
    sugar_map_path = Path(__file__).parent / "sugar-map.txt"

    # read the sugar map from the file
    sugar_distribution = np.genfromtxt(sugar_map_path, dtype=int)

    # flip the sugar distribution vertically
    sugar_distribution = np.flip(sugar_distribution, axis=0)

    return sugar_distribution

# --- Model Class ---
class SugarModel(Model):
    def __init__(self, width=50, height=50, num_agents=30):
        super().__init__()
        self.grid = MultiGrid(width, height, torus=False)
        self.grid_sugar = read_sugar_map()
        width, height = self.grid_sugar.shape # mandatory: ensure the sugar grid dimensions match the sugar-map.txt file
        
        # add a property layer for sugar 
        self.sugar_layer = PropertyLayer(
            "sugar",
            width=width,
            height=height,
            default_value=0.0,
            dtype=float,
        )
        self.sugar_layer.set_cells(self.grid_sugar)
        self.grid.add_property_layer(self.sugar_layer)

        sa.SugarAgent.create_agents(self, num_agents)
        for agent in self.agents:
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={"TotalSugar": lambda m: np.sum(m.grid_sugar)},
            agent_reporters={"SugarLevel": lambda a: a.sugar_level},
        )

    def step(self):
        self.agents.shuffle_do("step")
        self.update_environment_sugar()
        self.datacollector.collect(self)

    def update_environment_sugar(self):
        for x in range(1, self.grid.width - 1):
            for y in range(1, self.grid.height - 1):
                neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
                if sum(len(self.grid.get_cell_list_contents([n])) for n in neighbors) >= 3:
                    self.grid_sugar[x, y] += 2
                    self.sugar_layer.modify_cell((x, y), lambda v: v + 2)

# --- Run Workflow Example ---
if __name__ == '__main__':
    model = SugarModel()
    for i in range(1000):
        model.step()

    # Access collected data
    results = model.datacollector.get_model_vars_dataframe()
    agent_df = model.datacollector.get_agent_vars_dataframe()
    results.to_csv("sugar_model_results_lamda_1.csv")
    agent_df.to_csv("sugar_agent_results_lamda_1.csv")

    # Plotting (Histogram)
    import matplotlib.pyplot as plt
    # get step = 100, which is the last step
    print("Agent Sugar Level Distribution at Step 1000:")
    print(agent_df[agent_df.index.get_level_values(0) == 1000]["SugarLevel"])

    # group by sugar level and count occurrences
    plt.figure(figsize=(10, 6))
    distribution = agent_df[agent_df.index.get_level_values(0) == 1000].groupby("SugarLevel").size()
    print("Sugar Level Distribution:", distribution)

    agent_df[agent_df.index.get_level_values(0) == 1000]["SugarLevel"].hist(bins=range(0, 1000), edgecolor="black")
    plt.xlabel("Sugar Level")
    plt.ylabel("Number of Agents")
    plt.title("Agent Sugar Level Distribution")
    plt.show()