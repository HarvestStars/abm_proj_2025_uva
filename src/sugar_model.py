from mesa import Model
from mesa.space import MultiGrid, PropertyLayer
from mesa.datacollection import DataCollector
import numpy as np
import sugar_agent as sa


# --- Model Class ---
class SugarModel(Model):
    def __init__(self, width=10, height=10, num_agents=20):
        super().__init__()
        self.grid = MultiGrid(width, height, torus=False)
        self.grid_sugar = np.random.randint(1, 6, size=(width, height))
        self.sugar_layer = PropertyLayer(
            "sugar",
            width=width,
            height=height,
            default_value=0,
        )
        self.sugar_layer.set_cells(self.grid_sugar)
        self.grid.add_property_layer(self.sugar_layer)

        sa.SugarAgent.create_agents(self, num_agents)
        for agent in self.agents:
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={"TotalSugar": lambda m: np.sum(m.grid_sugar)}
        )

    def step(self):
        self.agents.shuffle_do("step")
        self.update_environment_sugar()
        self.datacollector.collect(self)

    def update_environment_sugar(self):
        for x in range(1, self.grid.width - 1):
            for y in range(1, self.grid.height - 1):
                neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
                if all(len(self.grid.get_cell_list_contents([n])) > 0 for n in neighbors):
                    self.grid_sugar[x, y] += 10
                    self.sugar_layer.modify_cell((x, y), lambda v: v + 10)


# --- Run Workflow Example ---
if __name__ == '__main__':
    model = SugarModel()
    for i in range(100):
        model.step()

    # Access collected data
    results = model.datacollector.get_model_vars_dataframe()
    print(results.tail())
