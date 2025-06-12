# Density-based measure
from SALib.sample import latin          
from SALib.analyze.pawn import analyze   
import numpy as np
import matplotlib.pyplot as plt
from sugar_model import SugarModel

def run_model():
    """Run the model with given parameters and return outputs. For later re-evaluation"""
    model = SugarModel(width=10, height=10, num_agents=100) # the input parapemetr needs to be adjusted for sensitivity analysis
    for _ in range(1000):
        model.step()
    results = model.datacollector.get_model_vars_dataframe() # only evaluate total sugar as output
    return results

# Parameters to be analyzed later
problem = {
    'num_vars': 2,
    'names': ["lambda","alpha"],  # can be adjusted later
    'bounds': [[0,1000], [-5,5]]  # can be adjusted later
}

N = 1000  # base sample size
input_space = latin.sample(problem, N) # hyperlatin sampling in parameter space

# re-evaluate the model with sampled parameters
outputs = []
for i, x in enumerate(input_space):
    results = [run_model(*x) for j in range(10)] # run each model 10 times for stochasticity
    avg_result = sum(results) / len(results) # get the average output out of 10 simulations
    outputs.append(avg_result)

# print the pawn results and plot graphs
X = np.array(input_space) 
Y = np.array(outputs) 

# plot unconditioned CDF
sorted_Y = np.sort(Y)
cdf_Y = np.arange(1, len(Y) + 1) / len(Y)
plt.figure(figsize=(10, 6))
plt.plot(sorted_Y, cdf_Y, label='Unconditional CDF', color='red', linewidth=2)

# plot conditioned CDFs for each parameter (to be implemented later)

# analysis with PAWN indices
Si = analyze(problem, X, Y, S=10, print_to_console=True)
print("Median PAWN indices:", Si['median'])  # this result is for factor prioritization only

# use PAWN indices for factor fixing
ks_all = Si['KS'] # read the KS score for each parameter
critical_value = 0.15 # set the criticial value (need to be based on literature)
fixed_factors = []
for ks_vals in ks_all:
    fixable = all(k < critical_value for k in ks_vals) # check the ks values and eliminate those below the critical values
    fixed_factors.append(fixable)
    print(f"Parameter {ks_vals} fixable: {fixable}")

# show which parameters are below the critical KS level
for i, ks_vals in enumerate(Si['KS']):
    plt.figure()
    plt.plot(ks_vals, 'o-', label='KS values') 
    plt.axhline(Si['median'][i], color='blue', linestyle='-', label='median KS') # plot the median
    plt.axhline(critical_value, color='black', linestyle='--', label='critical KS') # plot the critical level
    plt.title(f'Factor Fixing Check for {problem["names"][i]}')
    plt.ylabel('KS')
    plt.xlabel('Conditioning index')
    plt.legend()
    plt.grid(True)
    plt.show()