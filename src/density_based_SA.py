# Density-based measure
from SALib.sample import latin          
from SALib.analyze.pawn import analyze   
import numpy as np
import matplotlib.pyplot as plt
from sugar_model import SugarModel
from modify_pawn import modified_analyze # Import the modified analyze function

def run_model(alpha, lambda_):
    """Run the model with given parameters and return outputs. For later re-evaluation"""
    model = SugarModel(width=10, height=10, num_agents=100, alpha = alpha, lambda_ = lambda_) # the input parapemetr needs to be adjusted for sensitivity analysis
    for _ in range(100):
        model.step()
    results = model.datacollector.get_model_vars_dataframe()
    result = results["TotalSugar"].iloc[-1] # only evaluate total sugar as output (final step)
    return result


def hyperlatin_sample(problem, N):
    """Generate hyperlatin samples for the given problem.
    N: base sample size"""
    input_space = latin.sample(problem, N)
    return input_space


def model_revaluation(input_space):
    """Evaluate the model with the given input space.
    input_space: array of parameter samples"""
    outputs = []
    for i, x in enumerate(input_space):
        results = [run_model(*x) for j in range(10)]  # run each model 10 times for stochasticity
        avg_result = sum(results) / len(results)  # get the average output out of 10 simulations
        outputs.append(avg_result)
        print(f"Sample {i+1}: lambda={x[0]}, alpha={x[1]}, total_sugar={avg_result}")
    return np.array(outputs)


def pawn_analysis(outputs, input_space,problem):
    """Perform PAWN analysis on the outputs and input space.
    value: which output we want for the Si analysis"""
    Y = np.array(outputs) # only take the total sugar as output
    X = np.array(input_space)
    
    # Perform PAWN analysis
    Si = modified_analyze(problem, X, Y, S=10, print_to_console=True) #TODO: need to figure out how Si is calculated
    return Si # curently we only calculate the Si indices
    # read the document here: https://salib.readthedocs.io/en/latest/_modules/SALib/analyze/pawn.html


def plot_cdf(outputs, input_space, problem):
    """Plot the unconditional CDF and conditional CDFs for each parameter."""
    Y = np.array(outputs) # only take the total sugar as output
    X = np.array(input_space)

    # plot the unconditional CDF
    sorted_Y = np.sort(Y)
    cdf_Y = np.arange(1, len(Y) + 1) / len(Y)
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_Y, cdf_Y, label='Unconditional CDF', color='red', linewidth=2)
    plt.title("Unconditional CDF of Total Sugar")
    plt.xlabel("Model Output (e.g. total sugar)")
    plt.ylabel("Cumulative Probability")
    plt.grid()
    plt.legend()
    plt.show()

    # plot the conditional CDF
    percentiles = [10, 50, 90] # take 10th, 50th, and 90th percentiles for conditioning
    for index in range(problem['num_vars']):
        param_name = problem['names'][index]
        fixed_vals = np.percentile(X[:, index], percentiles) # calculate percentiple for each parameter
        
        for val in fixed_vals:
            # Create a mask for samples
            mask = np.abs(X[:, index] - val) < 0.05 * (problem['bounds'][index][1] - problem['bounds'][index][0])
            Y_cond = Y[mask]
            sorted_Yc = np.sort(Y_cond)
            cdf_Yc = np.arange(1, len(Y_cond) + 1) / len(Y_cond)
            print(f"{param_name} ≈ {val:.2f} → {np.sum(mask)} samples")
            plt.plot(sorted_Yc, cdf_Yc, label=f"{param_name} ≈ {val:.2f}", linestyle = "--", linewidth = 2) # plot the conditional cdf for each parameter

        plt.plot(sorted_Y, cdf_Y, label='Unconditional CDF', color='red', linewidth=2) # plot unconditional again
        plt.title(f"CDF Comparison: Conditioning on {param_name}")
        plt.xlabel("Model Output (e.g. total sugar)")
        plt.ylabel("Cumulative Probability")
        plt.grid()
        plt.legend()
        plt.show()


def factor_fixing(Si):
    """Perform factor fixing based on the Si indices."""
    print(Si.keys())
    ks_all = Si['KS'] # read the KS score for each parameter
    critical_value = 0.15 # set the criticial value (need to be based on literature)
    fixed_factors = []
    for ks_vals in ks_all:
        fixable = all(k < critical_value for k in ks_vals) # check the ks values and eliminate those below the critical values
        fixed_factors.append(fixable)
        print(f"Parameter {ks_vals} fixable: {fixable}")
    return fixed_factors


def plot_ks_level(Si, problem, critical_value = 0.15):
    """Plot the KS level and show which parameters are below the critical KS level."""
    for i, ks_vals in enumerate(Si['KS']):
        plt.figure()
        plt.plot(ks_vals, 'o-', label='KS values') 
        plt.axhline(Si['median'][i], color='blue', linestyle='-', label='median KS') # plot the median
        plt.axhline(critical_value, color='black', linestyle='--', label='critical KS') # plot the critical level
        plt.title(f'Factor Fixing Check for {problem["names"][i]}')
        plt.ylabel('KS')
        plt.xlabel('Conditioning index')
        plt.legend()
        plt.grid()
        plt.show()

if __name__ == "__main__":
    problem = {
    'num_vars': 2,
    'names': ["lambda","alpha"],  # can be adjusted later
    'bounds': [[0,100], [-5,5]]  # can be adjusted later
}
    # generate parameter sample space
    input_space = hyperlatin_sample(problem, N=50)
    print("Start creating the space")

    # re-evaluate the model with the input space
    outputs = model_revaluation(input_space)
    print("Model re-evaluation completed.")

    # perform PAWN analysis
    Si = pawn_analysis(outputs, input_space, problem)
    print("PAWN analysis completed.")

    # plot CDFs (both unconditional and conditional)
    plot_cdf(outputs, input_space, problem)

    # perform factor fixing
    fixed_factors = factor_fixing(Si)
    print("Fixed Factors:", fixed_factors)

    # plot KS graph
    plot_ks_level(Si, problem)



