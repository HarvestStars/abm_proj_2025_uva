from SALib.sample import latin
from SALib.analyze.pawn import analyze
import numpy as np
import matplotlib.pyplot as plt
from improved_sugar_model import SugarModel
import pandas as pd

def run_model_simulation(lambda_param, cooperation_rate, alpha_variance=1.0, num_steps=1000):
    """
    Run the model with given parameters and return key outputs
    
    Parameters:
    - lambda_param: Noise parameter for logit model
    - cooperation_rate: Proportion of cooperating agents
    - alpha_variance: Variance in risk parameters across agents
    - num_steps: Number of simulation steps
    """
    # Create model with specified parameters
    model = SugarModel(
        num_agents=100,
        lambda_param=lambda_param,
        cooperation_rate=cooperation_rate,
        alpha_range=(-alpha_variance, alpha_variance)
    )
    
    # Run simulation
    for _ in range(num_steps):
        model.step()
    
    # Extract key metrics
    results = model.datacollector.get_model_vars_dataframe()
    final_results = results.iloc[-1]
    
    return {
        'final_total_sugar': final_results['TotalSugar'],
        'final_gini': final_results['SugarGini'],
        'avg_sugar_level': final_results['AvgSugarLevel'],
        'risk_averse_avg': final_results['RiskAverseAvgSugar'],
        'neutral_avg': final_results['NeutralAvgSugar'],
        'risk_seeking_avg': final_results['RiskSeekingAvgSugar']
    }

# Define parameter space for sensitivity analysis
problem = {
    'num_vars': 3,
    'names': ['lambda_param', 'cooperation_rate', 'alpha_variance'],
    'bounds': [
        [0.1, 100.0],    # lambda: from low to high noise
        [0.0, 0.8],      # cooperation_rate: 0% to 80% cooperators
        [0.5, 3.0]       # alpha_variance: risk preference diversity
    ]
}

def conduct_sensitivity_analysis(N=500, num_replications=5):
    """
    Conduct comprehensive sensitivity analysis
    
    Parameters:
    - N: Base sample size for Latin Hypercube sampling
    - num_replications: Number of replications per parameter combination
    """
    print(f"Starting sensitivity analysis with {N} parameter combinations...")
    print(f"Each combination will be replicated {num_replications} times")
    
    # Generate parameter combinations using Latin Hypercube sampling
    param_samples = latin.sample(problem, N)
    
    # Storage for results
    all_outputs = {
        'final_total_sugar': [],
        'final_gini': [],
        'avg_sugar_level': [],
        'risk_averse_avg': [],
        'neutral_avg': [],
        'risk_seeking_avg': []
    }
    
    input_params = []
    
    # Run simulations
    for i, params in enumerate(param_samples):
        lambda_param, cooperation_rate, alpha_variance = params
        
        if i % 50 == 0:
            print(f"Progress: {i}/{N} parameter combinations completed")
        
        # Run multiple replications for each parameter combination
        replication_results = {key: [] for key in all_outputs.keys()}
        
        for rep in range(num_replications):
            try:
                results = run_model_simulation(
                    lambda_param=lambda_param,
                    cooperation_rate=cooperation_rate,
                    alpha_variance=alpha_variance,
                    num_steps=500  # Reduced steps for faster analysis
                )
                
                for key, value in results.items():
                    replication_results[key].append(value)
                    
            except Exception as e:
                print(f"Error in simulation {i}-{rep}: {e}")
                # Use default values if simulation fails
                for key in replication_results.keys():
                    replication_results[key].append(0)
        
        # Average across replications
        for key in all_outputs.keys():
            avg_value = np.mean(replication_results[key])
            all_outputs[key].append(avg_value)
        
        input_params.append(params)
    
    print("Simulation phase completed. Starting PAWN analysis...")
    
    # Convert to numpy arrays
    X = np.array(input_params)
    
    # Analyze each output variable
    sensitivity_results = {}
    
    for output_name, output_values in all_outputs.items():
        Y = np.array(output_values)
        
        # Skip if all values are the same (no variance)
        if np.std(Y) == 0:
            print(f"Skipping {output_name} - no variance in output")
            continue
        
        try:
            # Perform PAWN analysis
            Si = analyze(problem, X, Y, S=10, print_to_console=False)
            sensitivity_results[output_name] = Si
            
            print(f"\n=== PAWN Analysis for {output_name} ===")
            print("Median PAWN indices:")
            for i, param_name in enumerate(problem['names']):
                print(f"  {param_name}: {Si['median'][i]:.4f}")
            
        except Exception as e:
            print(f"Error in PAWN analysis for {output_name}: {e}")
    
    return sensitivity_results, X, all_outputs

def plot_sensitivity_results(sensitivity_results, save_plots=True):
    """
    Create comprehensive plots for sensitivity analysis results
    """
    param_names = problem['names']
    
    # Create subplots for each output variable
    n_outputs = len(sensitivity_results)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    plot_idx = 0
    
    for output_name, Si in sensitivity_results.items():
        if plot_idx >= len(axes):
            break
            
        ax = axes[plot_idx]
        
        # Plot median PAWN indices
        median_indices = Si['median']
        bars = ax.bar(param_names, median_indices, alpha=0.7)
        
        # Color bars by sensitivity level
        for i, bar in enumerate(bars):
            if median_indices[i] > 0.3:
                bar.set_color('red')  # High sensitivity
            elif median_indices[i] > 0.1:
                bar.set_color('orange')  # Medium sensitivity
            else:
                bar.set_color('green')  # Low sensitivity
        
        ax.set_title(f'Parameter Sensitivity for {output_name}')
        ax.set_ylabel('PAWN Index')
        ax.set_xlabel('Parameters')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add significance threshold line
        ax.axhline(y=0.1, color='black', linestyle='--', alpha=0.5, 
                   label='Significance threshold')
        ax.legend()
        
        plot_idx += 1
    
    # Remove empty subplots
    for i in range(plot_idx, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('sensitivity_analysis_results.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return fig

def analyze_parameter_interactions(X, all_outputs):
    """
    Analyze interactions between parameters and outputs
    """
    param_names = problem['names']
    
    # Create correlation matrix
    results_df = pd.DataFrame(X, columns=param_names)
    
    for output_name, output_values in all_outputs.items():
        results_df[output_name] = output_values
    
    # Calculate correlations
    correlation_matrix = results_df.corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    plt.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation Coefficient')
    
    # Add labels
    labels = list(correlation_matrix.columns)
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.yticks(range(len(labels)), labels)
    
    # Add correlation values as text
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                    ha='center', va='center', 
                    color='white' if abs(correlation_matrix.iloc[i, j]) > 0.7 else 'black')
    
    plt.title('Parameter-Output Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return correlation_matrix

# Main execution
if __name__ == "__main__":
    print("=== Improved Sugarscape Sensitivity Analysis ===")
    print("Testing effects of:")
    print("1. Lambda parameter (logit noise)")
    print("2. Cooperation rate")
    print("3. Alpha variance (risk preference diversity)")
    print()
    
    # Run sensitivity analysis
    sensitivity_results, param_matrix, output_data = conduct_sensitivity_analysis(
        N=200,  # Reduced for faster execution
        num_replications=3
    )
    
    # Create visualizations
    plot_sensitivity_results(sensitivity_results)
    correlation_matrix = analyze_parameter_interactions(param_matrix, output_data)
    
    # Print summary
    print("\n=== SUMMARY ===")
    print("Key findings:")
    
    for output_name, Si in sensitivity_results.items():
        print(f"\nFor {output_name}:")
        max_sensitivity_idx = np.argmax(Si['median'])
        max_param = problem['names'][max_sensitivity_idx]
        max_value = Si['median'][max_sensitivity_idx]
        print(f"  Most influential parameter: {max_param} (PAWN index: {max_value:.4f})")
        
        # Factor fixing recommendations
        for i, param_name in enumerate(problem['names']):
            if Si['median'][i] < 0.05:
                print(f"  Consider fixing {param_name} (low sensitivity: {Si['median'][i]:.4f})")
    
    print("\nAnalysis complete. Results saved to PNG files.")