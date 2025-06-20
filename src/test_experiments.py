"""
Test experiments for Sugarscape model
Separated from model.py for cleaner code organization
"""
import sys
import os
sys.path.append(os.path.abspath("."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sugar_model import SugarModel

def test_model_dimensions():
    """Test that model uses correct dimensions from sugar map"""
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

def run_mixed_parameter_testing():
    """Option 1: Mixed parameter testing (lambda + alpha combinations)"""
    print("Mixed Parameter Testing (Lambda + Alpha combinations)")
    print("=" * 60)
    
    MC_TEST_REPEAT = 3
    Parameters_lambda = [1, 2, 3, 4, 1000]
    Parameters_alpha = [-2, -1, 0, 1, 2]
    
    # Create Cartesian product
    param_combinations = [(l, a) for l in Parameters_lambda for a in Parameters_alpha]
    
    print(f"Parameter combinations: {len(param_combinations)}")
    print(f"MC repetitions per combination: {MC_TEST_REPEAT}")
    print(f"Total runs: {len(param_combinations) * MC_TEST_REPEAT}")

    # Create base output directory
    base_output_dir = Path("output") / "mixed_parameter_results"
    base_output_dir.mkdir(parents=True, exist_ok=True)

    for lambda_param, alpha in param_combinations:
        print(f"\nTesting λ={lambda_param}, α={alpha}")
        
        # Create directory for this lambda value
        lambda_dir = base_output_dir / f"lambda_{lambda_param}"
        lambda_dir.mkdir(exist_ok=True)

        for i in range(MC_TEST_REPEAT):
            try:
                # Use the parameters
                alpha_range = (alpha - 0.5, alpha + 0.5)
                
                model = SugarModel(
                    num_agents=100,
                    lambda_param=lambda_param,
                    cooperation_rate=0.3,
                    alpha_range=alpha_range,
                    agent_type="mixed"
                )
                
                print(f"  Run {i}: Grid {model.grid.width}x{model.grid.height}")
                
                # Run simulation
                for step in range(100):
                    model.step()
                
                # Save results
                results = model.datacollector.get_model_vars_dataframe()
                filename = f"test1_results_alpha_{alpha}_run_{i}.csv"
                results.to_csv(lambda_dir / filename)
                
                print(f"    Saved: {filename}")
                
            except Exception as e:
                print(f"    Error in run {i}: {e}")
                continue

def run_alpha_sensitivity_analysis():
    """Option 2: Alpha sensitivity analysis (main research focus)"""
    print("Alpha Sensitivity Analysis (Your desired format)")
    print("="*60)
    
    # Parameters for alpha sensitivity
    ALPHA_VALUES = np.linspace(-2, 2, 11)  # 11 points for testing
    MC_RUNS_PER_ALPHA = 5                  # 5 runs for testing
    FIXED_LAMBDA = 10
    FIXED_COOPERATION = 0.3
    
    print(f"Alpha values: {len(ALPHA_VALUES)} points from {ALPHA_VALUES[0]} to {ALPHA_VALUES[-1]}")
    print(f"MC runs per alpha: {MC_RUNS_PER_ALPHA}")
    print(f"Fixed lambda: {FIXED_LAMBDA}")
    print(f"Total runs: {len(ALPHA_VALUES) * MC_RUNS_PER_ALPHA}")
    
    # Create output directory
    alpha_output_dir = Path("output") / "alpha_sensitivity_results"
    alpha_output_dir.mkdir(parents=True, exist_ok=True)
    
    all_alpha_results = []
    
    for alpha_center in ALPHA_VALUES:
        print(f"\nTesting alpha = {alpha_center:.2f}")
        
        for mc_run in range(MC_RUNS_PER_ALPHA):
            try:
                # Create alpha range around center
                alpha_range = (alpha_center - 0.2, alpha_center + 0.2)
                
                # Create model with fixed lambda, varying alpha
                model = SugarModel(
                    num_agents=100,
                    lambda_param=FIXED_LAMBDA,
                    cooperation_rate=FIXED_COOPERATION,
                    alpha_range=alpha_range,
                    agent_type="mixed"
                )
                
                # Run simulation
                for step in range(200):
                    model.step()
                
                # Get results
                model_data = model.datacollector.get_model_vars_dataframe()
                final_row = model_data.iloc[-1]
                
                # Save in desired format
                filename = f"sugar_model_results_alpha_{alpha_center:.1f}_mcindex_{mc_run}.csv"
                model_data.to_csv(alpha_output_dir / filename)
                
                # Store summary
                result = {
                    'alpha': alpha_center,
                    'mc_run': mc_run,
                    'final_total_sugar': final_row['TotalSugar'],
                    'final_gini': final_row['SugarGini'],
                    'lambda_param': FIXED_LAMBDA,
                    'cooperation_rate': FIXED_COOPERATION
                }
                all_alpha_results.append(result)
                
                print(f"  Run {mc_run}: Final sugar = {final_row['TotalSugar']:.0f}")
                
            except Exception as e:
                print(f"  Error in MC run {mc_run}: {e}")
                continue
    
    # Save comprehensive results
    alpha_df = pd.DataFrame(all_alpha_results)
    alpha_df.to_csv(alpha_output_dir / "comprehensive_alpha_results.csv", index=False)
    
    # Create plot
    if len(alpha_df) > 0:
        create_alpha_plot(alpha_df, alpha_output_dir)
        
        # Print summary statistics
        print_alpha_summary(alpha_df)
    
    return alpha_df

def create_alpha_plot(alpha_df, output_dir):
    """Create alpha sensitivity plot"""
    print("Creating alpha sensitivity plot...")
    
    # Calculate statistics by alpha
    alpha_stats = alpha_df.groupby('alpha')['final_total_sugar'].agg([
        'mean', 'std', 'count'
    ]).reset_index()
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot individual points
    for alpha in alpha_df['alpha'].unique():
        alpha_data = alpha_df[alpha_df['alpha'] == alpha]
        if len(alpha_data) > 0:
            plt.scatter([alpha] * len(alpha_data), alpha_data['final_total_sugar'], 
                       alpha=0.6, s=30, color='lightblue')
    
    # Plot mean with error bars
    plt.errorbar(alpha_stats['alpha'], alpha_stats['mean'], 
                yerr=alpha_stats['std'], fmt='ro-', capsize=5, 
                linewidth=2, markersize=6, label='Mean ± Std')
    
    # Add risk-neutral line
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.7, 
               label='Risk Neutral (α=0)')
    
    plt.xlabel('Alpha (Risk Parameter)')
    plt.ylabel('Final Total Sugar')
    plt.title('Alpha Sensitivity Analysis')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save plot
    plot_path = output_dir / "alpha_sensitivity_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to: {plot_path}")

def print_alpha_summary(alpha_df):
    """Print summary statistics for alpha analysis"""
    alpha_stats = alpha_df.groupby('alpha')['final_total_sugar'].agg([
        'mean', 'std', 'count'
    ]).reset_index()
    
    print("ALPHA SENSITIVITY SUMMARY:")
    print("Alpha\tMean Sugar\tStd Sugar\tRuns")
    print("-" * 40)
    for _, row in alpha_stats.iterrows():
        print(f"{row['alpha']:5.1f}\t{row['mean']:8.1f}\t{row['std']:7.1f}\t{row['count']:4.0f}")

def run_full_experiment():
    """Run the complete experimental suite"""
    print("SUGARSCAPE MODEL TESTING SUITE")
    print("=" * 60)
    
    # Test 1: Model dimensions
    print("\n1. Testing model dimensions...")
    test_model_dimensions()
    
    # Test 2: Mixed parameter testing
    print("\n2. Running mixed parameter testing...")
    run_mixed_parameter_testing()
    
    # Test 3: Alpha sensitivity analysis
    print("\n3. Running alpha sensitivity analysis...")
    alpha_df = run_alpha_sensitivity_analysis()
    
    print("\nTESTING COMPLETE!")
    print("All files saved to output/ folder")
    print("\nFolder structure:")
    print("output/")
    print("├── mixed_parameter_results/")
    print("└── alpha_sensitivity_results/")
    
    print("\nRecommendations:")
    print("- Option 1: Tests multiple parameter combinations")
    print("- Option 2: Alpha sensitivity (matches your research goals)")
    print("- For full experiment: Set MC_RUNS_PER_ALPHA=100, ALPHA_VALUES=21 points")

if __name__ == "__main__":
    run_full_experiment()