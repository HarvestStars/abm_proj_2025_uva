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
import concurrent.futures
from pathlib import Path
from tqdm import tqdm 


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

def run_single_simulation(lambda_param, alpha, i, steps):
    from sugar_model import SugarModel

    try:
        alpha_range = (alpha - 0.5, alpha + 0.5)
        model = SugarModel(
            num_agents=200,
            lambda_param=lambda_param,
            cooperation_rate=1.0,
            max_sugar_per_cell=10,
            consume_per_step=1,
            consume_proportion_mode=False,
            alpha_range=alpha_range,
            agent_type="balanced"
        )
        
        for step in range(steps):
            model.step()

        results = model.datacollector.get_model_vars_dataframe()
        base_output_dir = Path("output") / "mixed_parameter_results"
        lambda_dir = base_output_dir / f"lambda_{lambda_param}"
        lambda_dir.mkdir(parents=True, exist_ok=True)
        filename = lambda_dir / f"test1_results_steps_{steps}_alpha_{alpha}_run_{i}.csv"
        results.to_csv(filename)
        return f"[λ={lambda_param}, α={alpha}, run={i}] ✅ saved"
    except Exception as e:
        return f"[λ={lambda_param}, α={alpha}, run={i}] ❌ error: {e}"

def run_mixed_parameter_testing(steps=200, max_workers=4):  # 默认限制并行为4个
    print("Mixed Parameter Testing (Lambda + Alpha combinations)")
    print("=" * 60)

    MC_TEST_REPEAT = 10
    Parameters_lambda = list(range(1, 21))
    Parameters_alpha = [0]  # 或使用 [-2, -1, 0, 1, 2]

    param_combinations = [(l, a) for l in Parameters_lambda for a in Parameters_alpha]
    total_jobs = len(param_combinations) * MC_TEST_REPEAT

    print(f"Parameter combinations: {len(param_combinations)}")
    print(f"MC repetitions per combination: {MC_TEST_REPEAT}")
    print(f"Total runs: {total_jobs}")
    print(f"Running with max {max_workers} workers")

    # 构建任务列表
    jobs = [(lambda_param, alpha, i, steps)
            for lambda_param, alpha in param_combinations
            for i in range(MC_TEST_REPEAT)]

    # 并行执行任务，带有限制的 worker 数
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_single_simulation, *job) for job in jobs]
        for f in tqdm(concurrent.futures.as_completed(futures), total=total_jobs):
            print(f.result())

    print("✅ All simulations completed.")


def run_alpha_sensitivity_analysis(steps=200):
    """Option 2: Alpha sensitivity analysis (main research focus)"""
    print("Alpha Sensitivity Analysis (Your desired format)")
    print("="*60)
    
    # Parameters for alpha sensitivity
    ALPHA_VALUES = np.linspace(-20, 20, 21)  # 11 points for testing
    MC_RUNS_PER_ALPHA = 10                  # 5 runs for testing
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
                    agent_type="balanced"
                )
                
                # Run simulation
                for step in range(steps):
                    model.step()
                
                # Get results
                model_data = model.datacollector.get_model_vars_dataframe()
                final_row = model_data.iloc[-1]
                
                # Save in desired format
                filename = f"sugar_model_results_steps_{steps}_alpha_{alpha_center:.1f}_mcindex_{mc_run}.csv"
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

def run_full_experiment(steps=200):
    """Run the complete experimental suite"""
    print("SUGARSCAPE MODEL TESTING SUITE")
    print("=" * 60)
    
    # Test 1: Model dimensions
    print("\n1. Testing model dimensions...")
    test_model_dimensions()
    
    # Test 2: Mixed parameter testing
    print("\n2. Running mixed parameter testing...")
    run_mixed_parameter_testing(steps)
    
    # Test 3: Alpha sensitivity analysis
    print("\n3. Running alpha sensitivity analysis...")
    alpha_df = run_alpha_sensitivity_analysis(steps)
    
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
    # run_full_experiment()
    max_workers = max(os.cpu_count() - 4, 1)
    print(f"Cpu cores count: {os.cpu_count()}, Using {max_workers} parallel workers for mixed parameter testing.")
    run_mixed_parameter_testing(steps=1000, max_workers=max_workers)
