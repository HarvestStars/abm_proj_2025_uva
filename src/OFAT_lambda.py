import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def merge_csv_lambda(column_name): 
    lambda_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    base_path = os.path.join("output", "mixed_parameter_results")

    # Create output directory if it doesn't exist
    output_dir = "lambda_processed_data"
    os.makedirs(output_dir, exist_ok=True)

    for lambda_val in lambda_values:
        folder_name = os.path.join(base_path, f"lambda_{lambda_val}")
        print(f"\nProcessing folder: {folder_name}")
        
        # Initialize a dictionary to store data by timestep
        timestep_data = {}
        
        # Find all matching CSV files
        csv_files = sorted(glob.glob(os.path.join(folder_name, "test1_results_steps_1000_alpha_0_run_*.csv")))
        print(f"Found {len(csv_files)} files in {folder_name}")
        
        for file in csv_files:
            print(f"Processing file: {os.path.basename(file)}")
            
            try:
                df = pd.read_csv(file)
                
                if column_name in df.columns:
                    run_number = int(file.split('_run_')[-1].split('.')[0])
                    
                    # Add each timestep's data to our dictionary
                    for step, value in enumerate(df[column_name]):
                        if step not in timestep_data:
                            timestep_data[step] = {'Step': step}
                        timestep_data[step][f'MC{run_number}'] = value
                else:
                    print(f"Warning: {column_name} column not found in {os.path.basename(file)}")
                    
            except Exception as e:
                print(f"Error processing {os.path.basename(file)}: {str(e)}")
        
        # Convert dictionary to DataFrame
        if timestep_data:
            result_df = pd.DataFrame.from_dict(timestep_data, orient='index')
            
            # Ensure all MC columns (0-9) exist, filling missing ones with NaN
            for run in range(10):
                col_name = f'MC{run}'
                if col_name not in result_df.columns:
                    result_df[col_name] = np.nan
            
            # Reorder columns: Step first, then MC0-MC9
            columns = ['Step'] + [f'MC{i}' for i in range(10)]
            result_df = result_df[columns]
            
            # Sort by Step
            result_df = result_df.sort_values('Step')
            
            # Save to CSV
            output_file = os.path.join(output_dir, f"{column_name}_lambda_{lambda_val}.csv")
            result_df.to_csv(output_file, index=False)
            print(f"Saved {len(result_df)} timesteps to {output_file}")
        else:
            print(f"No data collected for lambda = {lambda_val}")

    print("\nProcessing complete!")

def OFAT_lambda(T, column_name): 
    lambda_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    base_path = "lambda_processed_data"
    output_dir = "lambda_statistics"
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for lambda_val in lambda_values:
        # Look for the processed CSV file for this lambda value
        csv_file = os.path.join(base_path, f"{column_name}_lambda_{lambda_val}.csv")
        
        if not os.path.exists(csv_file):
            print(f"Warning: File not found - {csv_file}")
            continue
            
        try:
            df = pd.read_csv(csv_file)
            
            # Verify the file contains the required data
            if len(df) <= T:
                print(f"Warning: Timestep {T} not available in {csv_file} (max step: {len(df)-1})")
                continue
                
            # Extract all MC run columns (MC0-MC9)
            mc_columns = [col for col in df.columns if col.startswith('MC')]
            if not mc_columns:
                print(f"Warning: No MC columns found in {csv_file}")
                continue
                
            # Get values at timestep T from all MC runs
            step_values = df.loc[T, mc_columns].values
            
            # Calculate statistics
            stats = {
                'lambda': lambda_val,
                'timestep': T,
                'mean': np.mean(step_values),
                'min': np.min(step_values),
                'max': np.max(step_values),
                'num_MC_runs': len(step_values),
                'std_dev': np.std(step_values)  # Added standard deviation
            }
            results.append(stats)
            
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
            continue
    
    if not results:
        print("No valid data found for any lambda value")
        return None
    
    # Create and format the results DataFrame
    stats_df = pd.DataFrame(results)
    stats_df = stats_df[['lambda', 'timestep', 'mean', 'min', 'max', 'std_dev', 'num_MC_runs']]
    
    # Save to CSV
    output_file = os.path.join(output_dir, f"OFAT_{column_name}_step_{T}.csv")
    stats_df.to_csv(output_file, index=False, float_format='%.3f')  # Format floating point numbers
    
    print(f"Successfully saved statistics to {output_file}")
    print(f"Analyzed {len(results)} lambda values at timestep {T}")

def plot(T, column_name): 
    path = r'./lambda_statistics/'
    filename = f'OFAT_{column_name}_step_{T}.csv'
    df = pd.read_csv(path + filename)
    
    plt.figure(figsize=(6,4))
    plt.scatter(df['lambda'], df['mean'], color='green', marker='o', facecolors='none', label='Mean')
    plt.scatter(df['lambda'], df['max'], color='blue', marker='x', label='Max')
    plt.scatter(df['lambda'], df['min'], color='red', marker='^', facecolors='none', label='Min')
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel(fr'$\lambda$')
    plt.ylabel(column_name)
    # plt.title('OFAT Sensitivity: Effect of lambda on Gini Coefficients')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__': 
    for column_name in ['TotalSugar', 'AvgSugarLevel', 'SugarGini', 'RiskAverseAvgSugar', 'NeutralAvgSugar', 'RiskSeekingAvgSugar']:
        merge_csv_lambda(column_name)
        OFAT_lambda(1000, column_name)
        plot(1000, column_name)