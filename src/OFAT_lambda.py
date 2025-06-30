import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def merge_csv_lambda(column_name): 
    lambda_values = list(range(1, 21))
    base_path = 'output/mixed_parameter_results/'

    output_dir = "output/lambda_processed_data"
    os.makedirs(output_dir, exist_ok=True)

    for lambda_val in lambda_values:
        folder_name = os.path.join(base_path, f"lambda_{lambda_val}")
        print(f"\nProcessing folder: {folder_name}")
        
        timestep_data = {}
        csv_files = sorted(glob.glob(os.path.join(folder_name, f"test1_results_steps_100_alpha_0_run_*.csv")))
        print(f"Found {len(csv_files)} files in {folder_name}")
        
        for file in csv_files:
            print(f"Processing file: {os.path.basename(file)}")
            
            try:
                df = pd.read_csv(file)
                
                if column_name in df.columns:
                    run_number = int(file.split('_run_')[-1].split('.')[0])
                    
                    for step, value in enumerate(df[column_name]):
                        if step not in timestep_data:
                            timestep_data[step] = {'Step': step}
                        timestep_data[step][f'MC{run_number}'] = value
                else:
                    print(f"Warning: {column_name} column not found in {os.path.basename(file)}")
                    
            except Exception as e:
                print(f"Error processing {os.path.basename(file)}: {str(e)}")
        
        if timestep_data:
            result_df = pd.DataFrame.from_dict(timestep_data, orient='index')
            
            for run in range(10):
                col_name = f'MC{run}'
                if col_name not in result_df.columns:
                    result_df[col_name] = np.nan
            
            columns = ['Step'] + [f'MC{i}' for i in range(10)]
            result_df = result_df[columns]
            
            result_df = result_df.sort_values('Step')
            
            # Save to CSV
            output_file = os.path.join(output_dir, f"{column_name}_lambda_{lambda_val}.csv")
            result_df.to_csv(output_file, index=False)
            print(f"Saved {len(result_df)} timesteps to {output_file}")
        else:
            print(f"No data collected for lambda = {lambda_val}")

    print("\nProcessing complete!")

def OFAT_lambda(T, column_name): 
    lambda_values = list(range(1, 21))
    base_path = "output/lambda_processed_data"
    output_dir = "output/lambda_statistics"
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for lambda_val in lambda_values:
        csv_file = os.path.join(base_path, f"{column_name}_lambda_{lambda_val}.csv")
        
        if not os.path.exists(csv_file):
            print(f"Warning: File not found - {csv_file}")
            continue
            
        try:
            df = pd.read_csv(csv_file)
            
            if len(df) <= T:
                print(f"Warning: Timestep {T} not available in {csv_file} (max step: {len(df)-1})")
                continue
                
            mc_columns = [col for col in df.columns if col.startswith('MC')]
            if not mc_columns:
                print(f"Warning: No MC columns found in {csv_file}")
                continue
                
            step_values = df.loc[T, mc_columns].values
            
            stats = {
                'lambda': lambda_val,
                'timestep': T,
                'mean': np.mean(step_values),
                'min': np.min(step_values),
                'max': np.max(step_values),
                'num_MC_runs': len(step_values),
                'std_dev': np.std(step_values)  
            }
            results.append(stats)
            
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
            continue
    
    if not results:
        print("No valid data found for any lambda value")
        return None
    
    stats_df = pd.DataFrame(results)
    stats_df = stats_df[['lambda', 'timestep', 'mean', 'min', 'max', 'std_dev', 'num_MC_runs']]
    
    output_file = os.path.join(output_dir, f"OFAT_{column_name}_step_{T}.csv")
    stats_df.to_csv(output_file, index=False, float_format='%.3f')  # Format floating point numbers
    
    print(f"Successfully saved statistics to {output_file}")
    print(f"Analyzed {len(results)} lambda values at timestep {T}")

def plot(T, column_name, loglog=False, save=False): 
    path = 'output/lambda_statistics/'
    filename = f'OFAT_{column_name}_step_{T}.csv'
    df = pd.read_csv(path + filename)
    
    plt.figure(figsize=(6,4))
    plt.scatter(df['lambda'], df['mean'], color='green', marker='o', facecolors='none', label='Mean')
    plt.scatter(df['lambda'], df['max'], color='blue', marker='x', label='Max')
    plt.scatter(df['lambda'], df['min'], color='red', marker='^', facecolors='none', label='Min')
    # plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel(fr'$\lambda$')
    plt.ylabel(column_name)
    # plt.title('OFAT Sensitivity: Effect of lambda on Gini Coefficients')
    plt.legend()
    plt.tight_layout()
    log_title = ''

    if loglog == True: 
        fit = powerlaw.Fit(df['mean'])

        D = fit.power_law.D
        p_value = fit.power_law.KS()

        print("KS statistic (D):", D)
        print("Goodness-of-fit p-value:", p_value)

        plt.xscale('log')
        plt.yscale('log')
        log_title = '_log'

    if save: 
        plt.savefig(f'output/OFAT_lambda_figures/{column_name}{log_title}', dpi=300)
    else: 
        plt.show()

def plot_compare_agents(T, save=False): 
    path = r'output/lambda_statistics/'
    filename_aver = f'OFAT_RiskAverseAvgSugar_step_{T}.csv'
    filename_neut = f'OFAT_NeutralAvgSugar_step_{T}.csv'
    filename_seek = f'OFAT_RiskSeekingAvgSugar_step_{T}.csv'
    df_aver = pd.read_csv(path + filename_aver)
    df_neut = pd.read_csv(path + filename_neut)
    df_seek = pd.read_csv(path + filename_seek)
    
    plt.figure(figsize=(6,4))
    plt.plot(df_aver['lambda'], df_aver['mean'], color='red', marker='o', label='Risk Averse')
    plt.plot(df_neut['lambda'], df_neut['mean'], color='blue', marker='x', label='Risk Neural')
    plt.plot(df_seek['lambda'], df_seek['mean'], color='black', marker='^', label='Risk Seeking')
    plt.legend()
    
    if save: 
        plt.savefig(f'output/OFAT_lambda_figures/agents_compare', dpi=300)
    else: 
        plt.show()

if __name__ == '__main__': 
    T = 100
    for column_name in ['TotalSugar','AvgSugarLevel','SugarGini','RiskAverseAvgSugar','NeutralAvgSugar','RiskSeekingAvgSugar','RiskAverseCoopSugar','NeutralCoopSugar','RiskSeekingCoopSugar','RiskSeekingNonCoopSugar','RiskAverseNonCoopSugar','NeutralNonCoopSugar']:
        # merge_csv_lambda(column_name)
        # OFAT_lambda(T, column_name)
        plot(T, column_name, save=True)
    # plot(T, 'TotalSugar', loglog=True, save=True)
    # plot_compare_agents(T)