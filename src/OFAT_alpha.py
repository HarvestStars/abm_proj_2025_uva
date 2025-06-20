import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from functools import reduce

def merge_csv():
    path = r'./output/alpha_sensitivity_results/'
    all_files = glob.glob(path + "sugar_model_results_alpha_*.csv")

    alpha_groups = {}
    for file in all_files:
        match = re.search(r'alpha_(-?[\d.]+)_mcindex_(\d+)', file)
        if match:
            alpha = match.group(1)
            mc_index = match.group(2)
            alpha_groups.setdefault(alpha, []).append((file, mc_index))
            print(f"Found alpha: {alpha}, mc_index: {mc_index}")

    for alpha, file_info in alpha_groups.items():
        dfs = []
        for file, mc_index in file_info:
            df = pd.read_csv(file)
            df = df.reset_index().rename(columns={'index': 'timestep'})
            df = df[['timestep', 'SugarGini']].rename(columns={'SugarGini': f'SugarGini_mc{mc_index}'})
            dfs.append(df)
        
        combined_alpha = reduce(lambda left, right: pd.merge(left, right, on='timestep'), dfs)
        combined_alpha.to_csv(f'{path}sugarmodel_gini_alpha_{alpha}.csv', index=False)
        print(f"Saved combined csv file for alpha={alpha} with {len(file_info)} MC runs")

    print("Merge csv processing complete!")

def OFAT(T):
    path = r'./output/alpha_sensitivity_results/'
    results = []
    
    files = glob.glob(path + 'sugarmodel_gini_alpha_*.csv')
    
    for file in files:
        try:
            df = pd.read_csv(file)
            alpha = file.split('alpha_')[-1].split('.csv')[0]
            
            if 'timestep' not in df.columns:
                print("Warning: 'timestep' column not found in file")
                continue
                
            row = df[df['timestep'] == T]
            
            if not row.empty:
                gini_cols = [col for col in df.columns if col.startswith('SugarGini')]
                
                if not gini_cols:
                    print("Warning: No SugarGini columns found")
                    continue
                    
                gini_values = row[gini_cols].values[0]
                
                stats = {
                    'alpha': alpha,
                    'timestep': T,
                    'mean': float(np.mean(gini_values)),
                    'min': float(np.min(gini_values)),
                    'max': float(np.max(gini_values)),
                    'num_MC_runs': len(gini_values)
                }
                results.append(stats)
                print(f"Added stats: {stats}")
            else:
                print(f"Warning: Timestep {T} not found in file")  
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
    
    if results:
        summary_df = pd.DataFrame(results)
        try:
            summary_df['alpha_num'] = summary_df['alpha'].str.replace('neg', '-').astype(float)
            summary_df = summary_df.sort_values('alpha_num').drop('alpha_num', axis=1)
        except:
            summary_df = summary_df.sort_values('alpha')
        
        output_file = f"{path}OFAT_alpha_T_{T}.csv"
        summary_df.to_csv(output_file, index=False)
        print(f"\nSuccessfully saved results to {output_file}")
    else:
        print("\nNo results were generated.")
    
    return summary_df if results else None
    
def plot(T): 
    path = r'./output/alpha_sensitivity_results/'
    filename = f'OFAT_alpha_T_{T}.csv'
    df = pd.read_csv(path + filename)
    
    plt.figure(figsize=(6,4))
    plt.scatter(df['alpha'], df['mean'], color='green', marker='o', facecolors='none', label='Mean')
    plt.scatter(df['alpha'], df['max'], color='blue', marker='x', label='Max')
    plt.scatter(df['alpha'], df['min'], color='red', marker='^', facecolors='none', label='Min')
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel(fr'$\alpha$')
    plt.ylabel('Gini Coefficients')
    # plt.title('OFAT Sensitivity: Effect of Alpha on Gini Coefficients')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__': 
    # # test
    # merge_csv()
    OFAT(100)
    plot(100)