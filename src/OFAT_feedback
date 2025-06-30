import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from functools import reduce
import os

def merge_csv(column_name):
    path = 'output/feedback_sensitivity_results/'
    all_files = glob.glob(os.path.join(path, f"sugar_model_results_steps_100_feed_*.csv"))

    feed_groups = {}
    for file in all_files:
        match = re.search(r'feed_(-?[\d.]+)_mcindex_(\d+)', file)
        if match:
            feed = match.group(1)
            mc_index = match.group(2)
            feed_groups.setdefault(feed, []).append((file, mc_index))
            print(f"Found feed: {feed}, mc_index: {mc_index}")
    
    for feed, file_info in feed_groups.items():
        dfs = []
        for file, mc_index in file_info:
            df = pd.read_csv(file)
            df = df.reset_index().rename(columns={'index': 'timestep'})
            df = df[['timestep', column_name]].rename(columns={column_name: f'{column_name}_mc{mc_index}'})
            dfs.append(df)
        
        combined_feed = reduce(lambda left, right: pd.merge(left, right, on='timestep'), dfs)
        combined_feed.to_csv(os.path.join(path, f'sugarmodel_{column_name}_feed_{feed}.csv'), index=False)
        print(f"Saved combined csv file for feed={feed} with {len(file_info)} MC runs")

    print("Merge csv processing complete!")

def OFAT(T, column_name):
    path = 'output/feedback_sensitivity_results/'
    results = []
    
    files = glob.glob(os.path.join(path, f'sugarmodel_{column_name}_feed_*.csv'))
    
    for file in files:
        try:
            df = pd.read_csv(file)
            feed = file.split('feed_')[-1].split('.csv')[0]
            
            if 'timestep' not in df.columns:
                print("Warning: 'timestep' column not found in file")
                continue
                
            row = df[df['timestep'] == T]
            
            if not row.empty:
                gini_cols = [col for col in df.columns if col.startswith(column_name)]
                
                if not gini_cols:
                    print(f"Warning: No {column_name} columns found")
                    continue
                    
                gini_values = row[gini_cols].values[0]
                
                stats = {
                    'feed': feed,
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
            summary_df['feed_num'] = summary_df['feed'].str.replace('neg', '-').astype(float)
            summary_df = summary_df.sort_values('feed_num').drop('feed_num', axis=1)
        except Exception as e:
            print(f"feed sorting failed: {e}")
            summary_df = summary_df.sort_values('feed')
        
        output_file = os.path.join(path, f"OFAT_{column_name}_feed_T_{T}.csv")
        summary_df.to_csv(output_file, index=False)
        print(f"\nSuccessfully saved results to {output_file}")
    else:
        print("\nNo results were generated.")
    
    return summary_df if results else None
    
def plot(T, column_name, save=False): 
    path = 'output/feedback_sensitivity_results/'
    filename = f'OFAT_{column_name}_feed_T_{T}.csv'
    df = pd.read_csv(os.path.join(path, filename))
    
    plt.figure(figsize=(6,4))
    plt.scatter(df['feed'], df['mean'], color='green', marker='o', facecolors='none', label='Mean')
    plt.scatter(df['feed'], df['max'], color='blue', marker='x', label='Max')
    plt.scatter(df['feed'], df['min'], color='red', marker='^', facecolors='none', label='Min')
    plt.xlabel('sugar feedback')
    plt.ylabel(column_name)
    plt.legend()
    plt.tight_layout()
    if save: 
        plt.savefig(f'output/OFAT_feed_figures/{column_name}', dpi=300)
    else: 
        plt.show()

def plot_compare_agents(T, save=False): 
    path = 'output/feed_sensitivity_results/'
    filename_aver = f'OFAT_RiskAverseAvgSugar_feed_T_{T}.csv'
    filename_neut = f'OFAT_NeutralAvgSugar_feed_T_{T}.csv'
    filename_seek = f'OFAT_RiskSeekingAvgSugar_feed_T_{T}.csv'
    df_aver = pd.read_csv(path + filename_aver)
    df_neut = pd.read_csv(path + filename_neut)
    df_seek = pd.read_csv(path + filename_seek)
    
    plt.figure(figsize=(6,4))
    plt.plot(df_aver['feed'], df_aver['mean'], color='red', marker='o', label='Risk Averse')
    plt.plot(df_neut['feed'], df_neut['mean'], color='blue', marker='x', label='Risk Neural')
    plt.plot(df_seek['feed'], df_seek['mean'], color='black', marker='^', label='Risk Seeking')
    plt.legend()
    if save: 
        plt.savefig(f'output/OFAT_feed_figures/agents_compare', dpi=300)
    else: 
        plt.show()

if __name__ == '__main__': 
    T = 100
    for column_name in ['TotalSugar','AvgSugarLevel','SugarGini','RiskAverseAvgSugar','NeutralAvgSugar','RiskSeekingAvgSugar','RiskAverseCoopSugar','NeutralCoopSugar','RiskSeekingCoopSugar','RiskSeekingNonCoopSugar','RiskAverseNonCoopSugar','NeutralNonCoopSugar']:
        merge_csv(column_name)
        OFAT(T, column_name)
        plot(T, column_name, save=True)
    # plot_compare_agents(T)
