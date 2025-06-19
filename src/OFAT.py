import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
from functools import reduce

def merge_csv():
    path = r'../src/'
    all_files = glob.glob(path + "csv_result_alpha_*.csv")

    alpha_groups = {}
    for file in all_files:
        match = re.search(r'alpha_([0-9.]+)_test_', file)
        if match:
            alpha = match.group(1)
            alpha_groups.setdefault(alpha, []).append(file)

    for alpha, files in alpha_groups.items():
        dfs = []
        for i, file in enumerate(files):
            df = pd.read_csv(file, header=None, names=['time', f'MC_id_{i+1}'])
            dfs.append(df)
        
        combined_alpha = reduce(lambda left, right: pd.merge(left, right, on='time'), dfs)
        combined_alpha.to_csv(f'{path}csv_combined_result_alpha_{alpha}.csv', index=False)
        print(f"Saved combined file for alpha={alpha} with {len(files)} tests")

    print("Merge csv processing complete!")

def OFAT(T): 
    path = r'../src/'
    results = []

    for file in glob.glob(path + 'csv_combined_result_alpha_*.csv'):
        df = pd.read_csv(file)
        alpha = file.split('_')[-1].split('.csv')[0] 
        
        filtered_df = df[df['time'] > T]
        mc_cols = [col for col in df.columns if col != 'time']
        mc_means = filtered_df[mc_cols].mean(axis=0) 
        
        stats = {
            'alpha': alpha,
            'mean': mc_means.mean(),
            'min': mc_means.min(),
            'max': mc_means.max()
        }
        results.append(stats)

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(path + f'OFTA_alpha_T_{T}.csv', index=False)

    print("OFTA csv processing complete!")

def plot(T): 
    path = r'../src/'
    filename = f'OFTA_alpha_T_{T}.csv'
    df = pd.read_csv(path + filename)
    
    plt.figure(figsize=(6,4))
    plt.scatter(df['alpha'], df['mean'], color='green', marker='o', facecolors='none', label='Mean')
    plt.scatter(df['alpha'], df['max'], color='blue', marker='x', label='Max')
    plt.scatter(df['alpha'], df['min'], color='red', marker='^', facecolors='none', label='Min')
    plt.xlabel(fr'$\alpha$')
    plt.ylabel('Agents')
    # plt.title('OFAT Sensitivity: Effect of Alpha on Model Output')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__': 
    # # test
    # merge_csv()
    # OFAT(10)
    # plot(10)