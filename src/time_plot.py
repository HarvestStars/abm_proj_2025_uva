import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob
import os

def compare_agents(lambda_value=10): 
    path_aver = f"output/lambda_processed_data/RiskAverseCoopSugar_lambda_{lambda_value}.csv"
    path_risk = f"output/lambda_processed_data/RiskSeekingCoopSugar_lambda_{lambda_value}.csv"
    path_neut = f"output/lambda_processed_data/NeutralCoopSugar_lambda_{lambda_value}.csv"

    df_aver = pd.read_csv(path_aver)
    df_risk = pd.read_csv(path_risk)
    df_neut = pd.read_csv(path_neut)

    columns = ['MC0','MC1','MC2','MC3','MC4','MC5','MC6','MC7','MC8','MC9']
    df_aver['mean'] = df_aver[columns].mean(axis=1)
    df_risk['mean'] = df_risk[columns].mean(axis=1)
    df_neut['mean'] = df_neut[columns].mean(axis=1)

    plt.plot(df_aver['Step'], df_aver['mean'], label='Average')
    plt.plot(df_risk['Step'], df_risk['mean'], label='Risk')
    plt.plot(df_neut['Step'], df_neut['mean'], label='Neutral')

    plt.xlabel('Timestep')
    plt.ylabel('Sugar')
    plt.legend()
    plt.grid()
    plt.show()

def merge_alpha_timeplot_csv(column_name, alpha_value):
    pattern = f"output/alpha_sensitivity_results/sugar_model_results_steps_1000_alpha_{alpha_value:.1f}_mcindex_*.csv"
    output_folder = "output/merged/alpha"
    os.makedirs(output_folder, exist_ok=True)

    print(f"Looking for files matching: {pattern}")
    file_list = sorted(glob.glob(pattern))

    if not file_list:
        raise FileNotFoundError(f"No CSV files found matching {pattern}")

    timestep = None
    data_list = []

    for idx, file in enumerate(file_list):
        df = pd.read_csv(file)
        if column_name not in df.columns:
            raise ValueError(f"{file} does not contain {column_name}")
        if idx == 0:
            timestep = df.iloc[:, 0]
        data_list.append(df[column_name].reset_index(drop=True))

    merged_df = pd.DataFrame(data_list).T
    merged_df.columns = [f"mc_{i+1}" for i in range(len(data_list))]
    merged_df.insert(0, "Timestep", timestep.reset_index(drop=True))
    merged_df['mean'] = merged_df.loc[:, "mc_1":f"mc_{len(data_list)}"].mean(axis=1)

    output_path = os.path.join(output_folder, f"{column_name}_merged_{alpha_value}.csv")
    merged_df.to_csv(output_path, index=False)

def merge_lambda_timeplot_csv(column_name, lambda_value):
    folder_path = f"output/mixed_parameter_results/lambda_{lambda_value}/"
    output_folder = "output/merged/lambda"
    os.makedirs(output_folder, exist_ok=True)

    file_list = sorted(glob.glob(f"{folder_path}/*.csv")) 

    timestep = None
    data_list = []

    for idx, file in enumerate(file_list):
        df = pd.read_csv(file)
        if column_name not in df.columns:
            raise ValueError(f"{file} does not contain {column_name}")
        if idx == 0:
            timestep = df.iloc[:, 0]
        data_list.append(df[column_name].reset_index(drop=True))

    merged_df = pd.DataFrame(data_list).T
    merged_df.columns = [f"mc_{i+1}" for i in range(len(data_list))]
    merged_df.insert(0, "Timestep", timestep.reset_index(drop=True))

    merged_df['mean'] = merged_df.loc[:, "mc_1":f"mc_{len(data_list)}"].mean(axis=1)

    output_path = os.path.join(output_folder, f"{column_name}_merged_{lambda_value}.csv")
    merged_df.to_csv(output_path, index=False)

    print(f"Saved file to {output_path} successfully! ")

def timeplot_lambda(column_name, save_name=None): 
    filepath_0 = f'output/merged/lambda/{column_name}_merged_0.0001.csv'
    filepath_1 = f'output/merged/lambda/{column_name}_merged_0.001.csv'
    filepath_2 = f'output/merged/lambda/{column_name}_merged_0.01.csv'
    filepath_3 = f'output/merged/lambda/{column_name}_merged_0.1.csv'
    filepath_4 = f'output/merged/lambda/{column_name}_merged_1.csv'
    filepath_5 = f'output/merged/lambda/{column_name}_merged_10.csv'
    filepath_6 = f'output/merged/lambda/{column_name}_merged_100.csv'

    df_0 = pd.read_csv(filepath_0)
    df_1 = pd.read_csv(filepath_1)
    df_2 = pd.read_csv(filepath_2)
    df_3 = pd.read_csv(filepath_3)
    df_4 = pd.read_csv(filepath_4)
    df_5 = pd.read_csv(filepath_5)
    df_6 = pd.read_csv(filepath_6)

    lambda_values = ['1e-4', '1e-3', '1e-2', '1e-1', '1', '1e1', '1e2']
    color_list = [
        'gold',         
        'darkorange',  
        'orangered',   
        'crimson',     
        'mediumvioletred', 
        'mediumblue',   
        'navy'         
    ]

    plt.figure(figsize=(8, 5))
    for idx, df in enumerate([df_0, df_1, df_2, df_3, df_4, df_5, df_6]): 
        min_idx = df['mean'].idxmin()
        min_timestep = df.loc[min_idx, 'Timestep']
        min_value = df.loc[min_idx, 'mean']

        max_idx = df['mean'].idxmax()
        max_timestep = df.loc[max_idx, 'Timestep']
        max_value = df.loc[max_idx, 'mean']
        
        plt.plot(df['Timestep'], df['mean'], marker='o', markersize=0.3, color=color_list[idx], label=fr'$\lambda$={lambda_values[idx]}')
        plt.xlabel('Timestep', fontsize=14)
        plt.ylabel(column_name, fontsize=14)
        # if column_name == 'TotalSugar' and idx == 2: 
        #     plt.axhline(y=min_value, color='red', linestyle='--')

        # elif column_name == 'SugarGini' and idx == 0:
        #     plt.axhline(y=max_value, color='red', linestyle='--')
    plt.grid(True)
    plt.legend()
    if save_name: 
        plt.savefig(f'output/timeplot/{save_name}')
    else: 
        plt.show()

def timeplot_alpha(column_name, save_name=None):
    alpha_values = list(np.linspace(-20, 20, 11))
    cmap = mpl.colormaps['viridis']
    color_list = [cmap(i) for i in np.linspace(0, 1, len(alpha_values))]
    alpha_labels = [f'{a:.1f}' for a in alpha_values]

    plt.figure(figsize=(8, 5))
    for idx, alpha in enumerate(alpha_values):
        filepath = f'output/merged/alpha/{column_name}_merged_{alpha:.0f}.csv'
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
        df = pd.read_csv(filepath)
        plt.plot(
            df['Timestep'], df['mean'],
            marker='o', markersize=0.3,
            color=color_list[idx],
            label=fr'$\alpha$={alpha_labels[idx]}'
        )

    plt.xlabel('Timestep', fontsize=14)
    plt.ylabel(column_name, fontsize=14)
    plt.legend()
    plt.grid()
    if save_name: 
        plt.savefig(f'output/timeplot/{save_name}')
    else: 
        plt.show()
    
def timeplot_cooperation(column_name, save_name=None): 
    '''draw time plot for different cooperation rate. '''
    filepath_1 = f'output/cooperation_sensitivity_results/sugarmodel_{column_name}_coop_0.1.csv'
    filepath_2 = f'output/cooperation_sensitivity_results/sugarmodel_{column_name}_coop_0.2.csv'
    filepath_3 = f'output/cooperation_sensitivity_results/sugarmodel_{column_name}_coop_0.3.csv'
    filepath_4 = f'output/cooperation_sensitivity_results/sugarmodel_{column_name}_coop_0.4.csv'
    filepath_5 = f'output/cooperation_sensitivity_results/sugarmodel_{column_name}_coop_0.5.csv'
    filepath_6 = f'output/cooperation_sensitivity_results/sugarmodel_{column_name}_coop_0.6.csv'
    filepath_7 = f'output/cooperation_sensitivity_results/sugarmodel_{column_name}_coop_0.7.csv'
    filepath_8 = f'output/cooperation_sensitivity_results/sugarmodel_{column_name}_coop_0.8.csv'
    filepath_9 = f'output/cooperation_sensitivity_results/sugarmodel_{column_name}_coop_0.9.csv'
    filepath_10 = f'output/cooperation_sensitivity_results/sugarmodel_{column_name}_coop_1.0.csv'

    df_1 = pd.read_csv(filepath_1)
    df_2 = pd.read_csv(filepath_2)
    df_3 = pd.read_csv(filepath_3)
    df_4 = pd.read_csv(filepath_4)
    df_5 = pd.read_csv(filepath_5)
    df_6 = pd.read_csv(filepath_6)
    df_7 = pd.read_csv(filepath_7)
    df_8 = pd.read_csv(filepath_8)
    df_9 = pd.read_csv(filepath_9)
    df_10 = pd.read_csv(filepath_10)

    columns = [f"{column_name}_mc{i}" for i in range(10)]
 
    mean_values_1 = df_1.loc[:, columns].mean(axis=1)
    df_1['mean'] = mean_values_1
    mean_values_2 = df_2.loc[:, columns].mean(axis=1)
    df_2['mean'] = mean_values_2
    mean_values_3 = df_3.loc[:, columns].mean(axis=1)
    df_3['mean'] = mean_values_3
    mean_values_4 = df_4.loc[:, columns].mean(axis=1)
    df_4['mean'] = mean_values_4
    mean_values_5 = df_5.loc[:, columns].mean(axis=1)
    df_5['mean'] = mean_values_5
    mean_values_6 = df_6.loc[:, columns].mean(axis=1)
    df_6['mean'] = mean_values_6
    mean_values_7 = df_7.loc[:, columns].mean(axis=1)
    df_7['mean'] = mean_values_7
    mean_values_8 = df_8.loc[:, columns].mean(axis=1)
    df_8['mean'] = mean_values_8
    mean_values_9 = df_9.loc[:, columns].mean(axis=1)
    df_9['mean'] = mean_values_9
    mean_values_10 = df_10.loc[:, columns].mean(axis=1)
    df_10['mean'] = mean_values_10

    cmap1 = plt.get_cmap('Blues')
    colors_1 = [cmap1(i) for i in np.linspace(0.3, 0.9, 6)]
    cmap2 = plt.get_cmap('Oranges')
    colors_2 = [cmap2(i) for i in np.linspace(0.3, 0.9, 4)]
    colors = colors_1 + colors_2

    plt.figure(figsize=(8, 5))
    plt.plot(df_1['timestep'], mean_values_1, marker='o', markersize=0.3, color=colors[0], label='cooperation rate=0.1')
    plt.plot(df_1['timestep'], mean_values_2, marker='o', markersize=0.3, color=colors[1], label='cooperation rate=0.2')
    plt.plot(df_1['timestep'], mean_values_3, marker='o', markersize=0.3, color=colors[2], label='cooperation rate=0.3')
    plt.plot(df_1['timestep'], mean_values_4, marker='o', markersize=0.3, color=colors[3], label='cooperation rate=0.4')
    plt.plot(df_1['timestep'], mean_values_5, marker='o', markersize=0.3, color=colors[4], label='cooperation rate=0.5')
    plt.plot(df_1['timestep'], mean_values_6, marker='o', markersize=0.3, color=colors[5], label='cooperation rate=0.6')
    plt.plot(df_1['timestep'], mean_values_7, marker='o', markersize=0.3, color=colors[6], label='cooperation rate=0.7')
    plt.plot(df_1['timestep'], mean_values_8, marker='o', markersize=0.3, color=colors[7], label='cooperation rate=0.8')
    plt.plot(df_1['timestep'], mean_values_9, marker='o', markersize=0.3, color=colors[8], label='cooperation rate=0.9')
    plt.plot(df_1['timestep'], mean_values_10, marker='o', markersize=0.3,color=colors[9], label='cooperation rate=1.0')

    plt.xlabel('Timestep', fontsize=14)
    plt.ylabel(column_name, fontsize=14)
    plt.grid()
    plt.legend()
    if save_name: 
        plt.savefig(f'output/timeplot/{save_name}')
    else: 
        plt.show()

def timeplot_feedback(column_name, save_name=None): 
    '''draw time plot for different sugar feedback value. '''
    filepath_1 = f'output/feedback_sensitivity_results/sugarmodel_{column_name}_feed_0.5.csv'
    filepath_2 = f'output/feedback_sensitivity_results/sugarmodel_{column_name}_feed_1.0.csv'
    filepath_3 = f'output/feedback_sensitivity_results/sugarmodel_{column_name}_feed_1.5.csv'
    filepath_4 = f'output/feedback_sensitivity_results/sugarmodel_{column_name}_feed_2.0.csv'
    filepath_5 = f'output/feedback_sensitivity_results/sugarmodel_{column_name}_feed_2.5.csv'
    filepath_6 = f'output/feedback_sensitivity_results/sugarmodel_{column_name}_feed_3.0.csv'
    filepath_7 = f'output/feedback_sensitivity_results/sugarmodel_{column_name}_feed_3.5.csv'
    filepath_8 = f'output/feedback_sensitivity_results/sugarmodel_{column_name}_feed_4.0.csv'

    df_1 = pd.read_csv(filepath_1)
    df_2 = pd.read_csv(filepath_2)
    df_3 = pd.read_csv(filepath_3)
    df_4 = pd.read_csv(filepath_4)
    df_5 = pd.read_csv(filepath_5)
    df_6 = pd.read_csv(filepath_6)
    df_7 = pd.read_csv(filepath_7)
    df_8 = pd.read_csv(filepath_8)

    columns = [f"{column_name}_mc{i}" for i in range(10)] 

    mean_values_1 = df_1.loc[:, columns].mean(axis=1)
    df_1['mean'] = mean_values_1
    mean_values_2 = df_2.loc[:, columns].mean(axis=1)
    df_2['mean'] = mean_values_2
    mean_values_3 = df_3.loc[:, columns].mean(axis=1)
    df_3['mean'] = mean_values_3
    mean_values_4 = df_4.loc[:, columns].mean(axis=1)
    df_4['mean'] = mean_values_4
    mean_values_5 = df_5.loc[:, columns].mean(axis=1)
    df_5['mean'] = mean_values_5
    mean_values_6 = df_6.loc[:, columns].mean(axis=1)
    df_6['mean'] = mean_values_6
    mean_values_7 = df_7.loc[:, columns].mean(axis=1)
    df_7['mean'] = mean_values_7
    mean_values_8 = df_8.loc[:, columns].mean(axis=1)
    df_8['mean'] = mean_values_8

    cmap1 = plt.get_cmap('Blues')
    colors_1 = [cmap1(i) for i in np.linspace(0.3, 0.9, 3)]
    cmap2 = plt.get_cmap('Oranges')
    colors_2 = [cmap2(i) for i in np.linspace(0.3, 0.9, 5)]
    colors = colors_1 + colors_2

    plt.figure(figsize=(8, 5))
    plt.plot(df_1['timestep'], mean_values_1, marker='o', markersize=0.3, color=colors[0], label='sugar feedback=0.5')
    plt.plot(df_1['timestep'], mean_values_2, marker='o', markersize=0.3, color=colors[1], label='sugar feedback=1.0')
    plt.plot(df_1['timestep'], mean_values_3, marker='o', markersize=0.3, color=colors[2], label='sugar feedback=1.5')
    plt.plot(df_1['timestep'], mean_values_4, marker='o', markersize=0.3, color=colors[3], label='sugar feedback=2.0')
    plt.plot(df_1['timestep'], mean_values_5, marker='o', markersize=0.3, color=colors[4], label='sugar feedback=2.5')
    plt.plot(df_1['timestep'], mean_values_6, marker='o', markersize=0.3, color=colors[5], label='sugar feedback=3.0')
    plt.plot(df_1['timestep'], mean_values_7, marker='o', markersize=0.3, color=colors[6], label='sugar feedback=3.5')
    plt.plot(df_1['timestep'], mean_values_8, marker='o', markersize=0.3, color=colors[7], label='sugar feedback=4.0')
    
    plt.xlabel('Timestep', fontsize=14)
    plt.ylabel(column_name, fontsize=14)
    plt.grid()
    plt.legend()
    if save_name: 
        plt.savefig(f'output/timeplot/{save_name}')
    else: 
        plt.show()

if __name__ == '__main__': 
    lambda_value = 100
    for column_name in ['TotalSugar', 'SugarGini']: 
        # merge_lambda_timeplot_csv(column_name, lambda_value)
        timeplot_alpha(column_name, save_name=f'timeplot_alpha_{column_name}')
        timeplot_lambda(column_name, save_name=f'timeplot_lambda_{column_name}')
        # timeplot_cooperation(column_name, save_name=f'timeplot_coop_{column_name}')
        # timeplot_feedback(column_name, save_name=f'timeplot_feed_{column_name}')
