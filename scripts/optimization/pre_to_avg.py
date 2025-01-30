#!/usr/bin/env python3

import os
import pandas as pd


########################

def read_config():
    
    
    config = {}
    # Find the .ga file in the parent directory
    os.chdir(parent_dir)
    for filename in os.listdir():
        if filename.endswith('.ga'):
            with open(filename, 'r') as file:
                for line in file:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue  # Skip empty lines and comments
                    if '=' in line:
                        key, value = map(str.strip, line.split('=', 1))
                        config[key] = value
            break  # Stop searching after finding the first .ga file
    os.chdir(current_dir)
    return config

def preview_outputs(preview, sea_states_df):
        HRAs = []
        average_Hsigs = []
        average_powers = []
        Qfactors = []
        accessibility_increases = []
        arrays = []

        for array in preview['array'].unique():
                preview_array = preview[preview['array'] == array].copy().reset_index(drop=True)

                sea_states_df_best_preview_merged = sea_states_df.merge(preview_array, left_on='ST', right_on='sea_state')
                sea_states_df_best_preview_merged['hs_diff'] = sea_states_df_best_preview_merged['Background'] - sea_states_df_best_preview_merged['average_Hs']
                sea_states_df_best_preview_merged['HRA'] = 100* (sea_states_df_best_preview_merged['hs_diff'] / sea_states_df_best_preview_merged['Background'])

                HRA_cluster = sea_states_df_best_preview_merged.drop(columns=['ST','HS','TP', 'DIR', 'P', 'P_TOT', 'SHARE', 'Background', 'sea_state', 'array'])
                HRA_cluster = HRA_cluster.merge(hs_cluster_merged, on='Cluster')
                HRA_cluster['New_Hs'] = HRA_cluster['Hs'] * ((100-HRA_cluster['HRA']) / 100)

                COUNT_df = HRA_cluster.groupby('Cluster')[['Hs', 'New_Hs']].apply(lambda x: (x >= 1.5).sum())
                COUNT_df.rename(columns={'Hs': 'COUNT_Hs_ge_1.5', 'New_Hs': 'COUNT_New_Hs_ge_1.5'}, inplace=True)
                COUNT_df['Time_gain'] = 100*(COUNT_df['COUNT_Hs_ge_1.5'] - COUNT_df['COUNT_New_Hs_ge_1.5']) / COUNT_df['COUNT_Hs_ge_1.5']

                HRA_cluster = pd.merge(HRA_cluster, COUNT_df, on='Cluster')
                HRA_cluster['access_before (h)'] = 365*24*(HRA_cluster['COUNT'] - HRA_cluster['COUNT_Hs_ge_1.5']) / len_data
                HRA_cluster['access_after (h)'] = 365*24*(HRA_cluster['COUNT'] - HRA_cluster['COUNT_New_Hs_ge_1.5']) / len_data
                HRA_cluster['access_increase'] = (HRA_cluster['access_after (h)'] - HRA_cluster['access_before (h)']) / HRA_cluster['access_before (h)']
                HRA_cluster['access_increase'] = HRA_cluster['access_increase'].fillna(0)


                HRA = (HRA_cluster.groupby('Cluster')['HRA'].mean()*HRA_cluster.groupby('Cluster')['COUNT'].mean()).sum()/(HRA_cluster.groupby('Cluster')['COUNT'].mean().sum())
                average_Hsig = (HRA_cluster.groupby('Cluster')['average_Hs'].mean()*HRA_cluster.groupby('Cluster')['COUNT'].mean()).sum()/(HRA_cluster.groupby('Cluster')['COUNT'].mean().sum())
                average_power = (HRA_cluster.groupby('Cluster')['average_power'].mean()*HRA_cluster.groupby('Cluster')['COUNT'].mean()).sum()/(HRA_cluster.groupby('Cluster')['COUNT'].mean().sum())
                standard_deviation = (HRA_cluster.groupby('Cluster')['standard_deviation'].mean()*HRA_cluster.groupby('Cluster')['COUNT'].mean()).sum()/(HRA_cluster.groupby('Cluster')['COUNT'].mean().sum())
                coeff_variation = (HRA_cluster.groupby('Cluster')['coeff_variation'].mean()*HRA_cluster.groupby('Cluster')['COUNT'].mean()).sum()/(HRA_cluster.groupby('Cluster')['COUNT'].mean().sum())
                power_range = (HRA_cluster.groupby('Cluster')['power_range'].mean()*HRA_cluster.groupby('Cluster')['COUNT'].mean()).sum()/(HRA_cluster.groupby('Cluster')['COUNT'].mean().sum())
                peak_to_mean = (HRA_cluster.groupby('Cluster')['power_peak_to_mean'].mean()*HRA_cluster.groupby('Cluster')['COUNT'].mean()).sum()/(HRA_cluster.groupby('Cluster')['COUNT'].mean().sum())
                mean_to_minimum = (HRA_cluster.groupby('Cluster')['power_mean_to_minimum'].mean()*HRA_cluster.groupby('Cluster')['COUNT'].mean()).sum()/(HRA_cluster.groupby('Cluster')['COUNT'].mean().sum())



                Qfactor = average_power/single_WEC

                access_before= HRA_cluster.groupby('Cluster')['access_before (h)'].mean().sum()
                access_after = HRA_cluster.groupby('Cluster')['access_after (h)'].mean().sum()
                accessibility_increase = 100*(access_after-access_before)/access_before

                acess_data = {'': ['access_before', 'access_after','hours_increase', 'acessibility_increase'], 
                        'hour/year': [access_before, access_after,access_after-access_before,365*24*accessibility_increase/100], 
                        'percentage': [100*access_before/(365*24), 100*access_after/(365*24),100*(access_after-access_before)/(365*24),accessibility_increase]}
                accessibility = pd.DataFrame(acess_data).round(2)


                HRAs.append(HRA)
                average_Hsigs.append(average_Hsig)
                average_powers.append(average_power)
                Qfactors.append(Qfactor)
                accessibility_increases.append(accessibility_increase)
                arrays.append(array)

                '''print('array',array)
                print('HRA = ',HRA)
                print('average power = ',average_power)
                print('Q-factor = ',Qfactor)
                print('accessibility increase = ',accessibility_increase)'''

        results = pd.DataFrame({'array': arrays, 'HRA': HRAs, 'average_Hs': average_Hsigs,'average_power': average_powers, 'Q-factor': Qfactors, 'accessibility_increase': accessibility_increases, 
                               'standard_deviation': standard_deviation, 'coeff_variation': coeff_variation, 'power_range':power_range, 'power_peak_to_mean': peak_to_mean, 
                               'power_mean_to_minimum':mean_to_minimum})

        return results

##########################
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
config = read_config()

print('reading dataset')
complete_dataset = pd.read_csv(os.path.join(parent_dir,config['complete_dataset']))
len_data = len(complete_dataset)

# Sea States
print('reading sea states')
sea_states_df = pd.read_csv(config['sea_states_file'])
sea_states_df['Cluster'] = sea_states_df['ST']

print('reading cluster data')
cluster_data = pd.read_csv(os.path.join(parent_dir,config['cluster_data']))
hs_cluster_merged = sea_states_df.merge(cluster_data, on='Cluster')
hs_cluster_merged.drop(columns=[ 'Tp', 'Wd', 'P_x', 'ST', 'HS', 'TP', 'DIR', 'P_y', 'COUNT','P_TOT','SHARE','Background'], inplace=True)

print('reading preview')
preview = pd.read_csv('preview.csv')

single_WEC = float(config['single_WEC']) 

print('preview outputs')
weighted_array_avg_df = preview_outputs(preview, sea_states_df)

print('saving weighted_array_avg_df')
# Save the DataFrame to a CSV file named weighted_array_avg.csv, however if it already exists, append the data to it     
if os.path.exists('weighted_array_avg.csv'):
    weighted_array_avg_df.to_csv('weighted_array_avg.csv', mode='a', header=False, index=False)
else:
    weighted_array_avg_df.to_csv('weighted_array_avg.csv', index=False)