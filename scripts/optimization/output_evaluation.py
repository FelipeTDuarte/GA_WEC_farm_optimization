#!/usr/bin/env python3

import os
import re
import csv
import numpy as np
import pandas as pd
import scipy.io

###########################################################################################################
def read_config():
    config = {}
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
    return config

#########################################################################################################
config = read_config()

# Sea States
sea_states_df = pd.read_csv(config['sea_states_file'])

# Hs and P background
sea_states_df['Weighted_Background'] = sea_states_df['Background'] * sea_states_df['COUNT']
sea_states_df['Weighted_P'] = sea_states_df['P'] * sea_states_df['COUNT']
total_weight = sea_states_df['COUNT'].sum()
weighted_avg_background = sea_states_df['Weighted_Background'].sum() / total_weight
weighted_avg_P = sea_states_df['Weighted_P'].sum() / total_weight

num_wecs= int(config['num_wecs'])
averages_list = []
pattern = r'st(\d{2})a(\d{2,3})\.swn'

with open('preview.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['sea_state', 'array', 'average_Hs', 'average_power','standard_deviation','coeff_variation','power_range','power_peak_to_mean','power_mean_to_minimum'])

for swn_file in os.listdir('.'):
        if swn_file.endswith('.swn'):
            match = re.match(pattern, swn_file)
            
            if match:
                swn_file_name = os.path.splitext(swn_file)[0]
                sea_state = int(match.group(1))
                array = int(match.group(2))
    
            # Load the output .mat file
            all_hsig_values = []
            for mat_name in os.listdir('.'):
                if mat_name.endswith('.mat'):                    
                    mat_file = scipy.io.loadmat(mat_name)
                    
                    hsig_key = next(key for key in mat_file.keys() if 'Hsig' in key) 
                    hsig_values = mat_file[hsig_key]
                    all_hsig_values.append(hsig_values)

            # Converter a lista para um array e calcular a média ignorando NaNs
            all_hsig_values = np.hstack([hsig_values.ravel() for hsig_values in all_hsig_values])
            average_Hs = np.nanmean(all_hsig_values)
            print('average_Hs =', average_Hs, 'm')

            # Initialize a list to store the power values
            power_values = []

            # Open the POWER_ABS.OUT file and read the last number of wecs lines
            with open('POWER_ABS.OUT', 'r') as power_file:
                lines = power_file.readlines()[-num_wecs:]
                #print(lines)

                for line in lines:
                    # Split the line by the '=' sign and take the second part
                    value_part = line.split('=')[1].strip()
                    
                    # Extract the numerical value and convert it to a float
                    value = float(value_part.split('W')[0].strip())
                    
                    power_values.append(value)
                
                # Calculate the average of the power values
                average_power = sum(power_values) / len(power_values)

                # Calculate the standard deviation of the power values
                std_dev = np.std(power_values)

                # Calculate the coefficient of variation
                cv = std_dev / average_power

                # Calculate the rate between the maximum and minimum power values
                P2min= (max(power_values) - min(power_values)) / average_power

                # calculate the rate between the maximum and average power values
                P2avg = (max(power_values) - average_power) / average_power

                # Calculate the rate between the minimum and average power values
                avg2min = (average_power - min(power_values)) / average_power              
            
            print ('average_power =', average_power, 'W')
            
            # Write the average value to the CSV file, along with the name of the .swn file (without extension)
            with open('preview.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([sea_state, array, average_Hs, average_power,std_dev,cv,P2min,P2avg,avg2min])