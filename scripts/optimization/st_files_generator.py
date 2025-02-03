import os
import pandas as pd
import shutil
import subprocess
import numpy as np
import scipy.io

############################################## functions #############################################
def read_config():
    config = {}
    # Find the .ga file in the current directory
    for filename in os.listdir():
        if filename.endswith('.ga'):
            with open(filename, 'r') as file:
                for line in file:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue  # Skip empty lines and comments
                    if '=' in line:
                        key, value = map(str.strip, line.split('=', 1))
                        config[key] = None if value == 'None' else value
            break  # Stop searching after finding the first .ga5 file
    return config
############################################## Sea States and background #############################################
config = read_config()

sim_type = config['sim_type']
sim_parallel = config['sim_parallel']
swanrun_script = config['swanrun_script']
background_file = config['background_file']
AOI_output = config['AOI_output']

# Create sea states directory 
output_dir = config['sea_states_folder']
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


# Read sea states
sea_states_df = pd.read_csv(config['sea_states_file'])

for index, row in sea_states_df.iterrows():
    # Extract sea state parameters
    ST = row['ST']
    HS = row['HS']
    TP = row['TP']
    DIR = row['DIR']

    # Create the st*.swn filename (e.g., st00.swn, st01.swn, ...)
    filename = os.path.join(output_dir, f'st{index:02d}.swn')

    # Read the original st.swn file
    with open(background_file, 'r') as original_file:
        content = original_file.read()

    # Replace placeholders with sea state data
    content = content.replace('HSbs', str(HS))
    content = content.replace('TPbs', str(TP))
    content = content.replace('DIRbs', str(DIR))

    # Write the modified content to the new st*.swn file
    with open(filename, 'w') as new_file:
        new_file.write(content)

    # Copy input files from the main folder to the newly created folder.
    files_to_copy = [config['bottom_file'], config['grid_file'], config['snl_power_file'], config['sea_states_file'], config['swanrun_script']]

    for file_name in files_to_copy:
        source_file = os.path.join('./', file_name)  # Full path to the source file
        destination_file = os.path.join(output_dir, file_name)  # Full path to the destination file

        shutil.copy(source_file, destination_file)

# Change the current working directory to the sea states folder
os.chdir(output_dir)

# Loop through all of the .swn files
for swn_file in os.listdir('.'):
    if swn_file.endswith('.swn'):
        print('\n running', swn_file, '\n')
        
        # Remove the file extension from the swn_file
        swn_file_name = os.path.splitext(swn_file)[0]
        sea_state = int(swn_file_name[2:4]) 
        
        
        # Construct the command
        command = [f'./{swanrun_script}', '-input', swn_file, f'-{sim_type}', f'{sim_parallel}']

        subprocess.run(command)

        # Load the output .mat file
        mat_file = scipy.io.loadmat(AOI_output)

        # Get the average of all values of the matrix in the .mat file
        hsig_key = next((key for key in mat_file.keys() if key.startswith("Hsig")), None)

        # Compute the mean if the key exists
        if hsig_key:
            average_background = np.nanmean(mat_file[hsig_key])
        else:
            average_background = None

        print('Hs_background =', average_background)

        # Update the DataFrame with the average_background for the current sea state
        row_index = sea_states_df.index[sea_states_df['ST'] == sea_state].tolist()[0]
        sea_states_df.at[row_index, 'Background'] = average_background

        # Save the updated DataFrame back to the CSV file
        sea_states_df.to_csv('sea_states.csv', index=False)

        # Delete the files to free space
        os.remove(swn_file_name + '.prt')

os.remove(files_to_copy)

