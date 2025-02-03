import os
import shutil
import subprocess
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, LineString

################ FUNCTIONS ######################################

def create_array_sector(num_wecs, min_dist, wec_length, start_angle, end_angle, arc_center):
    random_points = []
    segments = []
    
    while len(random_points) < num_wecs:
        distance = np.random.uniform(200, 600)
        direction = np.random.uniform(start_angle, end_angle)
        
        x_offset = distance * np.cos(np.radians(direction))
        y_offset = distance * np.sin(np.radians(direction))
        
        random_point = (arc_center[0] + x_offset, arc_center[1] + y_offset)
        
        x_offset = wec_length / 2 * np.cos(np.radians(direction+90))
        y_offset = wec_length / 2 * np.sin(np.radians(direction+90))
        
        initial_point = (random_point[0] - x_offset, random_point[1] - y_offset)
        final_point = (random_point[0] + x_offset, random_point[1] + y_offset)
        
        if all(np.linalg.norm(np.array(p) - np.array(random_point)) >= min_dist for p in random_points):
            random_points.append(random_point)
            segments.append((initial_point, final_point))
            #print(segments)
            # print(f"Positioning WEC {len(random_points)}")
    
    return segments


def is_valid_point(new_point, points, min_distance):
    for point in points:
        if new_point.distance(point) < min_distance:
            return False
    return True

def create_array_polygon(polygon, num_points, min_distance, segment_length, Nautical_angle):
    min_x, min_y, max_x, max_y = polygon.bounds
    points = []
    segments = []
    angle = 360 - Nautical_angle
    while len(points) < num_points:
        random_point = Point(np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y))
        if polygon.contains(random_point):
            # Calculando os pontos de início e fim do segmento
            dx = segment_length / 2 * np.cos(np.radians(angle))
            dy = segment_length / 2 * np.sin(np.radians(angle))
            start_point = (random_point.x - dx, random_point.y - dy)
            end_point = (random_point.x + dx, random_point.y + dy)
            segment = LineString([start_point, end_point])
            # Verificando se o segmento está completamente dentro do polígono
            if polygon.contains(segment) and is_valid_point(random_point, points, min_distance):
                points.append(random_point)
                segments.append((start_point,end_point))
    return segments

# Function to write the WECs as an obstacle in SWAN input file
def write_segments_to_file(sea_state, array_number, segments):
    template_filename = os.path.join(sea_states_folder, f"st{sea_state:02}.swn")  # Assuming files are named st01, st02, ..., st10
    output_filename = os.path.join(output_folder, f"st{sea_state:02}a{array_number:02}.swn")
    marker = "$******************************** WEC ARRAY *****************************"
     
    with open(template_filename, "r") as template_file, open(output_filename, "w") as output_file:
        for line in template_file:
            output_file.write(line)
            if marker in line:
                output_file.write("\n")
                for segment in segments:
                    initial_point = segment[0]
                    final_point = segment[1]
                    output_file.write(f"OBST\tTRANS\t0\tLIN {initial_point[0]} {initial_point[1]} {final_point[0]} {final_point[1]}\n")
    
    print(f"WEC array written to {output_filename}")

def make_output_folder(output_folder):
    """Create a folder if it does not exist, or increment the numbering if it already exists.

    Args:
        output_folder: The path to the folder to create.
    """
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    else:
        # Increment the numbering if the folder already exists.
        i = int(output_folder[-3:]) + 1
        while os.path.exists(f"{output_folder[:-3]}{i:03d}"):
            i += 1
        output_folder = f"{output_folder[:-3]}{i:03d}"
        os.mkdir(output_folder)
    
    # Copy input files from the main folder to the newly created folder.
    files_to_copy = [config['bottom_file'], config['grid_file'], config['snl_power_file'], config['sea_states_file'], config['swanrun_script']]
    files_to_copy.extend(["output_evaluation.py", "pre_to_avg.py", "run_swan_parallel.sh"])


    for file_name in files_to_copy:
        source_file = os.path.join('./', file_name)  # Full path to the source file
        destination_file = os.path.join(output_folder, file_name)  # Full path to the destination file

        shutil.copy(source_file, destination_file)


    return output_folder

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

################ INPUTS ######################################
# Call the function to read the configuration
config = read_config()


# Read WEC farm parameters
num_arrays = int(config['num_arrays'])
num_wecs = int(config['num_wecs'])
wec_length = float(config['wec_length'])
min_dist = float(config['min_dist'])*wec_length

# Read deployment parameters
deploy_type = config['deploy_type']
deploy_area = config['deploy_area']
central_point = eval(config['central_point'])
sector_dist_min = float(config['sector_dist_min'])
sector_dist_max = float(config['sector_dist_max'])
start_angle = float(config['start_angle'])
end_angle = float(config['end_angle'])
fixed_angle = float(config['fixed_angle'])

# Folder containing st*.swn files
sea_states_folder = "sea_states"


area_plot = pd.read_csv(config['sea_states_file'], sep='\t', skiprows=2, header=None)
area_plot.columns = ['X', 'Y']
area_deploy = Polygon(zip(area_plot['X'], area_plot['Y']))

# Output folder for new files
if __name__ == "__main__":
    output_folder = f"gen000"
    #output_folder = f"single_wec000"
    output_folder = make_output_folder(output_folder)


# Generate a list of arrays
array_list = []
for array_number in range(num_arrays):
    if deploy_type == 'sector':
        #print('Deploy in a sector')
        segments_UTM = create_array_sector(num_wecs, min_dist, wec_length, start_angle, end_angle, central_point)
        #print(segments_UTM)
        array_list.append(segments_UTM)
    elif deploy_type == 'polygon':
        #print ('Deploy in a closed polygon')
        segments_UTM = create_array_polygon(area_deploy, num_wecs, min_dist, wec_length, fixed_angle)
        array_list.append(segments_UTM)
        #print(segments_UTM)
    else:
        print("Error! Choose 'polygon' or 'sector' for area of WEC placement")


# Iterate through all st*.swn files in the sea_states folder
for filename in os.listdir(sea_states_folder):
    if filename.startswith("st") and filename.endswith(".swn"):
        # Extract sea state number from the filename (assuming it's always a two-digit number)
        sea_state = int(filename[2:4])
        
        for array_number in range(num_arrays):  # Create arrays
            arrays_UTM = array_list[array_number]
            write_segments_to_file(sea_state, array_number, arrays_UTM)
        
# Change the current working directory to the sea states folder
current_directory = os.getcwd()
sea_states_df = pd.read_csv(os.path.join(sea_states_folder, 'sea_states.csv'))

os.chdir(output_folder)
#print('files created. Ok to run SWAN')
shell_script_path = os.path.join(output_folder, "run_swan_parallel.sh")
subprocess.run(['bash', "run_swan_parallel.sh"])

os.chdir("..")

