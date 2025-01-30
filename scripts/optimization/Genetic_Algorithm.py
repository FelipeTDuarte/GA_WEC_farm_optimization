import os
import shutil
import pickle
import subprocess
import numpy as np
import pandas as pd
import math
import random
from shapely.geometry import Point, Polygon, LineString


############################################# functions #############################################

# find the last generation simulated
def last_generation():
    all_directories = [d for d in os.listdir('.') if os.path.isdir(os.path.join('.', d))]
    matching_directories = [d for d in all_directories if d.startswith('gen')]
    last_generation_dir = max(matching_directories, default=None, key=lambda x: int(x[3:]) if x[3:].isdigit() else 0)
    last_generation_num = int(last_generation_dir[3:]) if last_generation_dir else None
    return last_generation_dir, last_generation_num

# load last generation data
def load_last_gen_data(output_folder):
    current_gen = output_folder
    previous_gen = 'gen' + str(int(current_gen[3:]) - 1).zfill(3)

    with open(os.path.join(previous_gen, 'previous_elite_parent_segments.pkl'), 'rb') as f:
        previous_elite_parent_segments = pickle.load(f)

    with open(os.path.join(previous_gen, 'previous_elite_parent.pkl'), 'rb') as f:
        previous_elite_parent = pickle.load(f)

    with open(os.path.join(previous_gen, 'previous_best_parent_segments.pkl'), 'rb') as f:
        previous_best_parent_segments = pickle.load(f)

    with open(os.path.join(previous_gen, 'previous_best_parent.pkl'), 'rb') as f:
        previous_best_parent = pickle.load(f)


    return previous_elite_parent_segments,previous_elite_parent, previous_best_parent_segments, previous_best_parent

# Create a new directory every generation loop

def make_output_folder(output_folder, files_to_copy):
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
    #print(files_to_copy)
    for file_name in files_to_copy:
        #print('file_name: ', file_name)
        source_file = os.path.join('./', file_name)  # Full path to the source file
        #print('source_file: ', source_file)
        destination_file = os.path.join(output_folder, file_name)  # Full path to the destination file
        #print('destination_file: ', destination_file) 

        if os.path.isdir(source_file):
            shutil.copytree(source_file, destination_file)
        else:
            shutil.copy(source_file, destination_file)


    return output_folder

# Evaluation 

## Fitness

def min_max_scaling(column):
    return (column - column.min()) / (column.max() - column.min())

def min_max_scaling_inverse(column):
    return 1 - ((column - column.min()) / (column.max() - column.min()))

def calculate_fitness(dataframe,HRA_normalization, s=None):
    # Calculate %HRA and %P in the DataFrame
    if HRA_normalization == 'max':
        dataframe['N_%HRA'] = min_max_scaling(dataframe['HRA'])
    elif HRA_normalization == 'min':
        dataframe['N_%HRA'] = min_max_scaling_inverse(dataframe['HRA'])
    else:
        raise ValueError("Normalization must be 'max' or 'min'.")
    
    dataframe['N_%P'] = min_max_scaling(dataframe['average_power'])

    if s is not None:
        # If s is defined, use the given value
        p = 1 - s
        array_fitness = np.average(dataframe[['N_%HRA', 'N_%P']], weights=[s, p], axis=1)
        dataframe['WLA'] = array_fitness
    else:
        # If s is not defined, use the range of s values from 0 to 1.0 with a step of 0.1
        s_values = np.arange(0, 1.1, 0.1)
        wla_values = []

        for s in s_values:
            p = 1 - s
            wla_values.append(np.average(dataframe[['N_%HRA', 'N_%P']], weights=[s, p], axis=1))

        array_fitness = np.average(wla_values, axis=0)
        dataframe['WLA'] = array_fitness

    return dataframe, array_fitness

## Termination

def check_termination(current_iteration, tolerance, max_consecutive_stable_iterations):
    if current_iteration < max_consecutive_stable_iterations:
        return False

    for i in range(current_iteration, current_iteration - max_consecutive_stable_iterations, -1):
        current_csv = os.path.join( f'gen{i:03d}', 'weighted_array_avg.csv')
        previous_csv = os.path.join( f'gen{(i-1):03d}', 'weighted_array_avg.csv')

        #print(current_csv)
        #print(previous_csv)

        current_data = pd.read_csv(current_csv)
        previous_data = pd.read_csv(previous_csv)

        # Check if both columns (%HRA and %P) do not vary more than the tolerance
        if (
            abs(current_data['HRA'].max() - previous_data['HRA'].max()) > tolerance
            or abs(current_data['average_power'].max() - previous_data['average_power'].max()) > tolerance*1000
        ):
            return False

    return True

# Selection

## Elite

def elite_selection(weighted_array_avg_df, population_size, top_percentil, output_folder, previous_elite_parent_segments):
    elite_len = int(population_size * top_percentil)
    elite_df = weighted_array_avg_df.nlargest(elite_len, 'WLA')[['array', 'WLA']].reset_index(drop=True)
    elite_weighted_array_avg = weighted_array_avg_df.nlargest(elite_len, 'WLA')[
        ['HRA', 'average_Hs', 'average_power','Q-factor','accessibility_increase','standard_deviation','coeff_variation','power_range','power_peak_to_mean','power_mean_to_minimum']].reset_index(drop=True)
    elite = set(elite_df['array'])

    #print(elite_weighted_array_avg)

    elite_parent_segments = []
    for index, row in elite_df.iterrows():
        elite_parent = int(row['array'])
        elite_parent_file = os.path.join(output_folder, f'st00a{str(elite_parent).zfill(2)}.swn')

        if os.path.exists(elite_parent_file):
            elite_parent_segment = extract_segments_from_file(elite_parent_file)
        elif previous_elite_parent_segments is not None:
            if elite_parent < (len(elite)):
                elite_parent_segment = previous_elite_parent_segments[elite_parent]
            else:
                
                elite_parent_segment = previous_best_parent_segments[elite_parent-len(elite)]
            
        else:
            elite_parent_segment = None

        elite_parent_segments.append(elite_parent_segment)
   
    return elite, elite_df, elite_weighted_array_avg, elite_parent_segments

## Selection by ranking

def selection_by_ranking (weighted_array_avg_df,population_size, elite):
    # Sort the DataFrame by 'WLA' in descending order
    weighted_array_avg_df_sorted = weighted_array_avg_df.sort_values(by='WLA', ascending=False)[['array','WLA']]

    # Add a new column 'Probability_Rank' based on the ranking
    weighted_array_avg_df_sorted['Rank'] = range(len(weighted_array_avg_df_sorted),0, -1)
    weighted_array_avg_df_sorted['cum_sum'] =weighted_array_avg_df_sorted['Rank'].cumsum()

    #print(weighted_array_avg_df_sorted)


    parents1 = []
    parents2 = []
    num_pairs_to_select = population_size-len(elite)

    while len(parents1) < num_pairs_to_select:
        # Generate two different random integers between 1 and the maximum cumulative sum
        random_integer1 = np.random.randint(1, weighted_array_avg_df_sorted['cum_sum'].max() + 1)
        random_integer2 = np.random.randint(1, weighted_array_avg_df_sorted['cum_sum'].max() + 1)

        #print(len(parents1))
        #print('random:',random_integer1, random_integer2)

        if random_integer1 != random_integer2:
            # Find the closest ranks based on the random integers
            parent1 = np.argmin(np.abs(weighted_array_avg_df_sorted['cum_sum'] - random_integer1))
            parent2 = np.argmin(np.abs(weighted_array_avg_df_sorted['cum_sum'] - random_integer2))

            if (parent1 != parent2) and ((parent1, parent2) not in zip(parents1, parents2)) and ((parent2, parent1) not in zip(parents1, parents2)):
                # Append the selected indices to the list
                parents1.append(parent1)
                parents2.append(parent2)

                

                #print('parent1:', parent1, 'parent2:', parent2)
    
    # Select the corresponding pairs from the DataFrame for parent1
    parent1_df = weighted_array_avg_df_sorted.loc[parents1, ['array', 'WLA']]
    parent1_df.columns = ['array1', 'WLA1']

    # Select the corresponding pairs from the DataFrame for parent2
    parent2_df = weighted_array_avg_df_sorted.loc[parents2, ['array', 'WLA']]
    parent2_df.columns = ['array2', 'WLA2']

    # Merge the two DataFrames on their indices
    selected_pairs_df = pd.concat([parent1_df.reset_index(drop=True), parent2_df.reset_index(drop=True)], axis=1)
    parents = parents1,parents2

    return selected_pairs_df, parents
    
# Reproduction

# crossover functions

# Function to extract segment information from a file
def extract_segments_from_file(file_path):
    segments = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith("OBST"):
                # Extracting coordinates of the extreme points
                _, _, _, _, initial_point_x, initial_point_y, final_point_x, final_point_y = line.split()
                segments.append((float(initial_point_x), float(initial_point_y), float(final_point_x), float(final_point_y)))
            
    return segments

def calculate_distance(segment1, segment2):
    #print('calculating distance: ',segment1, segment2)
    # Calculate the distance between the centers of two segments
    x1, y1 = (segment1[0] + segment1[2]) / 2, (segment1[1] + segment1[3]) / 2
    x2, y2 = (segment2[0] + segment2[2]) / 2, (segment2[1] + segment2[3]) / 2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    #print('distance: ',distance)
    #print('segment1: ', segment1, 'segment2: ', segment2)
    return distance

def create_new_segments(layout1, layout2, minimum_distance):
    num_segments = len(layout1)
    child_segments = []

    attempts = 0

    while len(child_segments) < num_segments:
        selected_segment = random.choice(layout1 if random.random() > 0.5 else layout2)

        # Check if the selected segment satisfies the minimum distance condition with all existing segments
        if all(calculate_distance(selected_segment, existing_segment) >= minimum_distance for existing_segment in child_segments):
            child_segments.append(selected_segment)
            attempts = 0  # Reset attempts counter if a valid segment is found
        else:
            attempts += 1
            if attempts > 100000:
                print('More than', attempts, 'attempts, restarting child')
                child_segments = []  # Restart child_segments list
                attempts = 0  # Reset attempts counter

    return child_segments


def apply_crossover(output_folder,selected_pairs_df, crossover_rate, minimum_distance):
    rng = np.random.default_rng()

    children_list = []
    best_parent_list = []
    
    parent1_segments = []
    parent2_segments = []
    #parent1_segment = []
    #parent2_segment = []

    for index, row in selected_pairs_df.iterrows():
        
        all_segment_list = elite_parent_segments.copy()
        parent1 = int(row['array1'])
        # print('parent1',parent1)
        parent1_file = os.path.join(output_folder, f'st00a{str(parent1).zfill(2)}.swn')
        parent1_segments = []
        if os.path.exists(parent1_file):
            parent1_segment = extract_segments_from_file(parent1_file)
            #print('parent1_segment bigger than elite + best parent')
        elif parent1 < (len(elite)):
                parent1_segment = elite_parent_segments[parent1]
                #print('parent1_segment in elite')
        elif previous_best_parent_segments:
            if previous_best_parent[(parent1-len(elite))] < (len(elite)):
                parent1_segment = elite_parent_segments[previous_best_parent[(parent1-len(elite))] ]
                #print('parent1_segment in best parent is elite')
            else:
                parent1_segment = previous_best_parent_segments[(parent1-len(elite))]
                #print('parent1_segment in best parent')
        
        parent1_segments.append(parent1_segment)

        parent2 = int(row['array2'])
        # print('parent2',parent2)
        parent2_file = os.path.join(output_folder, f'st00a{str(parent2).zfill(2)}.swn')
        parent2_segments = []
        if os.path.exists(parent2_file):
            parent2_segment = extract_segments_from_file(parent2_file)
            #print('parent2_segment bigger than elite + best parent')
        elif parent2 < (len(elite)):
                parent2_segment = elite_parent_segments[parent2]
                #print('parent2_segment in elite')
        elif previous_best_parent_segments:
            if previous_best_parent[(parent2-len(elite))] < (len(elite)):
                parent2_segment = elite_parent_segments[previous_best_parent[(parent2-len(elite))] ]
                #print('parent2_segment in best parent is elite')
            else:
                parent2_segment = previous_best_parent_segments[(parent2-len(elite))]
                #print('parent2_segment in best parent', previous_best_parent[(parent2-len(elite))])

        
        parent2_segments.append(parent2_segment)
       
        #print(index,'parent1:', parent1,'parent2:',parent2)

        if rng.random() <= crossover_rate:    
            #print('crossover_rate <', crossover_rate) 
            #print('parent1:', parent1_segment, 'parent2:', parent2_segment)       
            child_segment = create_new_segments(parent1_segment, parent2_segment,minimum_distance)         
            children_list.append(child_segment)
            all_segment_list.append(child_segment)
        else:  
            
            # Both parents in Elite perform crossover always
            if parent1 in elite and parent2 in elite:
                #print('both in elite')
                child_segment = create_new_segments(parent1_segment, parent2_segment,minimum_distance)
                children_list.append(child_segment)
                all_segment_list.append(child_segment)

            # One is in Elite choose the one that is not in elite
            elif parent1 in elite or parent2 in elite:
                
                not_elite_parent = parent1 if parent2 in elite else parent2
                #print('non elite parent:', not_elite_parent)
                not_elite_parent_file = os.path.join(output_folder, f'st00a{str(not_elite_parent).zfill(2)}.swn')

                if os.path.exists(not_elite_parent_file):
                    not_elite_parent_segments = extract_segments_from_file(not_elite_parent_file)

                    best_parent = not_elite_parent
                    if best_parent not in best_parent_list:
                        all_segment_list.append(not_elite_parent_segments)
                        best_parent_list.append(best_parent)
                    else:
                        child_segment = create_new_segments(parent1_segment, parent2_segment,minimum_distance)         
                        children_list.append(child_segment)
                        all_segment_list.append(child_segment)

                elif previous_best_parent_segments:
                    if not_elite_parent < (len(elite)):
                        not_elite_parent_segments = previous_elite_parent_segments[not_elite_parent]
                        
                        best_parent = not_elite_parent
                        if best_parent not in best_parent_list:
                            all_segment_list.append(not_elite_parent_segments)
                            best_parent_list.append(best_parent)
                        else:
                            child_segment = create_new_segments(parent1_segment, parent2_segment,minimum_distance)         
                            children_list.append(child_segment)
                            all_segment_list.append(child_segment)
                    else:
                        not_elite_parent_segments = previous_best_parent_segments[not_elite_parent-len(elite)]   
                        #print('non elite parent is from previous best parent')   
                
                        best_parent = not_elite_parent
                        if best_parent not in best_parent_list:
                            all_segment_list.append(not_elite_parent_segments)
                            best_parent_list.append(best_parent)
                        else:
                            child_segment = create_new_segments(parent1_segment, parent2_segment,minimum_distance)         
                            children_list.append(child_segment)
                            all_segment_list.append(child_segment)
                    
                else:
                    child_segment = create_new_segments(parent1_segment, parent2_segment,minimum_distance)         
                    children_list.append(child_segment)
            # None is in Elite
            else:
                #print('none in elite')
                
                if row['WLA1'] > row['WLA2']:
                    best_parent = parent1
                    if best_parent not in best_parent_list:
                        all_segment_list.append(parent1_segment)
                        best_parent_list.append(best_parent)
                    else:
                        best_parent = parent2
                        if best_parent not in best_parent_list:
                            all_segment_list.append(parent2_segment)
                            best_parent_list.append(best_parent)
                        else:
                            child_segment = create_new_segments(parent1_segment, parent2_segment,minimum_distance)         
                            children_list.append(child_segment)
                            all_segment_list.append(child_segment)
                else:
                    
                    best_parent = parent2
                    if best_parent not in best_parent_list:
                        all_segment_list.append(parent2_segment)
                        best_parent_list.append(best_parent)
                    else:
                        best_parent = parent1
                        if best_parent not in best_parent_list:
                            all_segment_list.append(parent1_segment)
                            best_parent_list.append(best_parent)
                        else:
                            child_segment = create_new_segments(parent1_segment, parent2_segment,minimum_distance)         
                            children_list.append(child_segment)
                            all_segment_list.append(child_segment)
                    
    
    children_df = pd.DataFrame(children_list)
    children_df.to_csv(os.path.join(output_folder,'children_data.csv'), index=False)
    
    best_parent_df = weighted_array_avg_df.iloc[(best_parent_list)].reset_index(drop=True)[['array','WLA']]
    best_parent_weighted_array_avg = weighted_array_avg_df.iloc[(best_parent_list)].reset_index(drop=True)[['HRA','average_Hs','average_power','Q-factor','accessibility_increase','standard_deviation','coeff_variation','power_range','power_peak_to_mean','power_mean_to_minimum']]
    #print(elite_weighted_array_avg)
    
    #print('lenght all_segment_list', len(all_segment_list))
    best_parent_segments = []
    for index, row in best_parent_df.iterrows():
        best_parent = int(row['array'])
        best_parent_file = os.path.join(output_folder, f'st00a{str(best_parent).zfill(2)}.swn')
        #print('best_parent (inside crossover)',best_parent)

        if os.path.exists(best_parent_file):
            best_parent_segment = extract_segments_from_file(best_parent_file)
        elif previous_best_parent_segments:
            
            if best_parent < (len(elite)):
                best_parent_segment = previous_elite_parent_segments[best_parent]
            else:
                #print(len(previous_best_parent_segments),best_parent-len(elite))
                best_parent_segment = previous_best_parent_segments[best_parent-len(elite)]
            
        else:
            best_parent_segment = None

        best_parent_segments.append(best_parent_segment)
    
    return children_list, best_parent_list,best_parent_df, best_parent_weighted_array_avg,best_parent_segments

# Mutation 

def apply_mutation(children, mutation_rate,minimum_distance):
    mutated_layouts = []
    for layout in children:
        mutated_segments = []
        for segment in layout:
            if deploy_type == 'sector':
                if random.random() < mutation_rate:
                    mutant = create_array_sector(1, min_dist, wec_length, start_angle, end_angle,sector_dist_min, sector_dist_max, central_point)
                    mutated_segment = (mutant[0][0][1], mutant[0][0][0],mutant[0][1][1],mutant[0][1][0])
                    
                    if all(calculate_distance(mutated_segment, existing_segment) >= minimum_distance for existing_segment in layout):
                        mutated_segments.append(mutated_segment)
                        #print('mutation')
                    else:
                        mutated_segments.append(segment)    
                        #print('mutation failed - can not place')
                else:
                    mutated_segments.append(segment)

            elif deploy_type == 'polygon':
                if random.random() < mutation_rate:
                    mutant = create_array_polygon(area_deploy, 1, min_dist, wec_length, fixed_angle)
                    mutated_segment = (mutant[0][1][0], mutant[0][0][1],mutant[0][1][0],mutant[0][1][1])
                    print(mutated_segment)
                    
                    if all(calculate_distance(mutated_segment, existing_segment) >= minimum_distance for existing_segment in layout):
                        mutated_segments.append(mutated_segment)
                        #print('mutation')
                    else:
                        mutated_segments.append(segment)    
                        #print('mutation failed - can not place')
                else:
                    mutated_segments.append(segment)
            

        mutated_layouts.append(mutated_segments)
    return mutated_layouts


# Create a random array in the selected area using UTM coordinates

def create_array_sector(num_wecs, min_dist, wec_length, start_angle, end_angle,start_distance, end_distance, arc_center):
    random_points = []
    segments = []
    
    while len(random_points) < num_wecs:
        distance = np.random.uniform(start_distance , end_distance )
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

            dx = segment_length / 2 * np.cos(np.radians(angle))
            dy = segment_length / 2 * np.sin(np.radians(angle))
            start_point = (random_point.x - dx, random_point.y - dy)
            end_point = (random_point.x + dx, random_point.y + dy)
            segment = LineString([start_point, end_point])

            if polygon.contains(segment) and is_valid_point(random_point, points, min_distance):
                points.append(random_point)
                segments.append((start_point,end_point))
    return segments

# Write the WECs as an obstacle in SWAN input file
# Function for the initial random array
def write_segments_to_file(output_folder,sea_state, array_number, segments):
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
                    output_file.write(f"OBST\tTRANS\t0\tLIN {initial_point[1]} {initial_point[0]} {final_point[1]} {final_point[0]}\n")
    
    print(f"WEC array written to {output_filename}")

# Function for the offspring array 
def write_offspring_to_file(output_folder,sea_state, array_number, segments):
    template_filename = os.path.join(sea_states_folder, f"st{sea_state:02}.swn")  # Assuming files are named st01, st02, ..., st10
    output_filename = os.path.join(output_folder, f"st{sea_state:02}a{array_number:02}.swn")
    marker = "$******************************** WEC ARRAY *****************************"
    
    with open(template_filename, "r") as template_file, open(output_filename, "w") as output_file:
        for line in template_file:
            output_file.write(line)
            if marker in line:
                output_file.write("\n")
         
                for segment in segments:
                    
                    #print ('seg:',segment)
                    initial_point = segment[0:2]
                    final_point = segment[2:4]
                    
                    output_file.write(f"OBST\tTRANS\t0\tLIN {initial_point[0]} {initial_point[1]} {final_point[0]} {final_point[1]}\n")
    
# Function to determine the sea states and array values in the current folder
def determine_sea_state_and_array_ranges():
    sea_states = set()
    array_values = set()

    for filename in os.listdir("."):
        if filename.endswith(".swn"):
            parts = filename.split("a")
            if len(parts) == 2 and parts[0][2:].isdigit() and parts[1][:2].isdigit():
                sea_states.add(int(parts[0][2:]))
                array_values.add(int(parts[1][:2]))

    return sorted(sea_states), sorted(array_values)

def delete_files(files_to_delete):
    current_folder = os.getcwd()

    for file_name in files_to_delete:
        file_path = os.path.join(current_folder, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)

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



############################################# Inputs #################################################
# Call the function to read the configuration
config = read_config()

# print configuration input
print("Configuration:")
for key, value in config.items():
    print(f"{key}: {value}")

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

# Read protection objectives
shield_value = config.get('shield')
if shield_value is None or shield_value == 'None':
    shield = None
else:
    try:
        shield = float(shield_value)  # Converte para float se for um número válido
    except ValueError:
        print("Invalid shield value in configuration file.")
        shield = None

HRA_normalization = config['HRA_normalization']

# Read optimization parameters
max_generations = int(config['max_generations'])
tolerance = float(config['tolerance'])
max_consecutive_stable_iterations = int(config['max_consecutive_stable_iterations'])
elite_size = float(config['elite_size'])
crossover_rate = float(config['crossover_rate'])
mutation_rate = float(config['mutation_rate'])

files_to_copy = [config['bottom_file'], config['grid_file'], config['snl_power_file'], config['sea_states_file'], config['swanrun_script']]
files_to_copy.extend(["output_evaluation.py", "pre_to_avg.py", "run_swan_parallel.sh"])

############################################# Sea States and background ##############################

# Background simulation
sea_states_folder = config['sea_states_folder']

if not os.path.exists(sea_states_folder):
    print('Generating background...')
    subprocess.run(['python3', 'st_files_generator.py'])
    print('Sea States and background HRA created')
else: 
    print('Sea States and background HRA exist')

# Sea States

sea_states_df = pd.read_csv(os.path.join(sea_states_folder, config['sea_states_file']))

# Hs and P background
sea_states_df['Weighted_Background'] = sea_states_df['Background'] * sea_states_df['COUNT']
sea_states_df['Weighted_P'] = sea_states_df['P'] * sea_states_df['COUNT']
total_weight = sea_states_df['COUNT'].sum()
weighted_avg_background = sea_states_df['Weighted_Background'].sum() / total_weight
weighted_avg_P = sea_states_df['Weighted_P'].sum() / total_weight

if deploy_type == 'polygon':
    area_plot = pd.read_csv(deploy_area, sep='\t', skiprows=2, header=None)
    area_plot.columns = ['X', 'Y']
    area_deploy = Polygon(zip(area_plot['X'], area_plot['Y']))
else:
    area_deploy = None

############################################# Initial Population ######################################

if not os.path.exists('gen000'):
    subprocess.run(['python3', 'Initial_population_and_run.py'])
    print('Initial generation created')
else: 
    print('Initial generation already exists')


############################################# GA LOOP ################################################

last_generation_dir, last_generation_num = last_generation()
evaluation_list = []
elite_list = []
crossover_list = []

if last_generation_dir == "gen000":
    previous_elite_parent_segments = []
    previous_best_parent_segments = []
    previous_elite_parent = []
    previous_best_parent = []
else:
    previous_elite_parent_segments,previous_elite_parent, previous_best_parent_segments, previous_best_parent = load_last_gen_data(last_generation_dir)


# Initialize GA loop

for generation in range(last_generation_num, max_generations + 1):
    
    print(' ')
    print('Evaluation of generation:',last_generation_dir)

    # Evaluation (fitness)
    #print('Evaluating...')
    weighted_array_avg_df = pd.read_csv(os.path.join(last_generation_dir, "weighted_array_avg.csv"))
 
    if shield is not None:
        weighted_array_avg_df, fitness = calculate_fitness(weighted_array_avg_df,HRA_normalization, s=shield)
    else:
        weighted_array_avg_df, fitness = calculate_fitness(weighted_array_avg_df,HRA_normalization)

    evaluation_list.append({
        'weighted_array_avg_df': weighted_array_avg_df,
        'fitness': fitness,
    })
       
    print('max Fitness:',fitness.max())
    print('max HRA:',weighted_array_avg_df['HRA'].max())
    print('max Power abs:',weighted_array_avg_df['average_power'].max())

    # check termination criteria
    print('\n Checking terminarion criteria...')
    if check_termination(generation, tolerance, max_consecutive_stable_iterations):
        print(f"Termination criteria met after {generation} generations.")
        break
    
    # Elite
    print('Setting Elite...')
    # print ('previous_elite(antes da nova seleção):', previous_elite_parent)
    elite, elite_df, elite_weighted_array_avg, elite_parent_segments = elite_selection(weighted_array_avg_df,num_arrays,elite_size,last_generation_dir, previous_elite_parent_segments)
    
    elite_list.append({
        'elite': elite,
        'elite_df': elite_df,
        'elite_weighted_array_avg': elite_weighted_array_avg,
        'elite_parent_segments': elite_parent_segments
    })
    
    with open(os.path.join(last_generation_dir,'elite_list.pkl'), 'wb') as f:
        pickle.dump(elite_list, f)

    previous_elite_parent_segments = elite_list[-1]['elite_parent_segments']
    with open(os.path.join(last_generation_dir, 'previous_elite_parent_segments.pkl'), 'wb') as f:
        pickle.dump(previous_elite_parent_segments, f)

    previous_elite_parent = elite_list[-1]['elite'].copy()   
    with open(os.path.join(last_generation_dir, 'previous_elite_parent.pkl'), 'wb') as f:
        pickle.dump(previous_elite_parent, f) 
    
    #print ('elite:',elite)

    # Selection by Ranking
    print('Selection by Ranking...')
    selected_pairs_df, parents = selection_by_ranking(weighted_array_avg_df,num_arrays, elite)

    with open(os.path.join(last_generation_dir,'selected_pairs.pkl'), 'wb') as f:
        pickle.dump(parents, f)

    #print('parents:',parents)

    # Crossover
    print('Crossover...')
    #print ('previous_best_parent(antes do crossover):', previous_best_parent)
    #print ('previous_best_parent_segments(antes do crossover):', previous_best_parent_segments)
    #print ('len_previous_best_parent_segments(antes do crossover):', len(previous_best_parent_segments))

    children,  best_parent,best_parent_df, best_parent_weighted_array_avg, best_parent_segments= apply_crossover(last_generation_dir,selected_pairs_df,crossover_rate, min_dist)

    crossover_list.append({
        'children': children,
        'best_parent': best_parent,
        'best_parent_df': best_parent_df,
        'best_parent_weighted_array_avg': best_parent_weighted_array_avg,
        'best_parent_segments': best_parent_segments,
    })
    #print('best_parent (after crossover):',best_parent)
    with open(os.path.join(last_generation_dir,'crossover_list.pkl'), 'wb') as f:
        pickle.dump(crossover_list, f)


    previous_best_parent_segments = crossover_list[-1]['best_parent_segments']
    with open(os.path.join(last_generation_dir, 'previous_best_parent_segments.pkl'), 'wb') as f:
        pickle.dump(previous_best_parent_segments, f) 

    previous_best_parent = crossover_list[-1]['best_parent']
    with open(os.path.join(last_generation_dir, 'previous_best_parent.pkl'), 'wb') as f:
        pickle.dump(previous_best_parent, f)

    #print('best_parent (after crossover):',best_parent)

    # Mutation
    print('Mutation...')
    offspring = apply_mutation(children,mutation_rate,min_dist)
    #offspring_df  = pd.DataFrame(offspring)
    #offspring_df.to_csv('offspring.csv', index=False)
    
    # Offspring (gen001)
    print('Next generation offspring...')
    if __name__ == "__main__":
        
        new_generation_dir = make_output_folder(last_generation_dir,files_to_copy)
    # Iterate through all st*.swn files in the sea_states folder
    for filename in os.listdir(sea_states_folder):
        if filename.startswith("st") and filename.endswith(".swn"):
            # Extract sea state number from the filename (assuming it's always a two-digit number)
            sea_state = int(filename[2:4])
            for array_number in range((len(elite)+len(best_parent)),num_arrays):  # Create arrays
                arrays_UTM = offspring[array_number-(len(elite)+len(best_parent))]
                
                write_offspring_to_file(new_generation_dir,sea_state, array_number, arrays_UTM)
        #copy the file sea_states.csv to the new folder
        if filename == config['sea_states_file']:
            shutil.copy(os.path.join(sea_states_folder, filename), new_generation_dir)





    # population to transmited without modification (Elite+ Best Parents)
    to_keep = pd.concat([elite_weighted_array_avg, best_parent_weighted_array_avg], axis=0).reset_index(drop=True)
    to_keep.reset_index(inplace=True,drop=False)
    to_keep.rename(columns={'index':'array'}, inplace=True)
    to_keep.to_csv(os.path.join(new_generation_dir, "weighted_array_avg.csv"), index=False, header=True, columns=['array', 'HRA','average_Hs', 'average_power','Q-factor','accessibility_increase','standard_deviation','coeff_variation','power_range','power_peak_to_mean','power_mean_to_minimum'])

    #run SWAN for offspring
    print('SWAN Simulation of generation: ', new_generation_dir)
    last_generation_dir = new_generation_dir
    os.chdir(new_generation_dir)
    #run_SNL_SWAN(num_wecs)
    #fake_run_SWAN()
    shell_script_path = os.path.join(new_generation_dir, "run_swan_parallel.sh")
    #print(shell_script_path)
    subprocess.run(['bash', "run_swan_parallel.sh"])
    delete_files(files_to_copy)
    os.chdir("..")
