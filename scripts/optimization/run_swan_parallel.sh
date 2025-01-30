#!/bin/bash

# Get the current directory
CURRENT_DIR=$(pwd)

# Navigate to the parent directory
PARENT_DIR=$(dirname "$(pwd)")

# Search for the first .ga file in the parent directory
CONFIG_FILE=$(find "$PARENT_DIR" -maxdepth 1 -name "*.ga" | head -n 1)

# Check if a .ga file was found
if [[ -z "$CONFIG_FILE" ]]; then
    echo "No .ga file found in the parent directory."
    exit 1
fi

# Extract and define variables from the configuration file
TYPE="$(awk -F "=" '/sim_type/ {gsub(/[ \t\r\n]+/, "", $2); print "-" $2}' "$CONFIG_FILE")"
NUM_PROCESSES=$(awk -F "=" '/sim_cores/ {gsub(/[^0-9]/, "", $2); print int($2)}' "$CONFIG_FILE")
MAX_PARALLEL=$(awk -F "=" '/sim_parallel/ {gsub(/[^0-9]/, "", $2); print int($2)}' "$CONFIG_FILE")

# Extract the model files
BOTTOM_FILE="$(awk -F "=" '/bottom_file/ {gsub(/[ \t\r\n]+/, "", $2); print "" $2}' "$CONFIG_FILE")"
GRID_FILE="$(awk -F "=" '/grid_file/ {gsub(/[ \t\r\n]+/, "", $2); print "" $2}' "$CONFIG_FILE")"
SWAN_EXECUTABLE="$(awk -F "=" '/swanrun_script/ {gsub(/[ \t\r\n]+/, "", $2); print "./" $2}' "$CONFIG_FILE")"
SNL_POWER_FILE="$(awk -F "=" '/snl_power_file/ {gsub(/[ \t\r\n]+/, "", $2); print "" $2}' "$CONFIG_FILE")"
ST_FILE="$(awk -F "=" '/sea_states_file/ {gsub(/[ \t\r\n]+/, "", $2); print "" $2}' "$CONFIG_FILE")"
#echo "ST_FILE: $ST_FILE"
#echo "TYPE: $TYPE"
#echo "NUM_PROCESSES: $NUM_PROCESSES"
#echo "MAX_PARALLEL: $MAX_PARALLEL"
#echo "BOTTOM_FILE: $BOTTOM_FILE"
#echo "GRID_FILE: $GRID_FILE"
echo "SWAN_EXECUTABLE: $SWAN_EXECUTABLE"
#echo "SNL_POWER_FILE: $SNL_POWER_FILE"

# Set the path to the Python script
PYTHON_EVALUATION="output_evaluation.py"
PYTHON_PRE_TO_AVG="pre_to_avg.py"

#get the name of this file
RUNNING_FILE=$(basename "$0")

SOURCE_FILES=("$BOTTOM_FILE" "$GRID_FILE" "$SNL_POWER_FILE" "$SWAN_EXECUTABLE" "$ST_FILE" "$PYTHON_EVALUATION" "$PYTHON_PRE_TO_AVG" "$RUNNING_FILE")
#echo "Source files: ${SOURCE_FILES[@]}"

# Get all .swn files in the current directory
SWAN_FILES=(*.swn)


# Check if preview.csv exists in the current directory
if [ -f "preview.csv" ]; then
    echo "File exists: preview.csv"
else
    # Create the base preview.csv file with the header
    echo "sea_state,array,average_Hs,average_power,standard_deviation,coeff_variation,power_range,power_peak_to_mean,power_mean_to_minimum" > "preview.csv"
    echo "File created: preview.csv"
fi


# Semaphore function to limit parallel processes
semaphore() {
    while [ "$(jobs -p | wc -l)" -ge "$MAX_PARALLEL" ]; do
        sleep 1
    done
}

# Function to clean up temporary directory
cleanup_temp_dir() {
    if [ -d "$TEMP_DIR" ]; then
        rm -r "$TEMP_DIR"
        #echo "Temporary directory $TEMP_DIR deleted."
    fi
    if [ -f "$TMP_OUTPUT_UNIQUE" ]; then
        rm "$TMP_OUTPUT_UNIQUE"
        #echo "Temporary preview $TMP_OUTPUT_UNIQUE deleted."
    fi
}


# Create a temporary directory, copy files, and run SWAN for each .swn file
for FILE in "${SWAN_FILES[@]}"; do
    semaphore
    (
        TEMP_DIR=$(mktemp -d)
        cd "$TEMP_DIR" || exit

        for SRC_FILE in "${SOURCE_FILES[@]}"; do
            #echo "$SRC_FILE"
            if [ -e "$CURRENT_DIR/$SRC_FILE" ]; then
                cp "$CURRENT_DIR/$SRC_FILE" .
                
            else
                echo "Error: $SRC_FILE not found in the main directory."
                exit 1
            fi
        done

        # Copy the .swn file
        cp "$CURRENT_DIR/$FILE" .
        # Copy the .ga file
        cp "$CONFIG_FILE" .
        
        if [ -x "$CURRENT_DIR/$SWAN_EXECUTABLE" ]; then
            echo "Running SWAN for $FILE"
            time "$CURRENT_DIR/$SWAN_EXECUTABLE" -input "$FILE" "$TYPE" "$NUM_PROCESSES"
        else
            echo "Error: SWAN executable not found at $CURRENT_DIR/$SWAN_EXECUTABLE."
            exit 1
        fi

        # Wait for SWAN process to finish before running Python script
        wait

        # Run Python script
        if [ -x "$CURRENT_DIR/$PYTHON_EVALUATION" ]; then
            echo "Running $PYTHON_EVALUATION for $FILE in $(pwd)"
            python3 "$CURRENT_DIR/$PYTHON_EVALUATION" 
            wait
                        
        else
            echo "Error: Python script not found at $CURRENT_DIR/$PYTHON_EVALUATION."
            exit 1
        fi

        # Save output to a temporary file with a unique name
        TMP_OUTPUT_UNIQUE=$(mktemp)
        tail -n +2 preview.csv > "$TMP_OUTPUT_UNIQUE"

        # Create a lock file and write the output
        lockfile="$CURRENT_DIR/preview_$$.lock"
        (
            flock -x 200

            # Append content to preview.csv in CURRENT_DIR
            cat "$TMP_OUTPUT_UNIQUE" >> "$CURRENT_DIR/preview.csv"

            # Remove the lock file
            rm "$lockfile"
        ) 200>"$lockfile"

        # Clean up temporary directory when the loop is finished
        cleanup_temp_dir

    ) &
done

# Wait for all background processes to finish
wait

# Run Python for weighted array average
if [ -x "$CURRENT_DIR/$PYTHON_PRE_TO_AVG" ]; then
    echo "Running $PYTHON_PRE_TO_AVG in $CURRENT_DIR"
    python3 "$CURRENT_DIR/$PYTHON_PRE_TO_AVG"
else
    echo "Error: $PYTHON_PRE_TO_AVG not found at $CURRENT_DIR/$PYTHON_PRE_TO_AVG."
    exit 1
fi

echo "SWAN simulations for $CURRENT_DIR completed."
