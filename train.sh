#!/bin/bash

# This is a comment

# Define variables
name="Kang_lin"

# Get current time
current_time=$(date +"%T")
echo "Current time: $current_time"

# Print a message
echo "Hello, $name! Your Scripts start training"

# Print current directory
echo "Current directory: $(pwd)"
# Change directory
cd /rsrch5/home/gen_med_onc_rsch/khsieh/single_cell_pdx/code
echo "Current directory: $(pwd)"
# List files in the folder
# Make .sh files executable
chmod +x /rsrch5/home/gen_med_onc_rsch/khsieh/single_cell_pdx/code/*.sh

# Make .py files executable
chmod +x /rsrch5/home/gen_med_onc_rsch/khsieh/single_cell_pdx/code/*.py
# Run Python code using accelerate launch
yaml_file=/rsrch5/home/gen_med_onc_rsch/khsieh/single_cell_pdx/code/igtp_config.yaml

# Check if accelerate is installed
if ! command -v accelerate launch &> /dev/null; then
    echo "Error: accelerate is not installed. Please install it before running the script."
    exit 1
fi

#nohup accelerate launch --config_file acc.ymal imdb.py > output.log 2>&1 &
# Install the required package
#pip install evaluate

nohup python iGTP_Kfold_train.py --config $yaml_file > output.log 2>&1 &

# Wait for the training to finish
wait

# Get finished time
finished_time=$(date +"%T")
echo "Finished time: $finished_time"

# Print a message
echo "Hello, $name! Your Scripts finished"

# Calculate time duration in hours
start_time=$(date -d "$current_time" +%s)
end_time=$(date -d "$finished_time" +%s)
duration=$((end_time - start_time))
duration_mins=$((duration / 60))
echo "Time duration: $duration_mins mins"

# Exit the script
exit 0