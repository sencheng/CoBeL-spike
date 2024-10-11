#!/bin/bash

# Check if a path is provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <path>"
  exit 1
fi

# Get the provided path
input_path=$1

# Check if the provided path is a valid directory
if [ ! -d "$input_path" ]; then
  echo "Error: $input_path is not a valid directory"
  exit 1
fi

# Iterate over each directory matching the pattern fig-x
for dir in "$input_path"/fig-[0-9]*; do
  # Check if it is indeed a directory
  if [ -d "$dir" ]; then
    # Execute the Python script in the directory
    echo "Processing directory: $dir"

    # Extract the number from the directory name
    dir_name=$(basename "$dir")
    num=$(echo "$dir_name" | grep -o '[0-9]\+')

    echo "$dir_name"
    echo "$num"


    # Construct the new data path
    new_data_path="${input_path}/agent${num}"

    python simulation_files/change_sim_params.py "$new_data_path" "$num"
    python ../analysis/analysis_runner.py
  fi
done
