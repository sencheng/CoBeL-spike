#!/bin/bash

: '
Performs the analysis for all agent-directories of the given path.
Can be used for updating the analysis_config and rerunning the analysis.
When starting the script input the path to the data folder like e.g. ./run_analysis.sh /../data
'

# Check if the provided path is a valid directory
if [ "$#" -ne 1 ]; then
  echo "Error: run_analysis.sh requires exactly one parameter."
  echo "Input one data directory, e.g. './data/test'"
  echo "Usage: $0 <parameter>"
  exit 1
fi

input_path=$1

dirs=$(find "$input_path" -type d -regex '.*/agent[0-9]+$')
for dir in $dirs; do
  num=$(echo $(basename "$dir") | grep -o '[0-9]\+')
  echo "Processing directory: $dir with num: $num"

  python ../analysis/analysis_runner.py --path "$dir" --num "$num"
done
