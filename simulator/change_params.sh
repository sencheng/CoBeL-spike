#!/bin/bash

: '
Changes following paramater files in parameter_sets/current_parameter: env_params.json, network_params_spikingnet.json, sim_params.json
With this script the parameters can be changed to a range of standard parameter sets, like e.g. openfield. 
For changing the parameters give one parameter_set from parameter_sets as input.
To e.g. change to the original_params, run: ./change_params.sh original_params
'

CURRENT_DIR=$(dirname "$0")
PARAM_FOLDER="parameter_sets"
CURRENT_PARAMS="$PARAM_FOLDER/current_parameter"

if [ "$#" -ne 1 ]; then
    echo "Error: change_parameters.sh requires exactly one parameter."
    echo "Input one of the directories from parameter_sets. The content will be copied to parameter_sets/current_parameter"
    echo "Usage: $0 <parameter>"
    exit 1
fi

if [ ! -d "$CURRENT_PARAMS" ]; then
  mkdir -p "$CURRENT_PARAMS"
fi

PARAM_SET="$PARAM_FOLDER/$1"

if [ "$1" = "original_params" ]; then
    cp "$PARAM_SET/original_analysis_config.json" "$CURRENT_PARAMS/analysis_config.json"
    cp "$PARAM_SET/original_env_params.json" "$CURRENT_PARAMS/env_params.json"
    cp "$PARAM_SET/original_network_params_spikingnet.json" "$CURRENT_PARAMS/network_params_spikingnet.json"
    cp "$PARAM_SET/original_sim_params.json" "$CURRENT_PARAMS/sim_params.json"
    cp "$PARAM_SET/original_nest_openfield.music" "$CURRENT_PARAMS/nest_openfield.music"
else
    cp "$PARAM_SET/env_params.json" "$CURRENT_PARAMS/env_params.json"
    cp "$PARAM_SET/network_params_spikingnet.json" "$CURRENT_PARAMS/network_params_spikingnet.json"
    cp "$PARAM_SET/sim_params.json" "$CURRENT_PARAMS/sim_params.json"
fi

exit 0
