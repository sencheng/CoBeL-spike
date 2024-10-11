#!/bin/bash

CURRENT_DIR=$(dirname "$0")
PARAM_FOLDER="parameter_sets"
CURRENT_PARAMS="$PARAM_FOLDER/current_parameter"

if [ "$#" -ne 1 ]; then
    echo "Error: change_parameters.sh requires exactly one parameter."
    echo "Usage: $0 <parameter>"
    exit 1
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
