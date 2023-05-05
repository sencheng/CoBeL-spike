#!/bin/bash

CURRENT_DIR=$(dirname "$0")

# names of the dynamic parameter files to be used during the simulation
NET_PARAM_NAME="network_params_spikingnet.json"
SIM_PARAM_NAME="sim_params.json"
OPF_PARAM_NAME="openfield.json"

# names of the static parameter files used as a reference
ORIG_NET_PARAM_NAME="original_network_params_spikingnet.json"
ORIG_SIM_PARAM_NAME="original_sim_params.json"
ORIG_OPF_PARAM_NAME="original_openfield.json"

# delete the just used parameter files
rm -v $CURRENT_DIR/$NET_PARAM_NAME $CURRENT_DIR/$SIM_PARAM_NAME $CURRENT_DIR/$OPF_PARAM_NAME

# if the original parameter files exist then copy them to the current folder else print an error and exit
if [ -f $CURRENT_DIR/original_params/$ORIG_NET_PARAM_NAME ]; then
    cp -v  $CURRENT_DIR/original_params/$ORIG_NET_PARAM_NAME $CURRENT_DIR/$NET_PARAM_NAME
else
    echo Could not find original network parameter file.
fi

if [ -f $CURRENT_DIR/original_params/$ORIG_NET_PARAM_NAME ]; then
    cp -v  $CURRENT_DIR/original_params/$ORIG_SIM_PARAM_NAME $CURRENT_DIR/$SIM_PARAM_NAME
else
    echo Could not find original simulation parameter file.
fi

if [ -f $CURRENT_DIR/original_params/$ORIG_NET_PARAM_NAME ]; then
    cp -v  $CURRENT_DIR/original_params/$ORIG_OPF_PARAM_NAME $CURRENT_DIR/$OPF_PARAM_NAME
else
    echo Could not find original openfield parameter file.
fi

