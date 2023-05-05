#!/bin/bash

CURRENT_DIR=$(dirname "$0")

RESET_PARAMS=false

# names of the dynamic parameter files to be used during the simulation
NET_PARAM_NAME="network_params_spikingnet.json"
SIM_PARAM_NAME="sim_params.json"
OPF_PARAM_NAME="openfield.json"

# names of the static parameter files used as a reference
ORIG_NET_PARAM_NAME="original_network_params_spikingnet.json"
ORIG_SIM_PARAM_NAME="original_sim_params.json"
ORIG_OPF_PARAM_NAME="original_openfield.json"

# activate python env and source nest vars keyword "source" cannot be used when running script with sh
PATHS_FILE=$CURRENT_DIR"/source_paths.sh"
ENV_PATH=$CURRENT_DIR"/../../packages/cobel/bin/activate"
VARS_PATH=$CURRENT_DIR"/../../packages/nest-simulator-2.20.0_install/bin/nest_vars.sh"


# test if the script is being run in a docker container
# test if the source_paths.sh exists in the same dir as this script
echo Looking for python enviornment and nest vars in the default locations
if [ -f $PATHS_FILE ]; then
    source $PATHS_FILE
# test if the enviornment exists and the nest_vars.sh exists then source them
elif [[ -f $ENV_PATH && -f $VARS_PATH ]]; then
    source $ENV_PATH
    source $VARS_PATH
else
    echo Could not find the enviornment or the nest vars in default locations.
    echo Assuming they are allready activated and continuing...
fi

# if the parameter files do not exist then copy to original parameters
if [ ! -f $CURRENT_DIR/$NET_PARAM_NAME ]; then
    echo -e '\033[0;33mNo network parameter file found. Copying original...\033[0m'
    cp original_params/$ORIG_NET_PARAM_NAME $CURRENT_DIR/$NET_PARAM_NAME 
fi

if [ ! -f $CURRENT_DIR/$SIM_PARAM_NAME ]; then
    echo -e '\033[0;33mNo simulation parameter file found. Copying original...\033[0m'
    cp original_params/$ORIG_SIM_PARAM_NAME $CURRENT_DIR/$SIM_PARAM_NAME 
fi

if [ ! -f $CURRENT_DIR/$OPF_PARAM_NAME ]; then
    echo -e '\033[0;33mNo openfield parameter file found. Copying original...\033[0m'
    cp original_params/$ORIG_OPF_PARAM_NAME $CURRENT_DIR/$OPF_PARAM_NAME 
fi



python update_music_file.py     # Make sure whether the updated parameters
                                # are also updated in the music conf. file
                                
ports="$(grep -Eo '[0-9]{1,}' hostfile)" # There is only one number in the hostfile, so this 
                                         # regex should always retrieve the correct number of ports

if [ $1 -le $2 ]; then

    echo "running simulation for trial $1 out of $2"
    rm -f grid_pos.json
    rm -rf ./of
    rm -f report_det.dat

    python update_random_seed.py $1 # Changing random seed for each trial
    python create_dirs.py           # Create the data directory    
    python create_grid_positions.py # Create place cells selectivity map
    python copy_par_files.py        # Copy parameter files to the simulation
                                    # output directory for later comparisons
    
    gymz-controller gym openfield.json &
    gym_pid=$(echo $!)
    mpirun --allow-run-as-root --hostfile hostfile -np $ports music nest_openfield.music
    # sleep 1m
    kill $gym_pid
    python ../analysis/analysis_runner.py
    
    ./run_sim_openfield_recursive.sh $(($1+1)) $2
fi

if [ $RESET_PARAMS = "true" ]; then
    echo Reseting to original parameters...
    ./reset_params.sh
fi

rm -f grid_pos.json
rm -rf ./of
rm -f report_det.dat

