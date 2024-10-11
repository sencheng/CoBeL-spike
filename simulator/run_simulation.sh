#!/bin/bash

: '
asa
'

CURRENT_DIR=$(dirname "$0")

RESET_PARAMS=false
SKIP_TRIAL_GEN=false

SEED1=$1
SEED2=$2
shift 2

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --skip-trial-gen) SKIP_TRIAL_GEN=true ;;
        --reset-params) RESET_PARAMS=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

PARAM_FOLDER="parameter_sets"
CURRENT_PARAMS="current_parameter"
ORIG_PARAMS="original_params"

# names of the dynamic parameter files to be used during the simulation
NET_PARAM_NAME="$PARAM_FOLDER/$CURRENT_PARAMS/network_params_spikingnet.json"
SIM_PARAM_NAME="$PARAM_FOLDER/$CURRENT_PARAMS/sim_params.json"
ENV_PARAM_NAME="$PARAM_FOLDER/$CURRENT_PARAMS/env_params.json"
ANALYSIS_CONFIG_NAME="$PARAM_FOLDER/$CURRENT_PARAMS/analysis_config.json"
MUS_PARAM_NAME="$PARAM_FOLDER/$CURRENT_PARAMS/nest_openfield.music"

# names of the static parameter files used as a reference
ORIG_NET_PARAM_NAME="$PARAM_FOLDER/$ORIG_PARAMS/original_network_params_spikingnet.json"
ORIG_SIM_PARAM_NAME="$PARAM_FOLDER/$ORIG_PARAMS/original_sim_params.json"
ORIG_ENV_PARAM_NAME="$PARAM_FOLDER/$ORIG_PARAMS/original_env_params.json"
ORIG_ANALYSIS_CONFIG_NAME="$PARAM_FOLDER/$ORIG_PARAMS/original_analysis_config.json"
ORIG_MUS_PARAM_NAME="$PARAM_FOLDER/$ORIG_PARAMS/original_nest_openfield.music"

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


if [ ! -d "$PARAM_FOLDER/$CURRENT_PARAMS" ]; then
  mkdir -p "$PARAM_FOLDER/$CURRENT_PARAMS"
fi

# if the parameter files do not exist then copy to original parameters
if [ ! -f $CURRENT_DIR/$NET_PARAM_NAME ]; then
    echo -e '\033[0;33mNo network parameter file found. Copying original...\033[0m'
    cp $CURRENT_DIR/$ORIG_NET_PARAM_NAME $CURRENT_DIR/$NET_PARAM_NAME 
fi

if [ ! -f $CURRENT_DIR/$SIM_PARAM_NAME ]; then
    echo -e '\033[0;33mNo simulation parameter file found. Copying original...\033[0m'
    cp $CURRENT_DIR/$ORIG_SIM_PARAM_NAME $CURRENT_DIR/$SIM_PARAM_NAME 
fi

if [ ! -f $CURRENT_DIR/$ENV_PARAM_NAME ]; then
    echo -e '\033[0;33mNo environment parameter file found. Copying original...\033[0m'
    cp $CURRENT_DIR/$ORIG_ENV_PARAM_NAME $CURRENT_DIR/$ENV_PARAM_NAME 
fi

if [ ! -f $CURRENT_DIR/$MUS_PARAM_NAME ]; then
    echo -e '\033[0;33mNo music config file found. Copying original...\033[0m'
    cp $CURRENT_DIR/$ORIG_MUS_PARAM_NAME $CURRENT_DIR/$MUS_PARAM_NAME 
fi

if [ ! -f $CURRENT_DIR/$ANALYSIS_CONFIG_NAME ]; then
    echo -e '\033[0;33mNo analysis config file found. Copying original...\033[0m'
    cp $CURRENT_DIR/$ORIG_ANALYSIS_CONFIG_NAME $CURRENT_DIR/$ANALYSIS_CONFIG_NAME
fi



for (( i=$SEED1; i<=$SEED2; i++ ))
do
    if [ "$SKIP_TRIAL_GEN" = false ]; then
        python simulation_files/trial_generator.py
    fi
    python simulation_files/update_music_file.py    # Make sure whether the updated parameters
                                                    # are also updated in the music conf. file

    ports="$(grep -Eo '[0-9]{1,}' hostfile)"    # There is only one number in the hostfile, so this 
                                                # regex should always retrieve the correct number of ports
    
    echo "running simulation $i from $SEED1 to $SEED2"

    python simulation_files/update_random_seed.py $i    # Changing random seed for each trial
    python simulation_files/create_dirs.py              # Create the data directory    
    python simulation_files/create_grid_positions.py    # Create place cells selectivity map
    python simulation_files/copy_par_files.py           # Copy parameter files to the simulation
                                                        # output directory for later comparisons
    gymz-controller gym $ENV_PARAM_NAME &
    gym_pid=$(echo $!)
    mpirun --hostfile hostfile -np $ports music $MUS_PARAM_NAME
    kill $gym_pid

    rm -f grid_pos.json
    rm -f report_det.dat
    rm -f reward_spikes.dat
    rm -f reward_value.dat
    rm -rf ./of


    python ../analysis/analysis_runner.py
done


if [ $RESET_PARAMS = "true" ]; then
    echo Reseting to original parameters...
    ./change_params.sh original_parameters
fi
