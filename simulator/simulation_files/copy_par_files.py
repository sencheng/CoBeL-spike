#!/usr/bin/env python3

import os
from shutil import copyfile
import json

# Load simulation parameters
with open('parameter_sets/current_parameter/sim_params.json', 'r') as fl:
    sim_dict = json.load(fl)

# Extract relevant parameters
master_rng_seed = str(sim_dict['master_rng_seed'])
data_dir = sim_dict['data_path']
main_data_dir = os.path.join(*data_dir.split('/')[:-1])

# Define target directories
fig_dir = os.path.join(main_data_dir, 'fig-' + master_rng_seed)
agent_dir = os.path.join(main_data_dir, 'agent' + master_rng_seed)
target_directories = [main_data_dir, agent_dir]

# Define files to be copied
par_fl_names = [
    'parameter_sets/current_parameter/network_params_spikingnet.json',
    'parameter_sets/current_parameter/sim_params.json',
    'parameter_sets/current_parameter/env_params.json',
    'grid_pos.json',
    'parameter_sets/current_parameter/trials_params.dat'
]

# Copy files to target directories
for target_dir in target_directories:
    for fl_name in par_fl_names:
        fl_basename = os.path.basename(fl_name)
        print(f'Copying {fl_name} to {target_dir} ...')
        copyfile(os.path.join('./', fl_name), os.path.join(target_dir, fl_basename))
