#!/usr/bin/env python3

"""
This Python script ensures the existence of the directories needed to write the 
simulation data. It reads from "sim_params.json". The reason we need this is 
that some components need to access the directory before being created for the 
NEST simulation kernel.
"""

import os
import json
import time

# This parameter defines how long execution will pause to allow the user to
# cancel in the event that data would be overwritten
waittime = 0

# Load simulation parameters
with open('parameter_sets/current_parameter/sim_params.json', 'r') as fl:
    sim_dict = json.load(fl)

master_rng_seed = str(sim_dict['master_rng_seed'])
data_dir = sim_dict['data_path']
main_data_dir = os.path.join(*data_dir.split('/')[:-1])
fig_dir = os.path.join(main_data_dir, 'fig-' + master_rng_seed)

# Create necessary directories
for target_dir in [fig_dir, data_dir]:
    try:
        print(f"Creating {target_dir} if it doesn't exist ...")
        os.makedirs(target_dir)
    except FileExistsError:
        print('\033[33m')  # Change text to yellow
        print(f"{target_dir} already exists! Existing data will be overwritten unless execution is halted!")
        print(f'Execution will continue after {waittime} seconds...')
        print('\033[0m')  # Reset text color
        time.sleep(waittime)
        print('Continuing...')
