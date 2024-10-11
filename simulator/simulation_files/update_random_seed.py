#!/usr/bin/env python3

"""
This script gets the ID of a random number generator as a system input argument
and updates the corresponding parameters in the corresponding parameter file. 
It also creates/updates another JSON file to store all data paths for a 
simulation session to facilitate more offline analysis.
"""

import sys
import json
import os

# Updating the parameter file
with open('parameter_sets/current_parameter/sim_params.json', 'r') as fl:
    sim_dict = json.load(fl)

des_seed = int(sys.argv[1])

sim_dict['master_rng_seed'] = des_seed
temp_fl = sim_dict['data_path'].split('agent')
sim_dict['data_path'] = temp_fl[0] + 'agent' + str(des_seed)

with open('parameter_sets/current_parameter/sim_params.json', 'w') as fl:
    json.dump(sim_dict, fl, indent=2)

parent_data_dir = os.path.join(*sim_dict['data_path'].split('/')[:-1])

os.makedirs(parent_data_dir, exist_ok=True)

# Creating or updating the JSON file that stores all data paths within a session
fl_path = os.path.join(parent_data_dir, 'data_paths.json')

if os.path.exists(fl_path):
    with open(fl_path, 'r') as fl:
        content = json.load(fl)
    content['data_path'].append(sim_dict['data_path'])
else:
    content = {'data_path': [sim_dict['data_path']]}

with open(fl_path, 'w') as fl:
    json.dump(content, fl, indent=2)
