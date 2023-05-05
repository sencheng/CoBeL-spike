#!/usr/bin/env python3

'''
This file gets the ID of random number generator as system input argument and
updates the corresponding parameters in the corresponding parameter file. It
also creates/updates another json file to store all data paths for a simulation
session to faciliate more offline analysis.
'''

import sys
import json
import os

# Updating the parameter file

with open('sim_params.json', 'r') as fl:
    net_dict = json.load(fl)
des_seed = int(sys.argv[1])

net_dict['master_rng_seed'] = des_seed
temp_fl = net_dict['data_path'].split('tr')
net_dict['data_path'] = temp_fl[0] + 'tr' + str(des_seed)

with open('sim_params.json', 'w') as fl:
    json.dump(net_dict, fl, indent=2)

parent_data_dir = os.path.join(*net_dict['data_path'].split('/')[0:-1])

os.makedirs(parent_data_dir, exist_ok=True)

# Creating json file that stores all data paths within a session

fl_path = os.path.join(parent_data_dir, 'data_paths.json')

if os.path.exists(fl_path):

    with open(fl_path, 'r') as fl:
        content = json.load(fl)
    content['data_path'].append(net_dict['data_path'])
else:
    content = {'data_path': [net_dict['data_path']]}
with open(fl_path, 'w') as fl:
    json.dump(content, fl)
