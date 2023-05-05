#!/usr/bin/env python3

'''
This python script ensures existance of the directories needed to write the 
simulation data in it. It reads from "sim_params.json". The reason we need this
is that some components need to access the directory before being created for
NEST simulation kernel.
'''

import os
import json

with open('sim_params.json', 'r') as fl:
    sim_dict = json.load(fl)

master_rng_seed = str(sim_dict['master_rng_seed'])
data_dir = sim_dict['data_path']
main_data_dir = os.path.join(*data_dir.split('/')[0:-1])
fig_dir = os.path.join(main_data_dir, 'fig-' + master_rng_seed)

for target_dir in [fig_dir, data_dir]:
    print("Creating {} if doesn't exist ...".format(target_dir))
    os.makedirs(target_dir)
