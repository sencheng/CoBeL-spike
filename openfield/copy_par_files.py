#!/usr/bin/env python3

import os
from shutil import copyfile
import json

with open('sim_params.json', 'r') as fl:
    sim_dict = json.load(fl)
master_rng_seed = str(sim_dict['master_rng_seed'])
data_dir = sim_dict['data_path']
main_data_dir = os.path.join(*data_dir.split('/')[0:-1])
fig_dir = os.path.join(main_data_dir, 'fig-' + master_rng_seed)

par_fl_names = ['network_params_spikingnet.json',
                'sim_params.json',
                'grid_pos.json']

for target_dir in [main_data_dir, fig_dir]:
    for fl_name in par_fl_names:
        print('Copying {} to {} ...'.format(fl_name, target_dir))
        copyfile(os.path.join('./', fl_name),
                 os.path.join(target_dir, fl_name))
