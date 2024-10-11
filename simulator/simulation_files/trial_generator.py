#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 16:19:42 2023

@author: ghazibhy
"""

import pandas as pd
import json

# Define parameters
with open('./parameter_sets/current_parameter/sim_params.json', 'r') as fl:
    tmp_dict = json.load(fl)

total_trial_num = tmp_dict['max_num_trs']
trial_param_dict = tmp_dict['trial_params']

# Prepare temporary dictionary for trial parameters
tmp = {
    'trial_num': range(1, total_trial_num + 1)
}

"""
This code translates the desired parameter sets into an explicit list of 
trial-by-trial parameters. It divides the total trials as evenly as possible
by each parameter set, which may have different lengths.
"""
for key, param in trial_param_dict.items():
    n = len(param)
    if n > total_trial_num:
        print(f'Too many parameters given! {n} trial parameters found for {total_trial_num} trials.')
        n = total_trial_num

    q = total_trial_num // n
    r = total_trial_num % n
    indexer = [x for x in range(n) for i in range(q)]
    
    tmp[key] = [param[i] for i in indexer]
    
    if r > 0:
        print(f'Trial parameters are not evenly distributed among trials! {q} parameter sets were given for {total_trial_num} trials.')
        tmp[key] += [param[-1] for i in range(r)]

# Calculate total simulation time
total_simtime = sum(tmp['max_tr_dur'], 1000)
tmp_dict['simtime'] = total_simtime

# Update the sim_params.json file
with open('./parameter_sets/current_parameter/sim_params.json', 'w+') as fl:
    json.dump(tmp_dict, fl, indent=2)

# Create DataFrame of trials
trials = pd.DataFrame(tmp)

# Write DataFrame to CSV file
trials.to_csv('./parameter_sets/current_parameter/trials_params.dat', sep="\t", index=False)
