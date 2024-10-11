import json
import os
import pandas as pd

def getAllEnvParameters():
    path = "parameter_sets/original_params/original_env_params.json"

    current_path = "parameter_sets/current_parameter/env_params.json"
    if os.path.isfile(current_path):
        path = current_path

    with open(path) as json_file:
        env_params_dict = json.load(json_file)

    return env_params_dict

def getOriginalEnvParameters():
    path = "parameter_sets/original_params/original_env_params.json"

    with open(path) as json_file:
        env_params_dict = json.load(json_file)

    return env_params_dict

def getEnvParameter(param):
    for key, value in recursive_items(getAllEnvParameters()):
        if param == key:
            return value
        
    return "not_found"

def getAllNetworkParameters():
    path = "parameter_sets/original_params/original_network_params_spikingnet.json"

    current_path = "parameter_sets/current_parameter/network_params_spikingnet.json"
    if os.path.isfile(current_path):
        path = current_path

    with open(path) as json_file:
        net_params_dict = json.load(json_file)

    return net_params_dict

def getOriginalNetworkParameters():
    path = "parameter_sets/original_params/original_network_params_spikingnet.json"

    with open(path) as json_file:
        net_params_dict = json.load(json_file)

    return net_params_dict

def getNetworkParameter(param):
    for key, value in recursive_items(getAllNetworkParameters()):
        if param == key:
            return value
        
    return "not_found"

def getAllSimParameters():
    dat_path = "parameter_sets/current_parameter/trials_params.dat"
    json_path = "parameter_sets/current_parameter/sim_params.json"
    original_json_path = "parameter_sets/original_params/original_sim_params.json"
    
    if os.path.exists(dat_path):
        # Load the original JSON file
        with open(original_json_path) as json_file:
            sim_params_dict = json.load(json_file)
        
        # Load the DAT file
        df = pd.read_csv(dat_path, sep="\t")
        
        # Group the dataframe by unique configurations
        grouped = df.groupby(list(df.columns.drop('trial_num')))
        
        # Create the trial_params dictionary
        trial_params = {
            'goal_shape': [],
            'goal_size1': [],
            'goal_size2': [],
            'goal_x': [],
            'goal_y': [],
            'start_x': [],
            'start_y': [],
            'max_tr_dur': []
        }
        
        nr_of_trials = []
        
        for _, group in grouped:
            for key in trial_params.keys():
                trial_params[key].append(group[key].iloc[0])
            nr_of_trials.append(len(group))
        
        # Update the sim_params_dict
        sim_params_dict['trial_params'] = trial_params
        sim_params_dict['max_num_trs'] = nr_of_trials 
        sim_params_dict['simtime'] = int(sum(x * y for x, y in zip(trial_params['max_tr_dur'], nr_of_trials)) + 1000)
        
    elif os.path.exists(json_path):
        with open(json_path) as json_file:
            sim_params_dict = json.load(json_file)
            # turn max_num_trs into a list
            sim_params_dict['max_num_trs'] = [sim_params_dict['max_num_trs']]
    else:
        with open(original_json_path) as json_file:
            sim_params_dict = json.load(json_file)
            sim_params_dict['max_num_trs'] = [sim_params_dict['max_num_trs']]

    return sim_params_dict

def deleteSimParameters():
    json_file = "parameter_sets/current_parameter/sim_params.json"
    dat_file = "parameter_sets/current_parameter/trials_params.dat"
    
    if os.path.exists(json_file):
        os.remove(json_file)   
    if os.path.exists(dat_file):
        os.remove(dat_file)

def getSimParameter(param):
    for key, value in recursive_items(getAllSimParameters()):
        if param == key:
            return value
        
    return "not_found"

def recursive_items(dictionary):
    for key, value in dictionary.items():
        if type(value) is dict:
            yield from recursive_items(value)
        else:
            yield (key, value)