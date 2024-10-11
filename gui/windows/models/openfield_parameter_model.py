import utils.CobelSpikeOriginalParams as OriginalParameters
import copy
import os
import json

class OpenfieldParams:
    def __init__(self):
        # Initialize the simulation parameters from an external source or with defaults
        self.reset_to_original_params()

        if os.path.exists("parameter_sets/current_parameter/network_params_spikingnet.json"):
            with open("parameter_sets/current_parameter/network_params_spikingnet.json") as json_file:
                self.network_params = json.load(json_file)
                self.network_param_dicts = [self.network_params]

        if os.path.exists("parameter_sets/current_parameter/env_params.json"):
            with open("parameter_sets/current_parameter/env_params.json") as json_file:
                self.env_params = json.load(json_file)


    def reset_to_original_params(self):
        self.sim_params = OriginalParameters.getAllSimParameters()
        self.network_params = OriginalParameters.getAllNetworkParameters()
        self.network_param_dicts = [self.network_params]
        self.env_params = OriginalParameters.getAllEnvParameters()

    def reset_network_params(self):
        self.network_params = OriginalParameters.getOriginalNetworkParameters()
        self.network_param_dicts = [self.network_params]

        if os.path.exists("parameter_sets/current_parameter/network_params_spikingnet.json"):
            os.remove("parameter_sets/current_parameter/network_params_spikingnet.json")

    def reset_env_params(self):
        self.env_params = OriginalParameters.getOriginalEnvParameters()

        if os.path.exists("parameter_sets/current_parameter/env_params.json"):
            os.remove("parameter_sets/current_parameter/env_params.json")

    def reset_sim_params(self):
        self.sim_params = OriginalParameters.getAllSimParameters()

    def get_env_params(self):
        return self.env_params
    
    def get_network_dicts(self):
        return self.network_param_dicts
    
    def get_network_params(self):
        return self.network_params
    
    def get_sim_params(self):
        return self.sim_params
    
    def set_env_params(self, env:dict):
        self.env_params = env

    def set_network_dicts(self, network_dicts:list):
        self.network_param_dicts = network_dicts

    def getAllNetworkParameterTypes(self):

        network_params_dict = OriginalParameters.getAllNetworkParameters()
        network_param_types_dict = copy.deepcopy(network_params_dict)

        self.recursive_items(network_param_types_dict)
        return network_param_types_dict
    
    def getAllEnvParameterTypes(self):

        env_params_dict = OriginalParameters.getAllEnvParameters()
        env_param_types_dict = copy.deepcopy(env_params_dict)

        self.recursive_items(env_param_types_dict)
        return env_param_types_dict
    
    def recursive_items(self, dictionary):
        for key, value in dictionary.items():
            if type(value) is dict:
                self.recursive_items(value)
            else:
                dictionary[key] = type(value)