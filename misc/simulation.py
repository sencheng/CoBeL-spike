#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 11:34:36 2019

@author: mohagmnr
"""

# import nest
# import nest.topology as topp
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from network import Network
from analysis import Analysis

with open('network_params.json', 'r') as fl:
    net_dict = json.load(fl)

with open('sim_params.json', 'r') as fl:
    sim_dict = json.load(fl)

net = Network(sim_dict=sim_dict, net_dict=net_dict)
net.setup()
net.simulate(1000)

data_obj = Analysis(net)
data_obj.read_spike_files()
data_obj.plot_raster_avgfr()

# populations = np.array(['place', 'action'])
# external_inputs = np.array(['poisson_generator', 'place'])
# targets = np.array(['action', None])
# exclusive_inp = [True, False]
# num_neurons = [[11, 11], 40]
# extent = np.array([[4, 4], None])
# model_neuron = np.array(['parrot_neuron', 'iaf_psc_exp']
