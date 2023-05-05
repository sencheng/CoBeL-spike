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
import timeit
from network import Network
from analysis import Analysis

start = timeit.default_timer()

with open('network_params.json', 'r') as fl:
    net_dict = json.load(fl)

with open('sim_params.json', 'r') as fl:
    sim_dict = json.load(fl)

''' Single test run    
#net_dict['place']['spatial_prop']['max_fr']=200
#net_dict['place']['trg_conn']['syn_spec']['weight']=30
net = Network(sim_dict=sim_dict, net_dict=net_dict)
net.setup()
#net.test_action_neurons_multinp(inp_fr=100, inp_w=30)
net.simulate(10000)
an = Analysis(net)
an.read_spike_files()
an.plot_raster_avgfr()
plt.show()
#'''

''' Single run: rat's path
net = Network(sim_dict=sim_dict, net_dict=net_dict)
net.setup()
for i in range(20):
    print('Current virtual rat\'s position x=%.2f, y=%.2f' %(net.curr_pos[0],
                                                             net.curr_pos[1]))
    net.simulate(50)
    net.navigate()
an = Analysis(net)
an.plot_rat_pos()
    
#'''

# ''' Place cell firing rate and place-to-action projection weights
# Parameter impact: rat's trajectory
fr_vec = np.arange(70, 85, 10)  # np.array([80])#
w_vec = np.arange(30, 55, 5)  # np.array([30])#
w_plus = 100
w_minus = -300

combvec = np.array(np.meshgrid(fr_vec, w_vec)).reshape(2, -1)

for idx in range(combvec.shape[1]):
    m_fr = combvec[0, idx]
    w = combvec[1, idx]
    net_dict['place']['spatial_prop']['max_fr'] = m_fr
    net_dict['place']['trg_conn']['syn_spec']['weight'] = w
    net_dict['action']['orientation_sel_dic']['winh'] = w_minus
    net_dict['action']['orientation_sel_dic']['wexc'] = w_plus
    net = Network(sim_dict=sim_dict, net_dict=net_dict)
    net.setup()
    # net.test_action_neurons_multinp(inp_fr=100, inp_w=30)
    for i in range(500):
        #        print('Current virtual rat\'s position x=%.2f, y=%.2f' %(net.curr_pos[0],
        #                                                                 net.curr_pos[1]))
        net.simulate(20)
        net.navigate()
    an = Analysis(net)
    an.read_spike_files()
    an.plot_raster_avgfr(fl_suffix='-%s-%s' % (m_fr, w))
    an.plot_rat_pos_arrow(fl_suffix='-%s-%s' % (m_fr, w))
# '''

''' Random seed impact on the explatory behavior
for rng_id in range(1, 11):
    sim_dict['master_rng_seed'] = rng_id
    net = Network(sim_dict=sim_dict, net_dict=net_dict)
    net.setup()
    #net.test_action_neurons_multinp(inp_fr=100, inp_w=30)
    for i in range(500):
#        print('Current virtual rat\'s position x=%.2f, y=%.2f' %(net.curr_pos[0],
#                                                                 net.curr_pos[1]))
        net.simulate(20)
        net.navigate()
    an = Analysis(net)
    an.read_spike_files()
    an.plot_raster_avgfr(fl_suffix='-%d' %(rng_id))
    an.plot_rat_pos_arrow(fl_suffix='-%d' %(rng_id))
#'''

''' Parameter impact
fr_vec = np.arange(50, 110, 10)#np.array([200])#
w_vec =np.array([100])#np.arange(10, 50, 5)#

combvec = np.array(np.meshgrid(fr_vec, w_vec)).reshape(2, -1)


for idx in range(combvec.shape[1]):
    m_fr = combvec[0, idx]
    w = combvec[1, idx]
    net_dict['place']['spatial_prop']['max_fr'] = m_fr
    net_dict['place']['trg_conn']['syn_spec']['weight'] = w
    net = Network(sim_dict=sim_dict, net_dict=net_dict)
    net.setup()
    #net.test_action_neurons_multinp(inp_fr=100, inp_w=30)
    net.simulate(10000)
    an = Analysis(net)
    an.read_spike_files()
    an.plot_raster_avgfr(fl_suffix='-%s-%s' %(m_fr, w))
    plt.show()
#'''

# populations = np.array(['place', 'action'])
# external_inputs = np.array(['poisson_generator', 'place'])
# targets = np.array(['action', None])
# exclusive_inp = [True, False]
# num_neurons = [[11, 11], 40]
# extent = np.array([[4, 4], None])
# model_neuron = np.array(['parrot_neuron', 'iaf_psc_exp']
stop = timeit.default_timer()
print('Time: {}'.format(stop - start))
