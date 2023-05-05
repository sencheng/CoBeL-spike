#!/usr/bin/env python3

from analysis import SpikefileNest as spk
from analysis import PositionFileAdapter as pos
from analysis import ActionVector as act
from analysis import Weight as W
from analysis import MultipleAnalysis as MA
from analysis import BehavioralPerformance as BP

import matplotlib.pyplot as plt
import numpy as np
import json
import os
import copy
import sys

performed_analysis = {}

main_data_path = '../data/nestsim/20-07-23-W60-rewparupdate-rep4-100s'
with open(os.path.join(main_data_path, 'data_paths.json'), 'r') as fl:
    data_path_dic = json.load(fl)

# for data_path in data_path_dic['data_path']:

data_path = data_path_dic['data_path'][int(sys.argv[1])]

p_a_list = []

print('Processing data in {}'.format(data_path))

seed = int(data_path.split('tr')[-1])
fig_p = os.path.join(*data_path.split('/')[0:-1], 'fig-{}/'.format(seed))
'''
spk_obj = spk(data_path=net_dict['data_path'], filename='place-0.gdf',
              fig_path=fig_p)
spk_obj.read_spike_files()
spk_obj.plot_avg_fr(fig_filename='place', frm='png')
'''
spk_obj = spk(data_path=data_path, filename='action-0.gdf', fig_path=fig_p)
spk_obj.read_spike_files()
spk_obj.plot_avg_fr(fig_filename='action', frm='png')
p_a_list.append('fr_action')
'''
pos_obj = pos(data_path='../openfield', filename='report_det.dat',
              fig_path=fig_p)
pos_obj.read_pos_file()
# pos_obj.save_movie_hr_highs_hdf()
pos_obj.plot_rat_pos(frm='png')
# pos_obj.plot_rat_pos_arrow(frm='png')
# pos_obj.plot_dir_dist_nonzero(frm='png')
'''
pos_obj = pos(data_path=data_path, filename='agents_location.dat', fig_path=fig_p)
pos_obj.plot_rat_pos(frm='png')
p_a_list.append('locations')
# # pos_obj.plot_pos_action(frm='png', path_dic={'data_path': net_dict['data_path'],
# #                                               'filename': 'action-0.gdf'})

# # act_obj = act(data_path='../openfield', filename='action_traces.dat',
# #               fig_path=fig_p)
# # act_obj.plot(fig_filename='action_vecs')

w_obj = W(data_path=data_path, filename='place-0.csv', fig_path=fig_p)
w_obj.read_files()
w_obj.get_times_from_DA(dop_flname='dopamine-0.gdf')
w_obj.plot_vector_field(frm='pdf')
p_a_list.append('weight_vectorfield')
# '''
animation = MA({'nest_data_path': data_path, 'other_data_path': data_path},
               {'place': 'place-0.gdf', 'action': 'action-0.gdf', 'weight': 'place-0.csv', 'loc': 'agents_location.dat',
                'dopamine': 'dopamine-0.gdf'}, fig_path=fig_p)
animation.animate_tr_by_tr_loc(frame_dur=50)
p_a_list.append('animation')
# '''
beh_obj = BP({'nest_data_path': data_path, 'other_data_path': data_path},
             {'loc': 'agents_location.dat', 'dopamine': 'dopamine-0.gdf'}, fig_path=fig_p)
beh_obj.get_performance(beh_obj.extract_trs_locfl)
beh_obj.plot_performance()
p_a_list.append('behavioral_performance')

performed_analysis[data_path] = copy.deepcopy(p_a_list)

with open(os.path.join(main_data_path, 'processed.json'), 'w') as fl:
    json.dump(performed_analysis, fl)
