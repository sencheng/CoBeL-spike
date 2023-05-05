#!/usr/bin/env python3

from analysis import SpikefileNest as spk
from analysis import PositionFile as pos
from analysis import ActionVector as act
from analysis import Weight as W
from analysis import MultipleAnalysis as MA
from analysis import BehavioralPerformance as BP
from analysis import RepresentationVisualization as RV

import matplotlib.pyplot as plt
import numpy as np
import json
import os

with open('../openfield/sim_params.json', 'r') as fl:
    net_dict = json.load(fl)

seed = net_dict['master_rng_seed']
fig_p = os.path.join(*net_dict['data_path'].split('/')[0:-1], 'fig-{}/'.format(seed))

repres_vis = RV(os.path.join(*net_dict['data_path'].split('/')[0:-1]), flname='grid_pos.json', fig_path=fig_p)
repres_vis.plot_firing_maps(frm='png')
