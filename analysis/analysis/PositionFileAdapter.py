#!/usr/bin/env python3

import os
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class PositionFileAdapter():

    def __init__(self, data_path, filename, fig_path=None):
        print("Starting processing agent's location using the file from adapters ...\n")
        self.file_path = os.path.join(data_path, filename)
        if not os.path.exists(self.file_path):
            print(
                'Could not find {}. Check whether the data file exists' + ' and whether the path to the file is correct!'.format(
                    filename))

        if fig_path is not None:
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)

        self.fig_path = fig_path
        with open('../openfield/network_params_spikingnet.json', 'r') as fl:
            tmp_dict = json.load(fl)

        self.env_extent = tmp_dict['place']['spatial_prop']
        self.fl_type = filename.split('.')[-1]

    def read_pos_file(self):
        """
        Reads the position file at `self.file_path` and stores the values at `self.pos_time` and `self.pos_xy`.
        """
        print('\tposfile_adapter: Reading file: {} ...\n'.format(self.file_path))
        tmp = pd.read_csv(self.file_path, sep='\t')
        self.pos_time = tmp.time.values * 1000
        self.pos_xy = tmp.values[:, 1:]

    def extract_trs_locfl(self):
        print('\tposfile_adapter: Extracting trials from the position file: {} ...\n'.format(self.file_path))

        init_pos = np.where((self.pos_xy[:, 0] == 0) & (self.pos_xy[:, 1] == 0))[0]

        init_pos_ex_begin = self.pos_time[init_pos[1:][np.diff(init_pos) > 1]]

        end_times = np.hstack((init_pos_ex_begin, self.pos_time[-1]))
        start_times = np.hstack((0.0, init_pos_ex_begin))

        return start_times, end_times

    def plot_rat_pos(self, fl_suffix='line', frm='pdf', vert_lim=(-1.2, 1.2), horiz_lim=(-1.2, 1.2)):
        print("\tposfile_adapter: Plotting (line) agent's position ...\n")
        self.horiz_lim = horiz_lim
        self.vert_lim = vert_lim
        if not hasattr(self, 'pos_data'):
            self.read_pos_file()

        st, end = self.extract_trs_locfl()
        for ind in range(st.size):
            sel_time = (self.pos_time >= st[ind]) & (self.pos_time < end[ind])
            rat_pos = self.pos_xy[sel_time]
            fig, ax = plt.subplots()
            ax.plot(rat_pos[:, 0], rat_pos[:, 1], color='blue', label='traversed path')
            ax.plot(rat_pos[0, 0], rat_pos[0, 1], marker='o', color='green', label='initial position')
            ax.plot(rat_pos[-1, 0], rat_pos[-1, 1], marker='o', color='red', label='final position')
            ax.set_xlabel('x (a.u.)')
            ax.set_ylabel('y (a.u.)')
            ax.set_xlim(self.horiz_lim)
            ax.set_ylim(self.vert_lim)
            ax.legend()
            fig.savefig(os.path.join(self.fig_path, 'ratpos-{:s}-tr{:d}.{}'.format(fl_suffix, ind + 1, frm)),
                        format=frm)
            plt.close(fig)
