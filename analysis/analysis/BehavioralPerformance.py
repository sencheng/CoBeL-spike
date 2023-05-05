#!/usr/bin/env python3

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from analysis import SpikefileNest
from analysis import PositionFileAdapter


class BehavioralPerformance():
    """ Computes some measures about the performance of the agent.
    """

    def __init__(self, path_dict={'nest_data_path': '', 'param_path': ''},
                 flname_dict={'tr_time_pos': '',
                              'param_file': {'sim': '',
                                             'net': ''}},
                 fig_path='', rec_all_beh_data=True,
                 beh_data_flname='complete_beh_data.dat',
                 reset_summary_file=False):
        with open(os.path.join(path_dict['param_path'],
                               flname_dict['param_file']['sim'])) as par_f:
            self.sim_params = json.load(par_f)

        with open(os.path.join(path_dict['param_path'],
                               flname_dict['param_file']['net'])) as par_f:
            self.net_params = json.load(par_f)

        self.data = pd.read_csv(os.path.join(path_dict['nest_data_path'],
                                             flname_dict['tr_time_pos']),
                                sep='\t')
        self.fig_path = fig_path
        if rec_all_beh_data:
            master_rng_seed = str(self.sim_params['master_rng_seed'])
            data_dir = self.sim_params['data_path']
            main_data_dir = os.path.join(*data_dir.split('/')[0:-1])
            fig_dir = os.path.join(main_data_dir, 'fig-' + master_rng_seed)
            beh_data_flpath = os.path.join(fig_dir, beh_data_flname)
            print('\n writing summary data in {}'.format(beh_data_flpath))
            file_exists = os.path.exists(beh_data_flpath)
            if file_exists and reset_summary_file:
                os.remove(beh_data_flpath)
                file_exists = False

            self.log_file = open(beh_data_flpath, 'a')  # creates the text file
            if not file_exists:
                self.write_header()

    def group_data(self):
        """ Grouping the data by trials in a pandas data frame format.
        """
        self.grp_data = self.data.groupby(by='trial')
        self.num_trials = self.grp_data.ngroups
        self.tr_ids = list(self.grp_data.groups.keys())

    def get_performance(self):
        """ Computing the performance of the agent as trial duration and
        trajectory length as a function of trials.
        """
        self.group_data()
        if self.sim_params['max_num_trs'] > self.num_trials:
            print('\nInsufficient number of trials ...\n')
        elif self.sim_params['max_num_trs'] < self.num_trials:
            self.num_trials = self.sim_params['max_num_trs']
            self.tr_ids = np.arange(1, self.num_trials + 1)

        self.traj_length = np.zeros(self.num_trials)
        self.tr_duration = np.zeros(self.num_trials)
        for i, tr in enumerate(self.tr_ids):
            tmp = self.grp_data.get_group(tr).to_numpy()[:, 1:]
            self.tr_duration[i] = (tmp[-1, 0] - tmp[0, 0]) * 1000
            self.traj_length[i] = np.sum(np.sqrt(np.diff(tmp[:, 1]) ** 2 + np.diff(tmp[:, 2]) ** 2))
            if hasattr(self, 'log_file'):
                self.write_data(tr, i)

        if hasattr(self, 'log_file'):
            self.log_file.close()

    def plot_performance(self, fl_suffix='', frm='pdf'):
        """ Plotting the computed performance.
        
        Parameters
        ----------
        fl_suffix : str, optional
            the string, which will be added to the end of the file.
            It can, e.g., be used to distinguish outcome of different simulations.
            The default is 'line'.
        frm : str, optional
            The format of the output figure. The default is 'pdf'.

        Returns
        -------
        None.

        """
        if not hasattr(self, 'tr_duration'):
            self.get_performance()

        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
        ax[0].bar(self.tr_ids, self.tr_duration)
        ax[1].bar(self.tr_ids, self.traj_length)
        ax[1].set_xlabel('Trial #')
        ax[0].set_ylabel('Trial duration (ms)')
        ax[1].set_ylabel('Trajectory length (a.u.)')
        fig.savefig(os.path.join(self.fig_path,
                                 'behavior_performance{}.{}'.format(fl_suffix, frm)),
                    format=frm)
        plt.close(fig)

    def write_data(self, tr, i):
        """ Writing/appending the performance data and the parameters that were used 
        for the related simulation in a file.

        Parameters
        ----------
        tr : int
            The trial number.
        i : int
            The index of the trial number.

        Returns
        -------
        None.

        """
        self.log_file.write('{}\t{}\t'.format(self.sim_params['master_rng_seed'],
                                              tr))

        self.log_file.write('{}\t{}\t'.format(self.tr_duration[i],
                                              self.traj_length[i]))

        self.log_file.write('\n')

    def write_header(self):
        """ Writing the header of the performance data file.
        """
        self.log_file.write('{}\t{}\t'.format('seed', 'trial'))

        self.log_file.write('{}\t{}\t'.format('tr_dur', 'traj_len'))

        self.log_file.write('\n')
