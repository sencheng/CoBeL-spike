#!/usr/bin/env python3

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis import SpikefileNest
from .PlotHelper import add_goal_zone, add_obstacles


class PositionFile():
    """
    Representing a position file.
    """

    def __init__(self, data_path, filename, fig_path, sim_file):
        print("Starting processing agent's location using json file ...\n")

        self.file_path = os.path.join(data_path, filename)
        if not os.path.exists(self.file_path):
            print('Could not find {}. Check whether the '
                  'data file exists and whether the '
                  'path to the file is correct!'.format(filename))
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        self.fig_path = fig_path
        # self.bin_width = 50.0
        # self.ms_scale = 1000.0
        with open('../openfield/network_params_spikingnet.json', 'r') as fl:
            tmp_dict = json.load(fl)
        self.env_extent = tmp_dict['place']['spatial_prop']
        self.fl_type = filename.split('.')[-1]
        self.read_sim_data()

    def read_sim_data(self, data_path='../openfield', sim_file='sim_params.json'):
        """
        loads the simulation parameters from the json file

        Parameters
        ----------
        data_path : TYPE, optional
            DESCRIPTION. The default is '../openfield'.
        sim_file : TYPE, optional
            DESCRIPTION. The default is 'sim_params.json'.

        Returns
        -------
        None.#        obs_dict = net_dict["environment"]["obstacles"]
#        self.obstacle_list = []
#        if obs_dict["flag"]:
#            for center, vert, horiz in zip(obs_dict["centers"], obs_dict["vert_lengths"], obs_dict["horiz_lengths"]):
#                delta_y = vert / 2. # Get the length and width 
#                delta_x = horiz / 2.  # as distances from the center point
#                
#                ll = (center[0] - delta_x, center[1] - delta_y) # lower left
#                lr = (center[0] + delta_x, center[1] - delta_y) # lower right
#                ur = (center[0] + delta_x, center[1] + delta_y) # upper right
#                ul = (center[0] - delta_x, center[1] + delta_y) # upper left
#                
#                self.obstacle_list.append([ll, lr, ur, ul])

        """
        self.sim_file_path = os.path.join(data_path, sim_file)
        with open(self.sim_file_path, "r") as f:
            self.sim_dict = json.load(f)

        with open('../openfield/sim_params.json', 'r') as fl:
            net_dict = json.load(fl)

        self.sim_env = net_dict['sim_env']
        self.env_limit_dic = net_dict['environment'][self.sim_env]
        if self.sim_env == 'openfield':
            self.xmin_position = float(self.env_limit_dic['xmin_position'])
            self.xmax_position = float(self.env_limit_dic['xmax_position'])
            self.ymin_position = float(self.env_limit_dic['ymin_position'])
            self.ymax_position = float(self.env_limit_dic['ymax_position'])
            self.opn_fld_xlim = np.array([self.xmin_position, self.xmax_position])
            self.opn_fld_ylim = np.array([self.ymin_position, self.ymax_position])
            self.hide_goal = net_dict['goal']['hide_goal']
            self.reward_recep_field = net_dict['goal']['reward_recep_field']
            self.goal_x = net_dict['goal']['x_position']
            self.goal_y = net_dict['goal']['y_position']
        else:
            print("environment {} undefined".format(self.sim_env))

        self.opn_fld_xlim = np.array([self.xmin_position, self.xmax_position])
        self.opn_fld_ylim = np.array([self.ymin_position, self.ymax_position])
        self.hide_goal = net_dict['goal']['hide_goal']
        self.reward_recep_field = net_dict['goal']['reward_recep_field']
        self.goal_x = net_dict['goal']['x_position']
        self.goal_y = net_dict['goal']['y_position']

    def read_pos_file(self):
        """
        Reads the position file located at `self.file_path`.
        It contains the position, type, firingrate and other parameters of each cell
        
        Returns
        -------
        None.
        """
        print('\tposfile_json: Reading real-time-position file: {} ...\n'.format(self.file_path))

        if self.fl_type == 'json':
            tmp = pd.read_json(self.file_path)
            self.pos_data = tmp.loc['obervation']
        else:
            self.pos_data = []
            tmp = pd.read_csv(self.file_path, sep='\t')
            self.grp_data = tmp.groupby(by='trial')
            self.num_trials = self.grp_data.ngroups
            self.tr_ids = list(self.grp_data.groups.keys())

            if self.sim_dict['max_num_trs'] > self.num_trials:
                print('\nInsufficient number of trials ...\n')

            elif self.sim_dict['max_num_trs'] < self.num_trials:
                self.num_trials = self.sim_dict['max_num_trs']
                self.tr_ids = np.arange(1, self.num_trials + 1)
  
              
    def get_times_from_pos_file(self):
        tmp = pd.read_csv(self.file_path, sep='\t')
        trial_times = []
        for i in range(1, tmp["trial"].nunique()+1):
            trial_data = tmp.loc[tmp["trial"] == i]
            trial_times.append({"trial": i, 
                                "start_time": int(trial_data["time"].values[0]*1000), 
                                "end_time": int(trial_data["time"].values[-1]*1000)})
        return trial_times


    def set_xy_lims(self):
        """
        Sets `self.horiz_lim` and `self.vert_lim`.
        """
        self.horiz_lim = np.array([0, 1]) * self.env_extent['width'] - self.env_extent['width'] / 2
        self.vert_lim = np.array([0, 1]) * self.env_extent['height'] - self.env_extent['height'] / 2
        self.horiz_lab = np.array([0, self.env_extent['width']])
        self.vert_lab = np.array([0, self.env_extent['height']])

    # Note: Must receive a *list* of formats, even if only a single element is contained
    def plot_rat_pos(self, fl_suffix='line', formats=['pdf'], title=False, legend_loc='best', show_obstacle=True):
        print("\tposfile_json: Plotting (line) agent's position ...\n")

        if not hasattr(self, 'grp_data'):
            self.read_pos_file()

        if not hasattr(self, 'vert_lim'):
            self.set_xy_lims()

        for ind, tr in enumerate(self.tr_ids):
            tmp = self.grp_data.get_group(tr)
            rat_pos = tmp.to_numpy()[:, 2:]
            tr_time = tmp.to_numpy()[:, 1] * 1000  # ms
            fig, ax = plt.subplots()
            if title:
                ax.set_title("Start: {}ms, End: {}ms\n Duration:{}ms".format(int(tr_time[0]), int(tr_time[-1]),
                                                                         int(tr_time[-1]) - int(tr_time[0])))
            ax.plot(rat_pos[:, 0], rat_pos[:, 1], color='blue', label='traversed path', lw=2.5)
            ax.plot(rat_pos[0, 0], rat_pos[0, 1], marker='o', color='green', label='initial position')
            ax.plot(rat_pos[-1, 0], rat_pos[-1, 1], marker='o', color='red', label='final position')
            ax.set_aspect(1)
            ax.set_xlim(self.horiz_lim)
            ax.set_ylim(self.vert_lim)
            
            if show_obstacle:
                add_obstacles(ax, self.sim_dict)
            
            # This rewrites the spatial labels to start from 0 instead of a negative number. It's easier than a coordinate transform
            plt.xticks(ticks=self.horiz_lim, labels=self.horiz_lab)
            plt.yticks(ticks=self.vert_lim, labels=self.vert_lab)
            
            if self.sim_dict['goal']['hide_goal'] == False:
                add_goal_zone(ax, self.sim_dict)

            # "best" legend location = 0
            # to draw the legend outside the axes requires bbox
            if legend_loc == 'best':
                ax.legend()
            elif legend_loc == 'out':
                ax.legend(bbox_to_anchor=(1.04, 0), loc="lower left")
            #else no legend
            
            plt.tight_layout()
            
            for frm in formats:
                fig.savefig(os.path.join(self.fig_path, 'ratpos-{:s}-tr{:d}.{}'.format(fl_suffix, tr, frm)), format=frm)
            plt.close(fig)

    def plot_rat_pos_arrow(self, fl_suffix='arrow', frm='pdf'):
        """ Plotting agent's position using arrows to have some indication
        of movement speed.

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
        print("\tposfile_json: Plotting (arrow) agent's position ...\n")
        if not hasattr(self, 'pos_data'):
            self.read_pos_file()

        if not hasattr(self, 'vert_lim'):
            self.set_xy_lims()

        for ind, p_data in enumerate(self.pos_data):
            rat_pos = np.array(p_data)
            fig, ax = plt.subplots()
            ax.quiver(rat_pos[:-1, 0], rat_pos[:-1, 1], rat_pos[1:, 0] - rat_pos[:-1, 0],
                      rat_pos[1:, 1] - rat_pos[:-1, 1], scale_units='xy', angles='xy', scale=1)
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

    def plot_pos_action(self, fl_suffix='pos-action', frm='pdf', path_dic={'data_path': None, 'filename': None}):
        print("\tposfile_json: Plotting agent's position and action vs. time ...\n")
        if not hasattr(self, 'pos_data'):
            self.read_pos_file()

        if not hasattr(self, 'vert_lim'):
            self.set_xy_lims()

        st_time = 0.0
        sim_res = 0.1
        spk_obj = SpikefileNest(path_dic['data_path'], path_dic['filename'], self.file_path)
        spk_obj.calc_avg_fr()
        for ind, p_data in enumerate(self.pos_data):
            rat_pos = np.array(p_data)
            time_vec = np.arange(st_time, st_time + rat_pos.shape[0], 1) * sim_res
            fr = spk_obj.fr_vec[:, (spk_obj.hist_edges >= time_vec[0]) & (spk_obj.hist_edges < time_vec[-1])]
            time_hist = spk_obj.hist_edges[(spk_obj.hist_edges >= time_vec[0]) & (spk_obj.hist_edges < time_vec[-1])]
            fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
            neuron_ids = np.arange(spk_obj.neuron_ids.min(), spk_obj.neuron_ids.max() + 1)
            im = ax[0].pcolor(time_hist, neuron_ids, fr, edgecolors='none')
            ax[0].set_ylabel('Neuron IDs')
            ax[1].plot(time_vec, rat_pos[:, 0], color='red', label='x')
            ax[1].plot(time_vec, rat_pos[:, 1], color='blue', label='y')
            ax[1].set_xlabel('Time (ms)')
            ax[1].set_ylabel('Coordinate (a.u.)')
            ax[1].set_ylim(self.vert_lim)
            ax[1].legend()
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            plt.colorbar(im, cax=cbar_ax)
            fig.savefig(os.path.join(self.fig_path, 'ratpos+fr-{:s}-tr{:d}.{}'.format(fl_suffix, ind + 1, frm)),
                        format=frm)
            plt.close(fig)
            st_time = st_time + rat_pos.shape[0]

    def plot_dir_dist(self, fl_suffix='phase_dist', frm='pdf'):
        """ Plotting the distribution of action vectors including zeros.

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
        print("\tposfile_json: Plotting the distribution of agent's movement direction ...\n")
        for ind, p_data in enumerate(self.pos_data):
            rat_pos = np.array(p_data)
            x_vecs = np.diff(rat_pos[:, 0])
            y_vecs = np.diff(rat_pos[:, 1])
            phases = np.arctan2(x_vecs, y_vecs) / np.pi * 180
            hist_edges = np.arange(-180, 180, 1)
            hist_edges_phase = (hist_edges[:-1] + hist_edges[1:]) / 2
            hist_vals_phase = np.histogram(phases, hist_edges, density=True)[0]
            fig, ax = plt.subplots()
            ax.bar(hist_edges_phase, hist_vals_phase)
            ax.set_xlabel('Movement direction (degree)')
            ax.set_ylabel('Probability')
            fig.savefig(os.path.join(self.fig_path, 'ratmovvec-{:s}-tr{:d}.{}'.format(fl_suffix, ind + 1, frm)),
                        format=frm)
            plt.close(fig)

    def plot_dir_dist_nonzero(self, fl_suffix='phase_dist', frm='pdf'):
        """ Plotting the distribution of action vectors that are nonzeros.
        
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
        for ind, p_data in enumerate(self.pos_data):
            rat_pos = np.array(p_data)
            x_vecs = np.diff(rat_pos[:, 0])
            y_vecs = np.diff(rat_pos[:, 1])
            phases = np.arctan2(x_vecs[(x_vecs != 0) & (y_vecs != 0)],
                                y_vecs[(x_vecs != 0) & (y_vecs != 0)]) / np.pi * 180
            hist_edges = np.arange(-180, 180, 1)
            hist_edges_phase = (hist_edges[:-1] + hist_edges[1:]) / 2
            hist_vals_phase = np.histogram(phases, hist_edges, density=True)[0]
            fig, ax = plt.subplots()
            ax.bar(hist_edges_phase, hist_vals_phase)
            ax.set_xlabel('Movement direction (degree)')
            ax.set_ylabel('Probability')
            fig.savefig(os.path.join(self.fig_path, 'ratmovvec-{:s}-tr{:d}.{}'.format(fl_suffix, ind + 1, frm)),
                        format=frm)
            plt.close(fig)
