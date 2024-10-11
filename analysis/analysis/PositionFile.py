#!/usr/bin/env python3

import os
import json
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from .PlotHelper import add_goal_zone, add_obstacles, add_tmaze


class PositionFile:
    """
    Representing a position file.
    """

    def __init__(self, data_path, filename, fig_path):
        print("Starting processing agent's location using json file ...\n")

        self.data_path = data_path
        self.file_path = os.path.join(data_path, filename)
        if not os.path.exists(self.file_path):
            print(
                'Could not find {}. Check whether the data file exists and whether the path to the file is correct!'.format(
                    filename
                )
            )
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        self.fig_path = fig_path
        with open(os.path.join(data_path, "network_params_spikingnet.json"), 'r') as fl:
            tmp_dict = json.load(fl)
        self.env_extent = tmp_dict['place']['spatial_prop']
        self.read_sim_data(data_path=data_path)

    def read_sim_data(self, data_path='../simulator', sim_file='sim_params.json', tr_file='trials_params.dat'):
        """
        Loads the simulation parameters from the JSON file.

        Parameters
        ----------
        data_path : str, optional
            Path to the data directory. The default is '../simulator'.
        sim_file : str, optional
            Simulation file name. The default is 'sim_params.json'.
        tr_file : str, optional
            Trial parameters file name. The default is 'trials_params.dat'.

        Returns
        -------
        None.
        """
        self.sim_file_path = os.path.join(data_path, sim_file)
        trial_data_path = os.path.join(data_path, tr_file)
        with open(self.sim_file_path, "r") as f:
            self.sim_dict = json.load(f)
        with open(os.path.join(data_path, 'env_params.json'), 'r') as f:
            self.env_dict = json.load(f)
        with open(trial_data_path, 'r') as f:
            self.trial_data = pd.read_csv(f, sep='\t')

        self.sim_env = self.env_dict['sim_env']
        self.env_limit_dic = self.env_dict['environment'][self.sim_env]
        if self.sim_env in ['openfield', 'tmaze']:
            self.xmin_position = float(self.env_limit_dic['xmin_position'])
            self.xmax_position = float(self.env_limit_dic['xmax_position'])
            self.ymin_position = float(self.env_limit_dic['ymin_position'])
            self.ymax_position = float(self.env_limit_dic['ymax_position'])
            self.opn_fld_xlim = np.array([self.xmin_position, self.xmax_position])
            self.opn_fld_ylim = np.array([self.ymin_position, self.ymax_position])
            self.hide_goal = self.env_dict['environment']['goal']['hide_goal']
        else:
            print("environment {} undefined".format(self.sim_env))
            raise NotImplementedError

    def read_pos_file(self):
        """
        Reads the position file located at `self.file_path`.
        It contains the position, type, firing rate, and other parameters of each cell.

        Returns
        -------
        None.
        """
        print('\tposfile_json: Reading real-time-position file: {} ...\n'.format(self.file_path))

        self.pos_file = pd.read_csv(self.file_path, sep='\t')
        self.pos_time = self.pos_file.time.values * 1000
        self.pos_xy = self.pos_file.values[:, 2:]
        self.occupancy = self.pos_file.drop_duplicates(subset=['time']).where(
            self.pos_file['trial'] <= self.sim_dict['max_num_trs']
        )
        self.grp_data = self.pos_file.groupby(by='trial')
        self.num_trials = self.grp_data.ngroups
        self.tr_ids = list(self.grp_data.groups.keys())

        if self.sim_dict['max_num_trs'] > self.num_trials:
            print('\nInsufficient number of trials ...\n')
        elif self.sim_dict['max_num_trs'] < self.num_trials:
            self.num_trials = self.sim_dict['max_num_trs']
            self.tr_ids = np.arange(1, self.num_trials + 1)
            self.pos_file = self.pos_file[self.pos_file["trial"] <= self.num_trials]

    def get_times_from_pos_file(self):
        tmp = pd.read_csv(self.file_path, sep='\t')
        trial_times = []
        for i in range(1, tmp["trial"].nunique() + 1):
            trial_data = tmp.loc[tmp["trial"] == i]
            trial_times.append(
                {
                    "trial": i,
                    "start_time": int(trial_data["time"].values[0] * 1000),
                    "end_time": int(trial_data["time"].values[-1] * 1000),
                }
            )
        return trial_times

    def set_xy_lims(self):
        """
        Sets `self.horiz_lim` and `self.vert_lim`.
        """
        self.horiz_lim = np.array([0, 1]) * self.env_extent['width'] - self.env_extent['width'] / 2
        self.vert_lim = np.array([0, 1]) * self.env_extent['height'] - self.env_extent['height'] / 2
        self.horiz_lab = np.array([0, self.env_extent['width']])
        self.vert_lab = np.array([0, self.env_extent['height']])

    def calc_DTW_optimal(self, calc_all_goals=True, w=None, opt_path_size=300, space=25, legend_loc='out', formats=['png']):
        """
        Calculates the Dynamic Time Warping (DTW) optimal path per trial.

        Parameters
        ----------
        calc_all_goals : bool, optional
            Whether to calculate all goals. The default is True.
        w : int, optional
            Window size for the DTW algorithm. The default is None.
        opt_path_size : int, optional
            Number of points in the optimal path. The default is 300.
        space : int, optional
            Spacing between lines drawn from one trajectory to the other. The default is 25.
        legend_loc : str, optional
            Location of the legend. The default is 'out'.
        formats : list, optional
            List of file formats for saving the plots. The default is ['png'].

        Returns
        -------
        DataFrame
            DataFrame containing DTW results.
        """
        if not hasattr(self, 'grp_data'):
            self.read_pos_file()
        if not hasattr(self, 'vert_lim'):
            self.set_xy_lims()
        if calc_all_goals:
            goal_list = self.trial_data.drop_duplicates(
                subset=['goal_shape', 'goal_size1', 'goal_size2', 'goal_x', 'goal_y']
            ).drop(columns='trial_num')
            tr_goal_pairs = (
                pd.concat([goal_list] * len(self.tr_ids), keys=self.tr_ids, names=['trial_num'])
            ).reset_index(level=0)
        else:
            tr_goal_pairs = self.trial_data

        DTW_dat = []

        for i, row in tr_goal_pairs.iterrows():
            tr = row['trial_num']
            goal_x, goal_y = row['goal_x'], row['goal_y']
            goal_rad = row['goal_size1']
            start_x, start_y = row['start_x'], row['start_y']
            goal_vec = np.array([goal_x - start_x, goal_y - start_y])
            goal_dir = goal_vec / np.linalg.norm(goal_vec)
            goal_edge_x = goal_x - goal_rad * goal_dir[0]
            goal_edge_y = goal_y - goal_rad * goal_dir[1]
            optimal_path = np.linspace((start_x, start_y), (goal_edge_x, goal_edge_y), opt_path_size)

            traj = self.grp_data.get_group(tr).drop_duplicates(subset=['time']).to_numpy()[:, 2:]
            fig, ax = plt.subplots()
            ax.plot(traj[:, 0], traj[:, 1], color='blue', label='traversed path', lw=2.5)
            ax.plot(traj[0, 0], traj[0, 1], marker='o', color='green', label='initial position')
            ax.plot(traj[-1, 0], traj[-1, 1], marker='o', color='red', label='final position')
            ax.plot(optimal_path[:, 0], optimal_path[:, 1], color='red', label=f'optimal path', lw=2.5)
            add_goal_zone(ax, os.path.join(self.data_path, 'trials_params.dat'), tr=tr, lw=2)

            n = len(traj)
            m = len(optimal_path)
            init_fill = np.inf
            DTW_mat = np.full((n, m), init_fill)
            DTW_mat[0, 0] = 0

            if w is None:
                for i in range(1, n):
                    for j in range(1, m):
                        cost = np.linalg.norm(traj[i] - optimal_path[j])
                        alternatives = np.array([DTW_mat[i - 1, j], DTW_mat[i, j - 1], DTW_mat[i - 1, j - 1]])
                        DTW_mat[i, j] = cost + np.min(alternatives)
            else:
                w = max(w, np.abs(n - m))
                for i in range(1, n):
                    for j in range(max(1, i - w), min(m, i + w)):
                        DTW_mat[i, j] = 0
                for i in range(1, n):
                    for j in range(max(1, i - w), min(m, i + w)):
                        cost = np.linalg.norm(traj[i] - optimal_path[j])
                        alternatives = np.array([DTW_mat[i - 1, j], DTW_mat[i, j - 1], DTW_mat[i - 1, j - 1]])
                        DTW_mat[i, j] = cost + np.min(alternatives)

            for i in range(1, n, space):
                match = np.argmin(DTW_mat[i])
                ax.plot(
                    [traj[i, 0], optimal_path[match, 0]],
                    [traj[i, 1], optimal_path[match, 1]],
                    color='grey',
                    alpha=0.6,
                    lw=1.3,
                )

            ax.set_aspect(1)
            ax.set_xlim(self.horiz_lim)
            ax.set_ylim(self.vert_lim)
            plt.xticks(ticks=self.horiz_lim, labels=self.horiz_lab)
            plt.yticks(ticks=self.vert_lim, labels=self.vert_lab)

            title = f'DTW vs optimal path for trial {tr}: {DTW_mat[-1, -1]:.2f}'
            fl_name = f'DTW-opt-tr{tr}-goal({goal_x},{goal_y})r={goal_rad}'

            if w is not None:
                title = f'{w} windowed ' + title
                fl_name = f'{w}-windowed-' + fl_name

            if legend_loc == 'best':
                ax.legend(fontsize='small')
            elif legend_loc == 'out':
                ax.legend(bbox_to_anchor=(0, 0), loc="lower left", fontsize='medium')
            ax.set_title(title)

            for frm in formats:
                fig.savefig(os.path.join(self.fig_path, fl_name + f'.{frm}'), format=frm)
            plt.close(fig)
            DTW_dat.append([tr, goal_x, goal_y, opt_path_size, w, DTW_mat[-1, -1]])
            plt.close(fig)

        df = pd.DataFrame(
            data=DTW_dat, columns=['trial', 'goal_x', 'goal_y', 'opt_path_size', 'w', 'DTW']
        )
        df.to_csv(os.path.join(self.fig_path, 'DTW_optimal_path.dat'), na_rep='None', sep="\t", index=False)
        return df

    def calc_DTW(self, trials, w=None, formats=['png'], show_obstacle=True, legend_loc='best'):
        """
        Calculates the Dynamic Time Warping (DTW) between two trials.

        Parameters
        ----------
        trials : list
            List containing two trial numbers to compare.
        w : int, optional
            Window size for the DTW algorithm. The default is None.
        formats : list, optional
            List of file formats for saving the plots. The default is ['png'].
        show_obstacle : bool, optional
            Whether to show obstacles in the plot. The default is True.
        legend_loc : str, optional
            Location of the legend. The default is 'best'.

        Returns
        -------
        float
            DTW value between the two trials.
        """
        if not hasattr(self, 'grp_data'):
            self.read_pos_file()
        if not hasattr(self, 'vert_lim'):
            self.set_xy_lims()

        tr1 = self.grp_data.get_group(trials[0]).drop_duplicates(subset=['time'])
        tr2 = self.grp_data.get_group(trials[1]).drop_duplicates(subset=['time'])
        traj1 = tr1.to_numpy()[:, 2:]
        traj2 = tr2.to_numpy()[:, 2:]

        fig, ax = plt.subplots()
        ax.plot(traj1[:, 0], traj1[:, 1], color='blue', label=f'Trial {trials[0]}', lw=2.5)
        ax.plot(traj2[:, 0], traj2[:, 1], color='red', label=f'Trial {trials[1]}', lw=2.5)
        ax.plot(traj1[0, 0], traj1[0, 1], marker='o', color='green', label='Initial Position')
        ax.plot(traj1[-1, 0], traj1[-1, 1], marker='o', color='red', label='Final Position')
        ax.plot(traj2[-1, 0], traj2[-1, 1], marker='o', color='red')

        ax.set_aspect(1)
        ax.set_xlim(self.horiz_lim)
        ax.set_ylim(self.vert_lim)

        if show_obstacle:
            add_obstacles(ax, self.env_dict)
        plt.xticks(ticks=self.horiz_lim, labels=self.horiz_lab)
        plt.yticks(ticks=self.vert_lim, labels=self.vert_lab)

        n = len(traj1)
        m = len(traj2)
        init_fill = np.inf
        DTW_mat = np.full((n, m), init_fill)
        DTW_mat[0, 0] = 0

        if w is None:
            for i in range(1, n):
                for j in range(1, m):
                    cost = np.linalg.norm(traj1[i] - traj2[j])
                    alternatives = np.array([DTW_mat[i - 1, j], DTW_mat[i, j - 1], DTW_mat[i - 1, j - 1]])
                    DTW_mat[i, j] = cost + np.min(alternatives)
        else:
            w = max(w, np.abs(n - m))
            for i in range(1, n):
                for j in range(max(1, i - w), min(m, i + w)):
                    DTW_mat[i, j] = 0
            for i in range(1, n):
                for j in range(max(1, i - w), min(m, i + w)):
                    cost = np.linalg.norm(traj1[i] - traj2[j])
                    alternatives = np.array([DTW_mat[i - 1, j], DTW_mat[i, j - 1], DTW_mat[i - 1, j - 1]])
                    DTW_mat[i, j] = cost + np.min(alternatives)

        space = 25
        for i in range(1, n, space):
            match = np.argmin(DTW_mat[i])
            ax.plot([traj1[i, 0], traj2[match, 0]], [traj1[i, 1], traj2[match, 1]], color='grey')

        if w is None:
            title = f'DTW for Trials {trials[0]} and {trials[1]}: {DTW_mat[-1, -1]:.2f}'
            fl_name = f'DTW-tr{trials[0]}-tr{trials[1]}'
        else:
            title = f'{w}-Windowed DTW for Trials {trials[0]} and {trials[1]}: {DTW_mat[-1, -1]:.2f}'
            fl_name = f'{w}-windowedDTW-tr{trials[0]}-tr{trials[1]}'

        add_goal_zone(ax, os.path.join(self.data_path, 'trials_params.dat'), tr=trials[0], lw=2)

        if legend_loc == 'best':
            ax.legend()
        elif legend_loc == 'out':
            ax.legend(bbox_to_anchor=(0, 0), loc="lower left")
        ax.set_title(title)

        for frm in formats:
            fig.savefig(os.path.join(self.fig_path, fl_name + f'.{frm}'), format=frm)
        plt.close(fig)

        return DTW_mat[-1, -1]

    def calc_DTW_all(self, trials=[], w=None, formats=['png'], show_obstacle=True, legend_loc='best'):
        DTW_dat = []

        trials = self.tr_ids if trials == [] else trials
        trials_combinations = list(itertools.combinations(trials, 2))
        trials_combinations = [list(item) for item in trials_combinations]

        fig_path_copy = self.fig_path
        self.fig_path = os.path.join(self.fig_path, "DTW_ALL")
        os.makedirs(self.fig_path, exist_ok=True)

        for trial in trials_combinations:
            dtw = self.calc_DTW(trial, w, formats, show_obstacle, legend_loc)
            DTW_dat.append([int(trial[0]), int(trial[1]), w, dtw])
        
        self.fig_path = fig_path_copy
        df = pd.DataFrame(data=DTW_dat, columns=['trial1', 'trial2', 'w', 'DTW'])
        df.to_csv(os.path.join(self.fig_path, 'DTW_all.dat'), na_rep='None', sep="\t", index=False)

    def plot_occupancy(self, start=1, end=None, grid_size=100, formats=['png'], show_title=True, show_target=True, show_colorbar=True, show_obstacle=True):
        if not hasattr(self, 'grp_data'):
            self.read_pos_file()
        if not hasattr(self, 'vert_lim'):
            self.set_xy_lims()
        if end is None:
            end = self.num_trials

        xdim = np.linspace(self.xmin_position, self.xmax_position, grid_size+1)
        ydim = np.linspace(self.ymin_position, self.ymax_position, grid_size+1)
        occupancy = self.occupancy.where((start <= self.occupancy['trial']) & (self.occupancy['trial'] <= end))
        
        occ_grid = np.zeros((grid_size, grid_size))
        
        lastx, lasty = None, None
        for xi, x in enumerate(xdim):
            for yi, y in enumerate(ydim):
                if lastx is not None and lasty is not None:
                    bucket_sum = occupancy[(lastx <= occupancy['x']) & (occupancy['x'] < x) & (lasty <= occupancy['y']) & (occupancy['y'] < y)]
                    occ_grid[yi-1, xi-1] = bucket_sum.time.values.size # Note that the y index comes first to match trajectory plots
                lasty = y
            lastx = x
            lasty = None

        fig, ax = plt.subplots()
        ax.set_aspect(1)

        plt.xticks(ticks=self.horiz_lim, labels=self.horiz_lab)
        plt.yticks(ticks=self.vert_lim, labels=self.vert_lab)

        colors = plt.cm.get_cmap('Blues')(np.linspace(0.2, 1.0, 100))
        cmap = LinearSegmentedColormap.from_list('truncated_blues', colors)
        mesh = ax.pcolormesh(xdim, ydim, occ_grid, cmap=cmap)

        if show_title:
            title = f'Occupancy for trial {start}' if start == end else f'Occupancy of trials {start}-{end}'
            ax.set_title(title)
        
        if show_colorbar:
            cbar = plt.colorbar(mesh)
            cbar.set_label("Time spent in ms")
        
        if show_obstacle:
            add_obstacles(ax, self.env_dict)
        
        if show_target:
            add_goal_zone(ax, os.path.join(self.data_path, 'trials_params.dat'), tr=1, lw=2)

        for frm in formats:
            fig.savefig(os.path.join(self.fig_path, f'trs{start}-{end}_occ.{frm}'), format=frm)
        plt.close(fig)

    def plot_intergoal_occupancy(self, start=1, end=None, buckets=100, formats=['png'], show_obstacle=True):
        """
        Plots the intergoal occupancy map for the agent's location.

        Parameters
        ----------
        start : int, optional
            The start trial number. The default is 1.
        end : int, optional
            The end trial number. The default is None.
        buckets : int, optional
            Number of buckets for the occupancy grid. The default is 100.
        formats : list, optional
            List of file formats for saving the plots. The default is ['png'].
        show_obstacle : bool, optional
            Whether to show obstacles in the plot. The default is True.

        Returns
        -------
        None.
        """
        if not hasattr(self, 'grp_data'):
            self.read_pos_file()
        if not hasattr(self, 'vert_lim'):
            self.set_xy_lims()
        if end is None:
            end = self.num_trials

        self.find_intergoal_traj(start, end)

        xdim = np.linspace(self.xmin_position, self.xmax_position, buckets + 1)
        ydim = np.linspace(self.ymin_position, self.ymax_position, buckets + 1)
        occ_grid = np.zeros((buckets, buckets))
        df = pd.DataFrame({'x': [], 'y': []})

        for tr in self.ig_traj.keys():
            if self.ig_traj[tr] is not None:
                x = self.ig_traj[tr]['ig'][:, 0]
                y = self.ig_traj[tr]['ig'][:, 1]
                new_df = pd.DataFrame({'x': x, 'y': y})
                df = pd.concat([df, new_df])

        lastx = None
        lasty = None
        for xi, x in enumerate(xdim):
            for yi, y in enumerate(ydim):
                if lastx is not None and lasty is not None:
                    bucket_sum = df[
                        (lastx <= df['x']) & (df['x'] < x) & (lasty <= df['y']) & (df['y'] < y)
                    ]
                    occ_grid[yi - 1, xi - 1] = bucket_sum.size
                lasty = y
            lastx = x
            lasty = None

        fig, ax = plt.subplots()
        ax.set_aspect(1)
        ax.pcolormesh(xdim, ydim, occ_grid)
        if start == end:
            title = f'Intergoal Occupancy for Trial {start}'
        else:
            title = f'Intergoal Occupancy of Trials {start}-{end}'
        ax.set_title(title)
        plt.xticks(ticks=self.horiz_lim, labels=self.horiz_lab)
        plt.yticks(ticks=self.vert_lim, labels=self.vert_lab)

        if show_obstacle:
            add_obstacles(ax, self.env_dict)

        add_goal_zone(ax, os.path.join(self.data_path, 'trials_params.dat'), tr=1, lw=2)

        for frm in formats:
            fig.savefig(os.path.join(self.fig_path, f'trs{start}-{end}_ig_occ.{frm}'), format=frm)
        plt.close(fig)

    def find_intergoal_traj(self, start=1, end=None):
        """
        Identifies the intergoal trajectory segments between goal zones.

        Parameters
        ----------
        start : int, optional
            The start trial number. The default is 1.
        end : int, optional
            The end trial number. The default is None.

        Returns
        -------
        None.
        """
        if not hasattr(self, 'grp_data'):
            self.read_pos_file()
        if hasattr(self, 'ig_traj'):
            return
        if end is None:
            end = self.num_trials

        self.ig_traj = {}

        tr_dat = self.trial_data.drop_duplicates(subset=['goal_size1', 'goal_x', 'goal_y'])
        goal1_dat = tr_dat.iloc[0]
        goal2_dat = tr_dat.iloc[1]
        goal1_pos = (goal1_dat['goal_x'], goal1_dat['goal_y'])
        goal1_rad = goal1_dat['goal_size1']
        goal2_pos = (goal2_dat['goal_x'], goal2_dat['goal_y'])
        goal2_rad = goal2_dat['goal_size1']

        for tr in self.tr_ids[start - 1:end]:
            tmp = self.grp_data.get_group(tr)

            ig_start = -1
            ig_end = -1
            for i, row in tmp.iterrows():
                pos = np.array([row.x, row.y])
                if ig_start == -1 and bool(np.linalg.norm(pos - goal1_pos) <= goal1_rad):
                    ig_start = i
                if ig_start != -1 and ig_end == -1 and bool(np.linalg.norm(pos - goal2_pos) <= goal2_rad):
                    ig_end = i
            if ig_start == -1:
                self.ig_traj[tr] = None
                continue

            self.ig_traj[tr] = {}
            self.ig_traj[tr]['ig'] = self.pos_xy[ig_start:ig_end]
            self.ig_traj[tr]['eg1'] = self.pos_xy[tmp.index[0]:ig_start]
            self.ig_traj[tr]['eg2'] = self.pos_xy[ig_end:tmp.index[-1]]
        
    def _create_velocity_trajectory(
        self, tr, ax, fig, title, show_obstacle, 
        legend_loc, linewidth=2.5, markersize=6.0, 
        titlesize=12, legendsize="small", ticksize=10.0
    ):
        """
        Create a plot of the agent's trajectory with color-coded velocity.

        Parameters:
        tr (int): The trial number.
        ax (matplotlib.axes.Axes): The axis to plot on.
        fig (matplotlib.figure.Figure): The figure to plot on.
        title (str): The title of the plot.
        show_obstacle (bool): Whether to display obstacles.
        legend_loc (str): Location of the legend.
        linewidth (float): Width of the trajectory line.
        markersize (float): Size of the markers.
        titlesize (float): Size of the title font.
        legendsize (str): Size of the legend font.
        ticksize (float): Size of the tick labels.
        """
        colors = plt.cm.get_cmap('Blues')(np.linspace(0.5, 1.0, 1024))
        cmap = LinearSegmentedColormap.from_list('truncated_blues', colors)

        points = (
            self.velocity_file[self.velocity_file['trial'] == tr][['x', 'y']]
            .values.reshape(-1, 1, 2)
        )
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap=cmap, norm=self.norm)
        lc.set_array(self.velocity_file['velocity'])
        lc.set_linewidth(linewidth)
        ax.add_collection(lc)

        cbar = fig.colorbar(lc, ax=ax, label='Velocity (m/s)')

        ticks = cbar.get_ticks()
        ticks = ticks[ticks <= self.max_velocity]

        ticklabels = ticks.copy()

        ticks = np.append(ticks, round(self.max_velocity, 2))
        ticklabels = np.append(
            ticklabels, r"$\geq $" + str(round(self.max_velocity, 2))
        )

        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticklabels)

        ax.plot([], [], markersize=0.1, color='blue', label='traversed path',
                lw=linewidth, zorder=1)
        ax.scatter(
            points[0, 0, 0], points[0, 0, 1], marker='o', color='green', 
            label='initial position', zorder=3, s=markersize**2
        )
        ax.scatter(
            points[-1, 0, 0], points[-1, 0, 1], marker='o', color='red', 
            label='final position', zorder=2, s=markersize**2
        )

        ax.set_aspect("equal")
        ax.set_xlim(self.horiz_lim)
        ax.set_ylim(self.vert_lim)

        ax.set_xticks(ticks=self.horiz_lim, labels=self.horiz_lab, fontsize=ticksize)
        ax.set_yticks(ticks=self.vert_lim, labels=self.vert_lab, fontsize=ticksize)

        if title is not None:
            ax.set_title(title, fontsize=titlesize)

        if show_obstacle:
            add_obstacles(ax, self.env_dict)

        if not self.hide_goal:
            add_goal_zone(ax, os.path.join(self.data_path, 'trials_params.dat'), tr=tr)

        if self.sim_env == 'tmaze':
            add_tmaze(ax, self.env_dict)

        if legend_loc == 'best':
            ax.legend(fontsize=legendsize)
        elif legend_loc == 'out':
            ax.legend(bbox_to_anchor=(0, 0), loc="lower left", fontsize=legendsize)
        elif legend_loc == 'outside':
            ax.legend(
                loc='center right', bbox_to_anchor=(-0.1, 0.5), fontsize=legendsize
            )

    def create_velocity(self, quantile=0.95, smoothing=10):
        self.velocity_file = self.pos_file.copy(deep=True)

        self.velocity_file = self.velocity_file.groupby(
            ['trial', 'time'], as_index=False
        ).mean()

        self.velocity_file['dx'] = self.velocity_file['x'].diff()
        self.velocity_file['dy'] = self.velocity_file['y'].diff()
        self.velocity_file['dt'] = self.velocity_file['time'].diff()

        self.velocity_file['displacement'] = np.sqrt(
            self.velocity_file['dx'] ** 2 + self.velocity_file['dy'] ** 2
        )

        self.velocity_file['velocity'] = (
            self.velocity_file['displacement'] / self.velocity_file['dt']
        )
        self.velocity_file['velocity'] = (
            self.velocity_file['velocity'].rolling(window=smoothing).mean()
        )
        self.velocity_file['velocity'] = (
            self.velocity_file['velocity'].fillna(0)
        )
        self.velocity_file['velocity'] = (
            self.velocity_file['velocity']
            * (self.velocity_file['trial'] == self.velocity_file['trial'].shift())
        )

        self.velocity_file = self.velocity_file.drop(
            columns=['dx', 'dy', 'dt', 'displacement']
        )

        self.max_velocity = self.velocity_file['velocity'].quantile(quantile)

        self.norm = plt.Normalize(
            self.velocity_file['velocity'].min(),
            self.velocity_file['velocity'].quantile(quantile)
        )

    def plot_rat_pos(
        self, formats=None, title=True, legend_loc='out', ticksize=10.0,
        linewidth=2.5, markersize=6.0, titlesize=12, legendsize="small",
        show_obstacle=True, show_velocity=True
    ):
        if formats is None:
            formats = ["pdf"]

        print("\tposfile_json: Plotting (line) agent's position ...\n")

        if not hasattr(self, 'grp_data'):
            self.read_pos_file()

        if not hasattr(self, 'vert_lim'):
            self.set_xy_lims()

        if show_velocity:
            self.create_velocity()

        for tr in self.tr_ids:
            traj_title = None
            if title:
                traj_title = f"Trial:{tr}"

            fig, ax = plt.subplots()
            if show_velocity:
                self._create_velocity_trajectory(
                    tr, ax, fig, traj_title, show_obstacle, legend_loc,
                    linewidth=linewidth, markersize=markersize,
                    titlesize=titlesize, legendsize=legendsize,
                    ticksize=ticksize
                )
            else:
                self._create_trajectory(
                    tr, ax, traj_title, show_obstacle, legend_loc,
                    linewidth=linewidth, markersize=markersize,
                    titlesize=titlesize, legendsize=legendsize,
                    ticksize=ticksize
                )
            fig.tight_layout()

            for frm in formats:
                fig.savefig(
                    os.path.join(self.fig_path, f'trajectory-tr{int(tr)}.{frm}'),
                    format=frm
                )
            plt.close(fig)

    def subplot_rat_pos1(
        self, subplot_ids=None, formats=None, title=True, legend_loc='out',
        ticksize=10.0, linewidth=2.5, markersize=6.0, titlesize=12,
        legendsize="small", show_obstacle=True, show_velocity=True
    ):
        if formats is None:
            formats = ["pdf"]

        for i, subplot_id in enumerate(subplot_ids):
            subplot_id = np.array(subplot_id)
            ids = subplot_id.flatten()
            row, col = subplot_id.shape

            fig, axes = plt.subplots(row, col, figsize=(col * 5, row * 5))
            print(f'Creating a subplot of {row}x{col} dimensions')

            for index, (ax, tr) in enumerate(zip(axes.flatten(), ids)):
                loc = None
                if index == len(ids) - 1:
                    loc = legend_loc

                traj_title = None
                if title:
                    traj_title = f"Trial:{tr}"

                if show_velocity:
                    self._create_velocity_trajectory(
                        tr, ax, fig, traj_title, show_obstacle, loc,
                        linewidth=linewidth, markersize=markersize,
                        titlesize=titlesize, legendsize=legendsize,
                        ticksize=ticksize
                    )
                else:
                    self._create_trajectory(
                        tr, ax, traj_title, show_obstacle, loc,
                        linewidth=linewidth, markersize=markersize,
                        titlesize=titlesize, legendsize=legendsize,
                        ticksize=ticksize
                    )
            fig.tight_layout()

            for frm in formats:
                fig.savefig(
                    os.path.join(self.fig_path, f'trajectories_{i}.{frm}'),
                    format=frm
                )
            plt.close(fig)

    def subplot_rat_pos2(
        self, shape=None, formats=None, title=True, legend_loc='out',
        ticksize=10.0, linewidth=2.5, markersize=6.0, titlesize=12,
        legendsize="small", show_obstacle=True, show_velocity=True
    ):
        if formats is None:
            formats = ["pdf"]

        row, col = shape[0], shape[1]
        ids = self.tr_ids

        fig, axes = plt.subplots(row, col, figsize=(col * 5, row * 5))
        print(f'Creating a subplot of {row}x{col} dimensions')

        for index, (ax, tr) in enumerate(zip(axes.flatten(), ids)):
            loc = None
            if index == len(ids) - 1:
                loc = legend_loc

            traj_title = None
            if title:
                traj_title = f"Trial:{tr}"

            if show_velocity:
                self._create_velocity_trajectory(
                    tr, ax, fig, traj_title, show_obstacle, loc,
                    linewidth=linewidth, markersize=markersize,
                    titlesize=titlesize, legendsize=legendsize,
                    ticksize=ticksize
                )
            else:
                self._create_trajectory(
                    tr, ax, traj_title, show_obstacle, loc,
                    linewidth=linewidth, markersize=markersize,
                    titlesize=titlesize, legendsize=legendsize,
                    ticksize=ticksize
                )
        fig.tight_layout()

        for frm in formats:
            fig.savefig(
                os.path.join(self.fig_path, f'trajectories.{frm}'),
                format=frm
            )
        plt.close(fig)

    def plot_rat_pos_3d(
        self, formats=None, colorbar_axis="z", title=True, legend_loc="out"
    ):
        if formats is None:
            formats = ["pdf"]

        if not hasattr(self, 'grp_data'):
            self.read_pos_file()

        if not hasattr(self, 'vert_lim'):
            self.set_xy_lims()

        for tr in self.tr_ids:
            tmp = self.grp_data.get_group(tr)

            x = tmp["x"].to_numpy()
            y = tmp["y"].to_numpy()
            z = tmp["time"].to_numpy()

            # Normalize z values for color mapping
            if colorbar_axis == "z":
                norm = plt.Normalize(z.min(), z.max())
            elif colorbar_axis == "x":
                norm = plt.Normalize(x.min(), x.max())
            elif colorbar_axis == "y":
                norm = plt.Normalize(y.min(), y.max())
            cmap = plt.get_cmap('viridis')

            # Create segments for the line
            points = np.array([x, y, z]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Create a Line3DCollection
            lc = Line3DCollection(segments, cmap=cmap, norm=norm)

            if colorbar_axis == "z":
                lc.set_array(z)
            elif colorbar_axis == "x":
                lc.set_array(x)
            elif colorbar_axis == "y":
                lc.set_array(y)

            # Create a 3D plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Add the collection to the plot
            ax.add_collection3d(lc)

            # Set plot limits
            ax.set_xlim(self.horiz_lim)
            ax.set_ylim(self.vert_lim)
            ax.set_zlim(z.min(), z.max())

            y_ratio = (
                (abs(self.vert_lim[0]) + np.abs(self.vert_lim[1]))
                / (abs(self.horiz_lim[0]) + np.abs(self.horiz_lim[1]))
            )
            ax.set_box_aspect([1, y_ratio, 1.0])

            # Show color bar
            cbar = plt.colorbar(lc, ax=ax)

            if colorbar_axis == "z":
                cbar.set_label('Time (z)')
            elif colorbar_axis == "x":
                cbar.set_label('Position (x)')
            elif colorbar_axis == "y":
                cbar.set_label('Position (y)')

            for frm in formats:
                fig.savefig(
                    os.path.join(self.fig_path, f'trajectory-3d-tr{int(tr)}.{frm}'),
                    format=frm
                )
            plt.close(fig)


    def animate_rat_pos(
        self, title=True, reset_time=True, show_obstacle=True,
        add_legend=True, frame_dur=0.05, last_frame_dur=1.0,
        fps_multiplier=1.0, dpi=300
        ):
        """Animates the rat position plot.

        Args:
            title (bool, optional): Specifies if the plot title should be shown 
                or not. The title shows the current trial and the time. 
                Defaults to True.
            reset_time (bool, optional): Specifies if the time should be set to 
                0 between the trials for the title. Defaults to True.
            show_obstacle (bool, optional): Specifies if obstacles in the 
                openfield should be shown or not. Defaults to True.
            frame_dur (float, optional): Specifies how long a single frame 
                should be shown in seconds. Making this value smaller 
                corresponds to a smoother animation but more memory usage 
                during generation and vice versa. Defaults to 0.05.
            last_frame_dur (float, optional): Specifies how long the last frame 
                should be shown additionally in seconds. Defaults to 1.0.
            fps_multiplier (float, optional): The fps are calculated so that 
                the gif runs approximately in real time. Change this value if 
                you want to make the gif slower (smaller than 1.0) or faster 
                (larger than 1.0). Defaults to 1.0.
            dpi (int, optional): Resolution of the plot. Defaults to 300.

        Returns:
            None
        """
        print("\tposfile_json: Animating (line) agent's position ...\n")

        linewidth = 2.5
        markersize = 6.0
        titlesize = 12
        legendsize = 'small'

        if not hasattr(self, 'grp_data'):
            self.read_pos_file()

        if not hasattr(self, 'vert_lim'):
            self.set_xy_lims()

        for tr in self.tr_ids:
            tmp = self.grp_data.get_group(tr)

            pos = tmp[['x', 'y']].to_numpy()
            time = tmp.to_numpy()[:, 1]
            last_time = time[-1]

            if last_frame_dur is not None:
                pos = np.vstack((pos, pos[-1, :]))
                time = np.append(time, last_time + last_frame_dur * fps_multiplier)

            time_bins = np.hstack((np.arange(time[0], time[-1], frame_dur), time[-1]))
            fps = int(time_bins.shape[0] / (time[-1] - time[0]) * fps_multiplier)

            fig, ax = plt.subplots()

            # Set limits and aspect
            ax.set_aspect("equal")
            ax.set_xlim(self.horiz_lim)
            ax.set_ylim(self.vert_lim)

            # Set spatial labels
            ax.set_xticks(ticks=self.horiz_lim, labels=self.horiz_lab)
            ax.set_yticks(ticks=self.vert_lim, labels=self.vert_lab)

            if show_obstacle:
                add_obstacles(ax, self.env_dict)

            if not self.hide_goal:
                add_goal_zone(ax, os.path.join(self.data_path, 'trials_params.dat'), tr=tr)

            if self.sim_env == 'tmaze':
                add_tmaze(ax, self.env_dict)

            traj, = ax.plot(
                0, 0, color='blue', label='traversed path',
                lw=linewidth, zorder=1
            )
            current = ax.scatter(
                0, 0, marker='o', color='red', label='final position',
                zorder=2, s=markersize**2
            )
            ax.scatter(
                pos[0, 0], pos[0, 1], marker='o', color='green',
                label='initial position', zorder=3, s=markersize**2
            )

            if add_legend:
                ax.legend(
                    bbox_to_anchor=(1, 0.5), loc='center left', fontsize=legendsize
                )

            def animate(i):
                t = time_bins[i]

                pos_x = pos[time <= t, 0]
                pos_y = pos[time <= t, 1]

                traj.set_data(pos_x, pos_y)
                current.set_offsets(np.c_[pos_x[-1], pos_y[-1]])

                if title and t <= last_time:
                    if reset_time:
                        ax.set_title(
                            f"Trial: {tr} Time: {t-time[0]:.1f}", fontsize=titlesize
                        )
                    else:
                        ax.set_title(
                            f"Trial: {tr} Time: {t:.1f}", fontsize=titlesize
                        )

                return traj, current

            plt.subplots_adjust(right=0.75)
            anim = FuncAnimation(
                fig, animate, blit=False, repeat=False,
                interval=frame_dur * 1000, frames=time_bins.shape[0]
            )
            anim.save(
                os.path.join(self.fig_path, f'animated_trajectory-tr{int(tr)}.gif'),
                dpi=dpi, writer=PillowWriter(fps=fps)
            )
            ax.cla()
            plt.close(fig)
