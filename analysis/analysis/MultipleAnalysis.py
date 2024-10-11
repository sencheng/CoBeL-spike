#!/usr/bin/env python3

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Wedge
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation, writers

from .SpikefileNest import SpikefileNest
from .Weight import Weight
from .PositionFile import PositionFile
from .RepresentationVisualization import RepresentationVisualization as RV
from .PlotHelper import add_goal_zone, add_obstacles, add_tmaze


class MultipleAnalysis:
    """
    Generates an animation for firing rates, agent's location, and weights.

    Parameters
    ----------
    path_dict : dict
        next_data_path : str
            The path of the spikefile.
        other_data_path : str
            The path of the positionfile.
    flname_dict : array
        1D or 2D array specifying the different subplots and their position in the animation.
        The array should contain 'loc'. Other subplots are e.g. 'action', 'action_ring', 'place',
        'grid', 'border', 'BLANK'.
        Example: flname_dict=[['loc', 'action', 'place'], ['weight', 'grid', 'border']]
    fig_path : str
    """

    def __init__(self, path, flname_dict=[], fig_path=None, max_num_anims_per_session=5):
        self.path = path
        self.fig_path = fig_path
        self.max_num_anims_per_session = max_num_anims_per_session
        self.flname_dict = np.array(flname_dict)
        self.entries = {}
        self.tr = 1

        if any("grid" in subplot for subplot in flname_dict):
            self.population = "grid"
        elif any("place" in subplot for subplot in flname_dict):
            self.population = "place"

        self._read_sim_data()
        self._init_structure()

    def _init_structure(self):
        for entry in self.flname_dict.flatten():
            if entry in self.entries or entry == 'BLANK':
                continue
            elif entry == 'loc':
                pop = PositionFile(self.path, 'locs_time.dat', fig_path=self.fig_path)
                pop.read_pos_file()
            elif entry in ['weight', 'p_stack']:
                pop = Weight(self.path, self.population, fig_path=self.fig_path)
                pop.read_files()
            elif entry == 'fmap':
                gpos_path = '../data/test/'
                pop = RV(
                    gpos_path,
                    'grid_pos.json',
                    fig_path=self.fig_path,
                    cal_covarage=True,
                    firing_rate_quota=.75,
                    param_file={
                        'sim': 'sim_params.json',
                        'net': 'network_params_spikingnet.json',
                        'env': 'env_params.json'
                    },
                    resolution=50
                )
            else:
                try:
                    if entry == 'action_ring':
                        pop = SpikefileNest(self.path, f'action-0.gdf', fig_path=self.fig_path)
                        weight = Weight(self.path, self.population, fig_path=self.fig_path)
                        weight.read_files()
                    else:
                        pop = SpikefileNest(self.path, f'{entry}-0.gdf', fig_path=self.fig_path)

                    if entry in ['action_ring', 'action']:
                        pop.check_pop_size('actionID_dir.dat')

                    pop.calc_avg_fr()
                except FileNotFoundError:
                    print(f'{entry} cell population not found for this simulation!')
                    i = np.nonzero(self.flname_dict == entry)
                    self.flname_dict[i][0] = 'BLANK'
                    continue

            if entry == 'action_ring':
                self.entries[entry] = [weight, pop]
            else:
                self.entries[entry] = [pop]

    def _read_sim_data(self, sim_file='sim_params.json', env_file='env_params.json'):
        self.sim_file_path = os.path.join(self.path, sim_file)
        with open(self.sim_file_path, "r") as f:
            self.sim_dict = json.load(f)
        self.env_file_path = os.path.join(self.path, env_file)
        with open(self.env_file_path, "r") as f:
            self.env_dict = json.load(f)
        self.sim_env = self.env_dict['sim_env']
        self.env_limit_dic = self.env_dict['environment'][self.sim_env]
        if self.sim_env in ['openfield', 'tmaze']:
            self.xmin_position = float(self.env_limit_dic['xmin_position'])
            self.xmax_position = float(self.env_limit_dic['xmax_position'])
            self.ymin_position = float(self.env_limit_dic['ymin_position'])
            self.ymax_position = float(self.env_limit_dic['ymax_position'])
            self.x_buffer = (abs(self.xmin_position) + abs(self.xmax_position)) * 0.1
            self.y_buffer = (abs(self.ymin_position) + abs(self.ymax_position)) * 0.1

            self.opn_fld_xlim = np.array([self.xmin_position, self.xmax_position])
            self.opn_fld_ylim = np.array([self.ymin_position, self.ymax_position])
            self.hide_goal = self.env_dict['environment']['goal']['hide_goal']
            self.horiz_lab = np.array([0, self.xmax_position - self.xmin_position])
            self.vert_lab = np.array([0, self.ymax_position - self.ymin_position])
        else:
            print(f"environment {self.sim_env} undefined")
            raise NotImplementedError

    def _border(self, y_min, y_max, x_min, x_max, key):
        cl = 'dimgray'  # cl = "k"
        lw_ = 2
        ls = ':'  # ls = 'solid'

        self.ax[key].vlines(x_min, y_min, y_max, color=cl, lw=lw_, linestyle=ls)
        self.ax[key].vlines(x_max, y_min, y_max, color=cl, lw=lw_, linestyle=ls)
        self.ax[key].hlines(y_min, x_min, x_max, color=cl, lw=lw_, linestyle=ls)
        self.ax[key].hlines(y_max, x_min, x_max, color=cl, lw=lw_, linestyle=ls)

    def _action_ring(self, item, key):
        item[0].get_data_smth(self.time_bins)
        self.ax[key].set_title('Network activity')
        fr_range = (item[1].fr_vec_smth.min(), item[1].fr_vec_smth.max())

        # Parameters of the donuts
        center = (0, 0)
        outer_outer_radius = 1.0
        inner_outer_radius = 0.7
        outer_inner_radius = 0.6
        inner_inner_radius = 0.0
        label_placement = 1.1

        # Define the number of data points (entries)
        num_entries = 40

        # Calculate the angles for the wedges
        angles = (np.linspace(360, 0, num_entries, endpoint=False) + 85) % 360

        # Create a colormap for the heatmap
        cmap = plt.get_cmap('viridis')

        # Get min and max of each dataset
        min1, max1 = fr_range[0], fr_range[1]
        min2, max2 = item[0].vmin, item[0].vmax

        # Normalize the data
        norm_outer = Normalize(vmin=min1, vmax=max1)
        norm_inner = Normalize(vmin=min2, vmax=max2)

        # Create a list to hold the Wedge patches for each entry
        wedges_outer = []
        color = cmap(norm_outer(0))  # Initialize with a dummy value (0)
        for i in range(num_entries):
            outer_wedge = Wedge(
                center, outer_outer_radius, angles[i], angles[i] + 360 / num_entries,
                width=outer_outer_radius - inner_outer_radius, facecolor=color
            )
            self.ax[key].add_patch(outer_wedge)
            wedges_outer.append(outer_wedge)
            if i % 5 == 0:
                label_x = label_placement * np.cos(np.deg2rad(angles[i] + 360 / num_entries / 2))
                label_y = label_placement * np.sin(np.deg2rad(angles[i] + 360 / num_entries / 2))
                self.ax[key].text(label_x, label_y, str(i), ha='center', va='center', fontsize=8, color='black')

        # Create wedges for the INNER donut
        wedges_inner = []
        for i in range(num_entries):
            color = cmap(norm_inner(0))  # Initialize with a dummy value (0)
            inner_wedge = Wedge(
                center, outer_inner_radius, angles[i], angles[i] + 360 / num_entries,
                width=outer_inner_radius - inner_inner_radius, facecolor=color
            )
            self.ax[key].add_patch(inner_wedge)
            wedges_inner.append(inner_wedge)

        # Set aspect ratio to be equal so the plot is circular
        self.ax[key].set_aspect('equal')

        # Set axis limits
        self.ax[key].set_xlim(-1.2, 1.2)
        self.ax[key].set_ylim(-1.2, 1.2)

        # Remove axis labels and ticks
        self.ax[key].axis('off')

        # Create a custom colorbar that spans both data ranges
        norm = Normalize(vmin=0, vmax=1)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=self.ax[key], location='right')
        ticks = [0, 0.5, 1]
        labels = [f"{min1:.1f}\n{min2:.1f}", f"{(max1 + min1) / 2:.1f}\n{(max2 + min2) / 2:.1f}", f"{max1:.1f}\n{max2:.1f}"]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(labels)
        cbar.ax.set_title(f'Action\n{self.population[0].upper() + self.population[1:]}', loc='right')

        self.entries[key].append(cmap)
        self.entries[key].append(wedges_outer)
        self.entries[key].append(norm_outer)
        self.entries[key].append(wedges_inner)
        self.entries[key].append(norm_inner)

    def __anim_init(self, trial_start_time):
        """
        Initialization function for the 'animation' module of 'matplotlib'.
        """
        st_time = self.time_bins[0]
        for key, item in self.entries.items():
            if key == "loc":
                self.ax[key].set_title('trajectory')
                self.ax[key].set_aspect("equal")
                self.ax[key].set_box_aspect(1)
                self.ax[key].set_xlim(self.xmin_position - self.x_buffer, self.xmax_position + self.x_buffer)
                self.ax[key].set_ylim(self.ymin_position - self.y_buffer, self.ymax_position + self.y_buffer)
                self.ax[key].set_xticks(ticks=self.opn_fld_xlim, labels=self.horiz_lab)
                self.ax[key].set_yticks(ticks=self.opn_fld_ylim, labels=self.vert_lab)
                if self.sim_env == 'tmaze':
                    add_tmaze(self.ax[key], self.env_dict)
                self.dot_pos, = self.ax[key].plot(item[0].pos_xy[0, 0], item[0].pos_xy[0, 1], color='blue', marker='o')
                self.line_pos, = self.ax[key].plot(item[0].pos_xy[0, 0], item[0].pos_xy[0, 1], color='blue', linestyle='-')
            elif key == "weight":
                self.ax[key].set_title('vector field')
                self.ax[key].set_aspect("equal")
                self.ax[key].set_box_aspect(1)
                self.ax[key].set_xlim(self.xmin_position - self.x_buffer, self.xmax_position + self.x_buffer)
                self.ax[key].set_ylim(self.ymin_position - self.y_buffer, self.ymax_position + self.y_buffer)
                weight_field = self.entries["weight"][0].vector_field(time=trial_start_time)
                self.vec_field = self.ax["weight"].quiver(
                    self.entries["weight"][0].place_locs[:, 0],
                    self.entries["weight"][0].place_locs[:, 1],
                    weight_field[0, :], weight_field[1, :],
                    scale_units='xy', angles='xy', scale=.2
                )
                self.ax[key].set_xticks(ticks=self.opn_fld_xlim, labels=self.horiz_lab)
                self.ax[key].set_yticks(ticks=self.opn_fld_ylim, labels=self.vert_lab)
                if self.sim_env == 'tmaze':
                    add_tmaze(self.ax[key], self.env_dict)
            elif key == "p_stack":
                self.ax[key].set_title(f'{self.population[0].upper() + self.population[1:]} vector stack')
                self.ax[key].set_aspect("equal")
                self.ax[key].set_box_aspect(1)
                self.ax[key].set_xlim((item[0].place_locs[:, 0].min() - .5, item[0].place_locs[:, 0].max() + .5))
                self.ax[key].set_ylim((item[0].place_locs[:, 0].min() - .5, item[0].place_locs[:, 0].max() + .5))
                field_pos, arrows, n = item[0].calc_vector_field_stack_at_time(trial_start_time)
                self.p_stack = self.ax['p_stack'].quiver(
                    field_pos[0, :], field_pos[1, :],
                    arrows[0, :] / n, arrows[1, :] / n,
                    scale_units='xy', angles='xy', scale=1
                )
                self.ax[key].set_xticks(ticks=self.opn_fld_xlim, labels=self.horiz_lab)
                self.ax[key].set_yticks(ticks=self.opn_fld_ylim, labels=self.vert_lab)
                if self.sim_env == 'tmaze':
                    add_tmaze(self.ax[key], self.env_dict)
            elif key == 'fmap':
                self.ax[key].set_title('firing map')
                self.ax[key].set_aspect((self.opn_fld_ylim[1] - self.opn_fld_ylim[0]) / (self.opn_fld_xlim[1] - self.opn_fld_xlim[0]))
                self.ax[key].set_box_aspect(1)
                self.ax[key].set_xlim(self.xmin_position, self.xmax_position)
                self.ax[key].set_ylim(self.ymin_position, self.ymax_position)
                x, y, field = self.entries['fmap'][0].plot_firing_maps_on_ax(self.ax['fmap'])
                pcolor = self.ax['fmap'].pcolormesh(x, y, field)
                self.entries['fmap'].append(pcolor)
                plt.colorbar(pcolor, ax=self.ax['fmap'], orientation="vertical")
                self.ax[key].set_xticks(ticks=self.opn_fld_xlim, labels=self.horiz_lab)
                self.ax[key].set_yticks(ticks=self.opn_fld_ylim, labels=self.vert_lab)
                if self.sim_env == 'tmaze':
                    add_tmaze(self.ax[key], self.env_dict)
            elif key == 'action_ring':
                self._action_ring(item, key)
            else:
                self.ax[key].set_title(f'{key} cells')
                fr_range = (item[0].fr_vec_smth.min(), item[0].fr_vec_smth.max())
                fr_mat = item[0].fr_vec_smth[:, (item[0].hist_edges >= st_time) & (item[0].hist_edges < st_time + self.frame_dur_fr)]
                time_vec = item[0].hist_edges[(item[0].hist_edges >= st_time) & (item[0].hist_edges < st_time + self.frame_dur_fr)]
                time_vec -= st_time
                neuron_ids = np.arange(item[0].neuron_ids.min(), item[0].neuron_ids.max() + 1)
                if fr_mat.size == 0:
                    pcolor = self.ax[key].pcolormesh(
                        np.arange(0, self.frame_dur_fr), neuron_ids, 
                        np.zeros((fr_mat.shape[0], self.frame_dur_fr)), 
                        edgecolors=None, vmin=fr_range[0], vmax=fr_range[1], shading='auto'
                    )
                    self.entries[key].append(pcolor)
                    self.entries[key].append(self.frame_dur_fr)
                else:
                    pcolor = self.ax[key].pcolormesh(time_vec, neuron_ids, fr_mat, edgecolors=None, vmin=fr_range[0], vmax=fr_range[1], shading='auto')
                    self.entries[key].append(pcolor)
                    self.entries[key].append(len(time_vec))
                plt.colorbar(pcolor, ax=self.ax[key], orientation="vertical")

        border_flag = True
        if border_flag:
            if 'weight' in self.entries:
                self._border(self.opn_fld_ylim[0], self.opn_fld_ylim[1], self.opn_fld_xlim[0], self.opn_fld_xlim[1], "weight")
            if 'loc' in self.entries:
                self._border(self.opn_fld_ylim[0], self.opn_fld_ylim[1], self.opn_fld_xlim[0], self.opn_fld_xlim[1], "loc")

        if not self.hide_goal:
            add_goal_zone(self.ax["loc"], os.path.join(self.sim_dict['data_path'], 'trials_params.dat'), self.tr)

        if self.env_dict['environment']['obstacles']['flag']:
            add_obstacles(self.ax["loc"], self.env_dict)

    def __update(self, cnt):
        """
        Update function for the 'animation' module of 'matplotlib'.

        Parameters
        ----------
        cnt : int
            The iterator (implicitly) provided by the 'animation' method
            of 'matplotlib'.

        Returns
        -------
        Outputs of the plot, pcolor, etc.
        """
        arr = []

        sys.stdout.write(f'\r{cnt / self.nframes * 100:.1f}%')
        sys.stdout.flush()
        T = self.time_bins[cnt]
        self.fig.suptitle(f'Time = {np.around(T / 1000, 1)} s')

        for key, item in self.entries.items():
            if key == "loc":
                if cnt != 0:
                    pos_x = item[0].pos_xy[(item[0].pos_time <= T) & (item[0].pos_time >= self.time_bins[0]), 0]
                    pos_y = item[0].pos_xy[(item[0].pos_time <= T) & (item[0].pos_time >= self.time_bins[0]), 1]
                    self.line_pos.set_data(pos_x, pos_y)
                    self.dot_pos.set_data(pos_x[-1], pos_y[-1])
                    arr.append(self.line_pos)
                    arr.append(self.dot_pos)
            elif key == "weight":
                continue  # trying this out- vector fields should generally only update at the moment between animations
#                if (cnt == 0) | (cnt == self.nframes - 1):
#                    weight_field = item[0].vector_field(time=T)
#                    self.vec_field.set_UVC(weight_field[0, :], weight_field[1, :])
#                    self.ax[key].set_title('Vector field, update at {} s'.format(np.around(T/1000, 1)))
#                    arr.append(self.vec_field)
            elif key == 'fmap':
                continue  # firing maps don't change during trials
            elif key == 'p_stack':
                continue  # vector fields should generally only update at the moment between animations
            elif key == 'action_ring':
                previousT = 0
                if cnt != 0:
                    previousT = self.time_bins[cnt - 1]

                fr_mat = item[1].fr_vec_smth[:, (item[1].hist_edges >= previousT) & (item[1].hist_edges < T)]
                if fr_mat.shape[1] >= 1:
                    fr_mat = np.mean(fr_mat, 1)[:, np.newaxis]
                else:
                    zeros = np.zeros((fr_mat.shape[0], 1 - fr_mat.shape[1]))
                    fr_mat = np.concatenate((zeros, fr_mat), axis=1)

                data_inner = item[0].data_smth[:, cnt]

                # Update the colors of the wedges for the current frame
                for i, (wedge_outer, wedge_inner) in enumerate(zip(item[3], item[5])):
                    color_outer = item[2](item[4](fr_mat[i]))
                    color_inner = item[2](item[6](data_inner[i]))
                    wedge_outer.set_facecolor(color_outer)
                    wedge_inner.set_facecolor(color_inner)
                    arr.append(wedge_outer)
                    arr.append(wedge_inner)
            else:
                fr_mat = item[0].fr_vec_smth[:, (item[0].hist_edges >= T - item[2]) & (item[0].hist_edges < T)]
                if fr_mat.shape[1] != item[2]:
                    zeros = np.zeros((fr_mat.shape[0], item[2] - fr_mat.shape[1]))
                    if self.time_bins[cnt] < 500:
                        fr_mat = np.concatenate((zeros, fr_mat), axis=1)
                    else:
                        fr_mat = np.concatenate((fr_mat, zeros), axis=1)
                item[1].set_array(fr_mat.ravel())
                arr.append(item[1])
        return arr

    def __animate(self, time_bins, mov_st_time, frame_dur=50, frame_dur_fr=500, dpi=10, fl_suffix=''):
        """
        Helper method to create the animation by calling
        '__anim_init' and '__update' method.

        Parameters
        ----------
        frame_dur : float, optional
            Duration of a frame in ms. The default is 50.
        mov_st_time : float, optional
            Starting time point of movie in ms. The default is 0.
        mov_end_time : float, optional
            Last time point of the movie in ms. The default is 10e3.
        frame_dur_fr : float, optional
            Determines how long into the past (in ms) the time-resolved
            firing rate should be computed. The default is 500.
        bin_size : float, optional
            Bin width (in ms) for computing the time-resolved firing rate.
            The default is 1.
        fl_suffix : str, optional
            The string, which will be added to the end of the file.
            It can, e.g., be used to distinguish outcome of different simulations.
            The default is ''.

        Returns
        -------
        None.
        """
        SMALL_SIZE = 12
        MEDIUM_SIZE = 14
        BIGGER_SIZE = 16

        plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        self.frame_dur = frame_dur
        self.frame_dur_fr = frame_dur_fr
        self.time_bins = time_bins

        self.nframes = self.time_bins.size
        self.fig, self.ax = plt.subplot_mosaic(
            self.flname_dict,
            empty_sentinel="BLANK",
            figsize=(16, 9),
            layout="tight"
        )
        self.__anim_init(mov_st_time)
        anim = FuncAnimation(
            self.fig,
            self.__update,
            frames=self.nframes,
            interval=50,
            blit=False,
            repeat=False
        )

        writer = writers['ffmpeg'](fps=15, metadata=dict(artist='Nejad'), bitrate=1800)
        path = os.path.join(self.fig_path, f'animation-{fl_suffix}.mp4')
        anim.save(path, writer=writer, dpi=dpi)
        plt.close(self.fig)

        for key, item in self.entries.items():
            if key in ['loc', 'weight', 'fmap', 'p_stack']:
                continue
            if key == 'action_ring':
                item.pop()
                item.pop()
                item.pop()
            item.pop()
            item.pop()

    def animate_tr_by_tr_loc(self, frame_dur=100, trials=[], summary=True, dpi=100):
        """
        Create animation based on the timestamps (separating the trials)
        derived from the agent's position data.

        Parameters
        ----------
        frame_dur : float, optional
            Duration of individual frames in ms. The default is 100.
        summary : bool, optional
            Whether to reduce the computation time by cutting down
            the number of trials. The default is True.

        Returns
        -------
        None.
        """
        trial_times = self.entries["loc"][0].get_times_from_pos_file()
        start_times = np.asarray([trial['start_time'] for trial in trial_times])
        end_times = np.asarray([trial['end_time'] for trial in trial_times])
        max_trs = self.sim_dict['max_num_trs']

        if len(start_times) > max_trs:
            start_times = start_times[0:max_trs]
            end_times = end_times[0:max_trs]

        if trials:
            trials = np.asarray(trials) - 1
            start_times = start_times[trials]
            end_times = end_times[trials]
        elif summary and (len(start_times) >= self.max_num_anims_per_session):
            print(f'Too many successful trials! Animating only for the first {self.max_num_anims_per_session} trials and the last one...')
            start_times = np.hstack((start_times[0:5], start_times[-1]))
            end_times = np.hstack((end_times[0:5], end_times[-1]))
            trials = np.hstack([np.arange(end_times.shape[0] - 1), [max_trs - 1]])
        else:
            trials = np.arange(max_trs)

        # Calculate time_bins
        global_time_bins = []
        for st, et in zip(start_times, end_times):
            time_bins = np.hstack((np.arange(st, et, frame_dur), et))
            global_time_bins.append(time_bins)

        if 'action_ring' in self.entries:
            self.entries['action_ring'][0].get_data_range(global_time_bins)

        for t, ts, tb in zip(trials, start_times, global_time_bins):
            self.tr = t + 1
            print(f'\nRendering animation for trial #{self.tr} distinguished by location data')
            self.__animate(time_bins=tb, mov_st_time=ts, frame_dur=frame_dur, dpi=dpi, fl_suffix=f'-tr{self.tr}')
