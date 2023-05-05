#!/usr/bin/env python3

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from .SpikefileNest import SpikefileNest
from .Weight import Weight
from .PositionFileAdapter import PositionFileAdapter
from .PlotHelper import add_goal_zone, add_obstacles


class MultipleAnalysis():
    """
    Generates an animation for firing rates, agent's location and weights.

    Parameters
    ----------
    path_dict: Dictionary
        next_data_path: String
            The path of the spikefile.
        other_data_path: String
            The path of the positionfile.
    flname_dict: Array
        1d or 2d Array specifying the different subplots and their position in the animation. 
        The Array should contain 'loc' and 'weight'.
        Other subplots are e.g. 'action', 'place', 'grid' or 'border'. 
        Example: flname_dict=[['loc', 'action', 'place']
                             ,['weight', 'grid', 'border']]

    fig_path: String
    dpi: Integer
        Resolution of the animation
    """
    def __init__(self, 
                 path_dict={'nest_data_path': '', 'other_data_path': ''}, 
                 flname_dict=[], 
                 fig_path=None,
                 dpi=100):
        self.fig_path = fig_path
        self.dpi = dpi
        self.max_num_anims_per_session = 5

        self._read_sim_data(data_path='../openfield', sim_file='sim_params.json')
        self.flname_dict = np.array(flname_dict)
        self.flname_dict_shape = self.flname_dict.shape
        self.entries = {}
        self.empty =[]
        for y, x in np.ndindex(self.flname_dict.shape):
            if self.flname_dict[y, x] == 'loc':
                pop = PositionFileAdapter(path_dict['other_data_path'], 'agents_location.dat', fig_path=fig_path)
                pop.read_pos_file()
            elif self.flname_dict[y, x] == 'weight':
                pop = Weight(path_dict['nest_data_path'], 'place-0.csv', fig_path=fig_path)
                pop.read_files()
            else:
                try:
                    pop = SpikefileNest(path_dict['nest_data_path'], f'{self.flname_dict[y, x]}-0.gdf', fig_path=fig_path)
                    pop.calc_avg_fr()
                except:
                    print(f"{self.flname_dict[y, x]}-0.gdf does not exist")
                    self.empty.append([y, x])
                    continue
            self.entries[self.flname_dict[y, x]] = [[y, x], pop]     
        
        

    def _read_sim_data(self, data_path='../openfield', sim_file='sim_params.json'):
        self.sim_file_path = os.path.join(data_path, sim_file)
        with open(self.sim_file_path, "r") as f:
            self.sim_dict = json.load(f)
        # with open(sim_file, 'r') as fl:
        #     net_dict = json.load(fl)
        self.sim_env = self.sim_dict['sim_env']
        self.env_limit_dic = self.sim_dict['environment'][self.sim_env]
        if self.sim_env == 'openfield':
            self.xmin_position = float(self.env_limit_dic['xmin_position'])
            self.xmax_position = float(self.env_limit_dic['xmax_position'])
            self.ymin_position = float(self.env_limit_dic['ymin_position'])
            self.ymax_position = float(self.env_limit_dic['ymax_position'])
            self.opn_fld_xlim = np.array([self.xmin_position, self.xmax_position])
            self.opn_fld_ylim = np.array([self.ymin_position, self.ymax_position])
            self.hide_goal = self.sim_dict['goal']['hide_goal']
            self.reward_recep_field = self.sim_dict['goal']['reward_recep_field']
            self.goal_x = self.sim_dict['goal']['x_position']
            self.goal_y = self.sim_dict['goal']['y_position']
        else:
            print("environment {} undefined".format(self.sim_env))
        self.opn_fld_xlim = np.array([self.xmin_position, self.xmax_position])
        self.opn_fld_ylim = np.array([self.ymin_position, self.ymax_position])
        self.hide_goal = self.sim_dict['goal']['hide_goal']
        self.reward_recep_field = self.sim_dict['goal']['reward_recep_field']
        self.goal_x = self.sim_dict['goal']['x_position']
        self.goal_y = self.sim_dict['goal']['y_position']


    def _border(self, y_min, y_max, x_min, x_max, ax1, ax2):
        cl = 'dimgray' #cl = "k"
        lw_ = 2
        ls = ':' #ls = 'solid'

        self.ax[ax1, ax2].vlines(y_min, x_min, x_max, color=cl, lw=lw_, linestyle=ls)
        self.ax[ax1, ax2].vlines(y_max, x_min, x_max, color=cl, lw=lw_, linestyle=ls)
        self.ax[ax1, ax2].hlines(x_min, y_min, y_max, color=cl, lw=lw_, linestyle=ls)
        self.ax[ax1, ax2].hlines(x_max, y_min, y_max, color=cl, lw=lw_, linestyle=ls)  


    def __anim_init(self):
        """ 
        Initialization function for the 'animation' module of 'matplotlib'
        """
        st_time = self.time_bins[0]
        for key, value in self.entries.items():
            if key == "loc":
                self.ax[value[0][0], value[0][1]].set_title('trajectory')
                self.ax[value[0][0], value[0][1]].set_aspect((self.opn_fld_ylim[1]-self.opn_fld_ylim[0])/(self.opn_fld_xlim[1]-self.opn_fld_xlim[0]))
                self.ax[value[0][0], value[0][1]].set_xlim((self.entries["weight"][1].place_locs[:, 0].min() - .5, self.entries["weight"][1].place_locs[:, 0].max() + .5))
                self.ax[value[0][0], value[0][1]].set_ylim((self.entries["weight"][1].place_locs[:, 0].min() - .5, self.entries["weight"][1].place_locs[:, 0].max() + .5))
            elif key == "weight":
                self.ax[value[0][0], value[0][1]].set_title('vector field')
                self.ax[value[0][0], value[0][1]].set_aspect((self.opn_fld_ylim[1]-self.opn_fld_ylim[0])/(self.opn_fld_xlim[1]-self.opn_fld_xlim[0]))
                self.ax[value[0][0], value[0][1]].set_xlim((self.entries["weight"][1].place_locs[:, 0].min() - .5, self.entries["weight"][1].place_locs[:, 0].max() + .5))
                self.ax[value[0][0], value[0][1]].set_ylim((self.entries["weight"][1].place_locs[:, 0].min() - .5, self.entries["weight"][1].place_locs[:, 0].max() + .5))
            else:
                self.ax[value[0][0], value[0][1]].set_title(f'{key} cells')
                fr_range = (self.entries[key][1].fr_vec_smth.min(), self.entries[key][1].fr_vec_smth.max())
                fr_mat = self.entries[key][1].fr_vec_smth[:,(self.entries[key][1].hist_edges >= st_time) & (self.entries[key][1].hist_edges < st_time + self.frame_dur_fr)]
                time_vec = self.entries[key][1].hist_edges[(self.entries[key][1].hist_edges >= st_time) & (self.entries[key][1].hist_edges < st_time + self.frame_dur_fr)]
                time_vec -= st_time
                neuron_ids = np.arange(self.entries[key][1].neuron_ids.min(), self.entries[key][1].neuron_ids.max() + 1)
                pcolor = self.ax[self.entries[key][0][0], self.entries[key][0][1]].pcolormesh(time_vec, neuron_ids, fr_mat, edgecolors=None, vmin=fr_range[0], vmax=fr_range[1], shading='auto')
                plt.colorbar(pcolor, ax=self.ax[self.entries[key][0][0], self.entries[key][0][1]])
                self.entries[key].append(pcolor)
                self.entries[key].append(len(time_vec))

        self.dot_pos, = self.ax[self.entries["loc"][0][0], self.entries["loc"][0][1]].plot(self.entries["loc"][1].pos_xy[0, 0], self.entries["loc"][1].pos_xy[0, 1], color='blue', marker='o')
        self.line_pos, = self.ax[self.entries["loc"][0][0], self.entries["loc"][0][1]].plot(self.entries["loc"][1].pos_xy[0, 0], self.entries["loc"][1].pos_xy[0, 1], color='blue', linestyle='-')
        weight_field = self.entries["weight"][1].vector_field(time=st_time)
        self.vec_field = self.ax[self.entries["weight"][0][0], self.entries["weight"][0][1]].quiver(self.entries["weight"][1].place_locs[:, 0], 
                                                                                                    self.entries["weight"][1].place_locs[:, 1],
                                                                                                    weight_field[0, :], weight_field[1, :], 
                                                                                                    scale_units='xy', angles='xy', scale=.2)
        
        border_flag = True
        if border_flag:
            self._border(self.entries["weight"][1].place_locs[:, 0].min(), self.entries["weight"][1].place_locs[:, 0].max(), self.entries["weight"][1].place_locs[:, 1].min(), self.entries["weight"][1].place_locs[:, 1].max(), self.entries["weight"][0][0], self.entries["weight"][0][1])
            self._border(self.opn_fld_ylim[0], self.opn_fld_ylim[1], self.opn_fld_xlim[0], self.opn_fld_xlim[1], self.entries["loc"][0][0], self.entries["loc"][0][1])

        if not self.sim_dict['goal']['hide_goal']:
            add_goal_zone(self.ax[0, 0], self.sim_dict)
            
        if self.sim_dict['environment']['obstacles']['flag']:
            add_obstacles(self.ax[0, 0], self.sim_dict)
            

    def __update(self, cnt):
        """ 
        update function for the 'animation' module of 'matplotlib'
        
        Parameter
        ----------
        cnt : int
            The iterator (implicitly) provided by the 'animation' method
            of the 'matplotlib'.
            
        Returns
        -------
        Outputs of the plot, pcolor and etc.
        """
        arr = []
        
        sys.stdout.write('\r{:.1f}%'.format(cnt / self.nframes * 100))
        sys.stdout.flush()
        T = self.time_bins[cnt]
        self.fig.suptitle('Time = {} s'.format(np.around(T/1000, 1)))
    
        for key, _ in self.entries.items():
            if key == "loc": 
                if cnt != 0:
                    self.line_pos.set_data(self.entries[key][1].pos_xy[(self.entries[key][1].pos_time <= T) & (self.entries[key][1].pos_time >= self.time_bins[0]), 0],
                                           self.entries[key][1].pos_xy[(self.entries[key][1].pos_time <= T) & (self.entries[key][1].pos_time >= self.time_bins[0]), 1])
                    self.dot_pos.set_data(self.entries[key][1].pos_xy[abs(self.entries[key][1].pos_time - T) < 1e-2, 0],
                                          self.entries[key][1].pos_xy[abs(self.entries[key][1].pos_time - T) < 1e-2, 1])
                    arr.append(self.line_pos)
                    arr.append(self.dot_pos)
            elif key == "weight":
                if (cnt == 0) | (cnt == self.nframes - 1):
                    weight_field = self.entries[key][1].vector_field(time=T)
                    self.vec_field.set_UVC(weight_field[0, :], weight_field[1, :])
                    self.ax[self.entries[key][0][0], self.entries[key][0][1]].set_title('Vector field, update at {} s'.format(np.around(T/1000, 1)))
                    arr.append(self.vec_field)
            else:
                if cnt != 0 and ((T - self.time_bins[0]) > self.frame_dur_fr):   
                    fr_mat = self.entries[key][1].fr_vec_smth[:,(self.entries[key][1].hist_edges >= T - self.entries[key][3]) & (self.entries[key][1].hist_edges < T)]
                    if fr_mat.shape[1] != self.entries[key][3]:
                        zeros = np.zeros((fr_mat.shape[0], self.entries[key][3]-fr_mat.shape[1]))
                        fr_mat = np.concatenate((fr_mat, zeros), axis=1)
                    self.entries[key][2].set_array(fr_mat.ravel())
                    arr.append(self.entries[key][2])
        return arr


    def __animate(self, frame_dur=50, mov_st_time=0, mov_end_time=10e3, frame_dur_fr=500, bin_size=1, fl_suffix=''):
        """ helper method to create the animation by calling
        '__anim_init' and '__update' method.

        Parameters
        ----------
        frame_dur : float, optional
            duration of a frame in ms. The default is 50.
        mov_st_time : float, optional
            starting time point of movie in ms. The default is 0.
        mov_end_time : float, optional
            last time point of the movie in ms. The default is 10e3.
        frame_dur_fr : float, optional
            determines how long into the past (in ms) the time-resolved
            firing rate should be computed. The default is 500.
        bin_size : float, optional
            bin width (in ms) for computing the time-resolved firing rate.
            The default is 1.
        fl_suffix : str, optional
            the string, which will be added to the end of the file.
            It can, e.g., be us        except ValueError:
            print("No dopamine file")ed to distinguish outcome of different simulations.
            The default is ''.

        Returns
        -------
        None.

        """
        self.frame_dur = frame_dur
        self.frame_dur_fr = frame_dur_fr
        self.time_bins = np.hstack((np.arange(mov_st_time, mov_end_time, frame_dur), mov_end_time))
        self.nframes = self.time_bins.size
        self.fig, self.ax = plt.subplots(nrows=self.flname_dict_shape[0], 
                                         ncols=self.flname_dict_shape[1], 
                                         figsize=(self.flname_dict_shape[1]*3.3, self.flname_dict_shape[0]*2.6), 
                                         layout="constrained",
                                         squeeze=False)
        if self.empty:
            for entry in self.empty:
                self.fig.delaxes(self.ax[entry[0]][entry[1]])

        self.__anim_init()
        anim = animation.FuncAnimation(self.fig, 
                                       self.__update, 
                                       frames=self.nframes, 
                                       interval=50, 
                                       blit=False,
                                       repeat=False)

        writer=animation.writers['ffmpeg'](fps=15, metadata=dict(artist='Nejad'), bitrate=1800)
        path = os.path.join(self.fig_path, 'pos-vecfield-frs-{}.mp4'.format(fl_suffix))
        anim.save(path, writer=writer, dpi=self.dpi)
        plt.close(self.fig)
        
        for key, _ in self.entries.items():
            if key == "loc" or key == "weight":
                continue
            self.entries[key].pop()
            self.entries[key].pop()


    def animate_tr_by_tr_DA(self, frame_dur=100):
        """ 
        create animation based on the time stamps (separating the trials)        
            derived from the spiking data of the dopamine neurons.
        Parameters
        ----------
        frame_dur : float, optional
            duration of individual frames in ms. The default is 100.

        Returns
        -------
        None.

        """
        dop_spks = self.dopamine.spk_data.get_group(0).time.values
        ids = np.hstack((np.where(np.abs(np.diff(dop_spks) - 0.1) > 1e-2)[0], -1))
        reward_times_end = np.hstack((dop_spks[ids], self.pos.pos_time[-1]))
        reward_times_st = np.hstack((0, dop_spks[ids]))
        for idx, ts in enumerate(reward_times_st):
            print('\nRendering animation for trial #{} distinguished by DA spike times'.format(idx + 1))
            self.__animate(frame_dur=frame_dur, mov_st_time=ts, mov_end_time=reward_times_end[idx],
                           fl_suffix='DA_spk-tr{}'.format(idx + 1))


    def animate_tr_by_tr_loc(self, frame_dur=100, summary=True, cells="grid"):
        """ 
        create animation based on the time stamps (separating the trials)        
            derived from the agent's position data.

        Parameters
        ----------
        frame_dur : float, optional
            duration of individual frames in ms. The default is 100.
        summary : bool, optional
            whether to reduce the computation time by cutting down
            the number of trials. The default is True.

        Returns
        -------
        None.

        """   
        init_pos = np.where((self.entries["loc"][1].pos_xy[:, 0] == self.sim_dict['start']['x_position']) &
                            (self.entries["loc"][1].pos_xy[:, 1] == self.sim_dict['start']['y_position']))[0]
        init_pos_ex_begin = self.entries["loc"][1].pos_time[init_pos[1:][np.diff(init_pos) > 1]]

        start_times = np.hstack((0.0, init_pos_ex_begin))
        end_times = np.hstack((init_pos_ex_begin, self.entries["loc"][1].pos_time[-1]))


        if summary & (start_times.size >= self.max_num_anims_per_session):
            print('Too many successful trials! Animating only for the first' + ' {} trials and the last one ...'.format(self.max_num_anims_per_session))
            start_times = np.hstack((start_times[0:5], start_times[-1]))
            end_times = np.hstack((end_times[0:5], end_times[-1]))
            
        for idx, (ts, es) in enumerate(zip(start_times, end_times)):
            print('\nRendering animation for trial #{} distinguished by location data'.format(idx + 1))
            self.__animate(frame_dur=frame_dur, mov_st_time=ts, mov_end_time=es, fl_suffix='locfl-tr{}'.format(idx + 1))


    def plot_rat_pos_tr_by_tr_loc(self, fl_suffix='line', frm='pdf'):
        """ 
        plotting the agent's position based on position file.
            It does not fit here and should move to a related class.

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
        init_pos = np.where((self.pos.pos_xy[:, 0] == 0) & (self.pos.pos_xy[:, 1] == 0))[0]
        init_pos_ex_begin = self.pos.pos_time[init_pos[1:][np.diff(init_pos) > 1]]
        start_times = np.hstack((0.0, init_pos_ex_begin))

        for idx, ts in enumerate(start_times):
            print("\nRat's trajectory for trial #{} determined by location data".format(idx + 1))

            fig, ax = plt.subplots()
            ax.set_xlim((self.weight.place_locs[:, 0].min() - .5, self.weight.place_locs[:, 0].max() + .5))

            ax.set_ylim((self.weight.place_locs[:, 1].min() - .5, self.weight.place_locs[:, 1].max() + .5))

            ax.plot(self.pos.pos_xy[:, 0], self.pos.pos_xy[:, 1], color='blue', label='traversed path')
            ax.plot(self.pos.pos_xy[0, 0], self.pos.pos_xy[0, 1], marker='o', color='green', label='initial position')
            ax.plot(self.pos.pos_xy[-1, 0], self.pos.pos_xy[-1, 1], marker='o', color='red', label='final position')

            ax.hlines(self.opn_fld_ylim[0], self.opn_fld_xlim[0], self.opn_fld_xlim[1], color='dimgray',
                      label='environment')
            ax.hlines(self.opn_fld_ylim[1], self.opn_fld_xlim[0], self.opn_fld_xlim[1], color='dimgray')
            ax.vlines(self.opn_fld_xlim[0], self.opn_fld_ylim[0], self.opn_fld_ylim[1], color='dimgray')
            ax.vlines(self.opn_fld_xlim[1], self.opn_fld_ylim[0], self.opn_fld_ylim[1], color='dimgray')
            ax.set_title('trajectory trial:{}'.format(idx + 1))

            ax.set_xlabel('x (a.u.)')
            ax.set_ylabel('y (a.u.)')
            ax.legend()
            fig.savefig(os.path.join(self.fig_path, 'ratpos-{:s}-tr{:d}.{}'.format(fl_suffix, idx + 1, frm)),
                        format=frm)
            plt.close(fig)
