"""
this file plots vector fields
"""
import os
import copy
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .SpikefileNest import SpikefileNest
from .PlotHelper import add_goal_zone

class Weight():
    """
    Represents a weight file
    """

    def __init__(self, data_path, filename, fig_path=None, times=[0]):
        print("Starting connections weights data from {} ...\n".format(filename))
        self.cell_type = filename.split('-')[0]
        self.file_path = os.path.join(data_path, filename)
        self.data_path = data_path
        if not os.path.exists(self.file_path):
            print('Could not find {}. Check whether the data file exists'
                  ' and whether the path to the file is correct!'.format(filename))
        if fig_path is not None:
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)
        self.fig_path = fig_path
        self.read_sim_data()
        # Used to rewrite the environment limits to start from 0
        env_limits = self.sim_dict['environment']['openfield']
        self.h_limit = [env_limits['xmin_position'], env_limits['xmax_position']]
        self.v_limit = [env_limits['ymin_position'], env_limits['ymax_position']]
        self.h_label = [0, self.h_limit[1] - self.h_limit[0]]
        self.v_label = [0, self.v_limit[1] - self.v_limit[0]]
        
        self.times = times

    def read_sim_data(self, data_path='../openfield',
                      sim_file='sim_params.json'):
        """
        reads the simulation parameters from the json file

        Parameters
        ----------
        data_path : string
            DESCRIPTION. The default is '../openfield'.
        sim_file : TYPE, optional
            DESCRIPTION. The default is 'sim_params.json'.

        Returns
        -------
        None.

        """
        self.sim_file_path = os.path.join(data_path, sim_file)
        with open(self.sim_file_path, "r") as f:
            self.sim_dict = json.load(f)

    def read_files(self):
        """
        Reads the weight, action vector and place field files and 
        writes them into the object.
        """
        print('\tWeight: Reading weight file: {}\n'.format(self.file_path))

        data = pd.read_csv(self.file_path, sep='\t',
                           names=['pre', 'post', 'time', 'weight', 'N'])

        self.data = data.drop('N', axis=1)

        self.get_init_ws()
        self.get_action_vecs()
        self.get_firing_fields(filename=(self.cell_type + 'ID_pos.dat'))

        w_min = data.weight.values.min()
        w_max = data.weight.values.max()
        self.w_range = np.array([w_min, w_max])

    def get_init_ws(self, filename='initial_weights.dat'):
        """
        Reads the weight file and writes it into the object.
        """
        print('\tWeight: Getting initial weights ...\n')

        self.init_ws = pd.read_csv(os.path.join(self.data_path, filename),
                                   sep='\t')

    def get_action_vecs(self, filename='actionID_dir.dat'):
        """
        Reads the action vector file and writes it into the object.
        """
        print('\tWeight: Getting action vectors from {} ...\n'.format(filename))

        self.action_vecs = pd.read_csv(os.path.join(self.data_path, filename),
                                       sep='\t')

        self.post_ids = self.action_vecs.id.values
        self.action_dirs = self.action_vecs.to_numpy()[:, 1:].T

    def get_firing_fields(self, filename='placeID_pos.dat'):
        """
        Reads the firing field file and writes it into the object.
        """
        
        print('\tWeight: Getting firing fields from {} ...\n'.format(filename))

        self.pre_fields = pd.read_csv(os.path.join(self.data_path, filename),
                                      sep='\t')

        self.pre_ids = self.pre_fields.id.values
        self.place_locs = self.pre_fields.to_numpy()[:, 1:]

        self.x_range = np.array([0, np.diff(np.unique(self.place_locs[:, 0]))[0]])
        self.y_range = np.array([0, np.diff(np.unique(self.place_locs[:, 1]))[0]])

    def vector_border(self, ax, cl = "k", lw_ = 2, ls = 'solid'):
        ax.vlines(self.place_locs[:, 0].min(),
                  self.place_locs[:, 1].min(),
                  self.place_locs[:, 1].max(),
                  color=cl,
                  lw=lw_,
                  linestyle=ls)
        ax.vlines(self.place_locs[:, 0].max(),
                  self.place_locs[:, 1].min(),
                  self.place_locs[:, 1].max(),
                  color=cl,
                  lw=lw_,
                  linestyle=ls)
        ax.hlines(self.place_locs[:, 1].min(),
                  self.place_locs[:, 0].min(),
                  self.place_locs[:, 0].max(),
                  color=cl,
                  lw=lw_,
                  linestyle=ls)
        ax.hlines(self.place_locs[:, 1].max(),
                  self.place_locs[:, 0].min(),
                  self.place_locs[:, 0].max(),
                  color=cl,
                  lw=lw_,
                  linestyle=ls)

    def plot_vector_field_placecells(self, fl_suffix='', frm='pdf'):
        """
        plots the vector feedforward weight of each PlACE cell and represents
        an arrow at the topological position of that cell

        Parameters
        ----------
        fl_suffix : str
            DESCRIPTION. The default is ''.
        frm : TYPE, format of the saved file
            DESCRIPTION. The default is 'pdf'.

        Returns
        -------
        None.

        """
        self.get_times_from_DA(dop_flname='dopamine_p-0.gdf')
        for time in self.times:
            flname = 'vectorfield-weight_p-{:.0f}ms-{}.{}'.format(time, fl_suffix, frm)
            print('\tWeight: Plotting vector field for time {}'.format(time) +
                  ' and saving it as {} ...\n'.format(flname))
            fig, ax = plt.subplots()
            vector_field = self.vector_field(time)
            ax.quiver(self.place_locs[:, 0],
                      self.place_locs[:, 1],
                      vector_field[0, :],
                      vector_field[1, :],
                      scale_units='xy',
                      angles='xy',
                      scale=.2)
#            ax.set_xlim((self.place_locs[:, 0].min() - .5,
#                         self.place_locs[:, 0].max() + .5))
#            ax.set_ylim((self.place_locs[:, 1].min() - .5,
#                         self.place_locs[:, 1].max() + .5))
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('t = {} ms'.format(time))
            ax.set_aspect(1)
            # rewrite the environment limits to start from 0
            plt.xticks(ticks=self.h_limit, labels=self.h_label)
            plt.yticks(ticks=self.v_limit, labels=self.v_label)
            
            border_flag = True
            if border_flag:
                self.vector_border(ax)
            fig.savefig(os.path.join(self.fig_path, flname), format=frm)
            plt.close(fig)

    def plot_vector_field_gridcells(self, fl_suffix='', frm='pdf'):
        """
        plots the vector feedforward weight of each GRID cell and represents
        an arrow at the topological position of that cell

        Parameters
        ----------
        fl_suffix : str
            DESCRIPTION. The default is ''.
        frm : TYPE, format of the saved file
            DESCRIPTION. The default is 'pdf'.

        Returns
        -------
        None.

        """
        self.get_times_from_DA(dop_flname='dopamine_g-0.gdf')
        for time in self.times:
            flname = 'vectorfield-weight_g-{:.0f}ms-{}.{}'.format(time, fl_suffix, frm)
            print('\tWeight: Plotting vector field for time {}'.format(time) +
                  ' and saving it as {} ...\n'.format(flname))
            fig, ax = plt.subplots()
            vector_field = self.vector_field(time)
            ax.quiver(self.place_locs[:, 0],
                      self.place_locs[:, 1],
                      vector_field[0, :],
                      vector_field[1, :],
                      scale_units='xy',
                      angles='xy',
                      scale=.2)
#            ax.set_xlim((self.place_locs[:, 0].min() - .5,
#                         self.place_locs[:, 0].max() + .5))
#            ax.set_ylim((self.place_locs[:, 1].min() - .5,
#                         self.place_locs[:, 1].max() + .5))

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('t = {} ms'.format(time))
            ax.set_aspect(1)
            # rewrite the environment limits to start from 0
            plt.xticks(ticks=self.h_limit, labels=self.h_label)
            plt.yticks(ticks=self.v_limit, labels=self.v_label)
            
            border_flag = True
            if border_flag:
                self.vector_border(ax, ls='dotted')
            fig.savefig(os.path.join(self.fig_path, flname), format=frm)
            plt.close(fig)
            
    def plot_vector_field_bordercells(self, fl_suffix='', frm='pdf', show_obs=False):
        """
        plots the vector feedforward weight of each BORDER cell and represents
        an arrow at the topological position of that cell

        Parameters
        ----------
        fl_suffix : str
            DESCRIPTION. The default is ''.
        frm : TYPE, format of the saved file
            DESCRIPTION. The default is 'pdf'.

        Returns
        -------
        None.

        """
        # Requires a bit of a different set-up since weights are fixed and
        # there's no csv for all border cell spikes
        self.get_init_ws()
        self.get_action_vecs()
        self.get_firing_fields(filename='borderID_pos.dat')
        self.w_range = [0, 1] # Arbitrary normalzation values since the border cell connections are fixed and equal
        
        time = 0.
        flname = 'vectorfield-weight_b-{:.0f}ms-{}.{}'.format(time, fl_suffix, frm)
        print('\tWeight: Plotting vector field for time {}'.format(time) +
              ' and saving it as {} ...\n'.format(flname))
        fig, ax = plt.subplots()
        vector_field = self.vector_field(time)
        ax.quiver(self.place_locs[:, 0],
                  self.place_locs[:, 1],
                  vector_field[0, :],
                  vector_field[1, :],
                  scale_units='xy',
                  angles='xy',
                  scale=1.8)
#            ax.set_xlim((self.place_locs[:, 0].min() - .5,
#                         self.place_locs[:, 0].max() + .5))
#            ax.set_ylim((self.place_locs[:, 1].min() - .5,
#                         self.place_locs[:, 1].max() + .5))
        
        if show_obs:
            if self.sim_dict['environment']['obstacles']['flag']:
                self.get_firing_fields(filename='obstacleID_pos.dat')
                vector_field = self.vector_field(time)
                ax.quiver(self.place_locs[:, 0],
                self.place_locs[:, 1],
                vector_field[0, :],
                vector_field[1, :],
                scale_units='xy',
                angles='xy',
                scale=1.8)
                
            
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('t = {} ms'.format(time))
        ax.set_aspect(1)
        # rewrite the environment limits to start from 0
        plt.xticks(ticks=self.h_limit, labels=self.h_label)
        plt.yticks(ticks=self.v_limit, labels=self.v_label)
        
        border_flag = True
        if border_flag:
            self.vector_border(ax)
        fig.savefig(os.path.join(self.fig_path, flname), format=frm)
        plt.close(fig)

    def vector_grid(self, vector_field, cell_num, rep_type):
        """
        calculates the vectorfield of one grid cell in the environment

        Parameters
        ----------
        vector_field : TYPE
            DESCRIPTION.
        cell_num : 
            number of the cell
        rep_type : float or int
            DESCRIPTION place=0, grid=1, border=2

        Returns
        -------
        None.

        """
        for j in [2, 3]:
            for k in range(self.temp_vec_field.shape[1]):
                self.temp_vec_field[j, k] = vector_field[j - 2, cell_num]
        max_fr = self.place_locs[cell_num, 5]
        #print('maxfr', max_fr)
        if self.place_locs[cell_num, 4] == 0:  # place
            #print("place")
            for j in range(self.temp_vec_field.shape[1]):
                d_x = ((self.temp_vec_field[0, j] - self.place_locs[cell_num, 0]) / self.place_locs[cell_num, 2]) ** 2
                d_y = ((self.temp_vec_field[1, j] - self.place_locs[cell_num, 1]) / self.place_locs[cell_num, 3]) ** 2
                self.temp_vec_field[2, j] = max_fr * self.temp_vec_field[2, j] * np.exp(-(d_x + d_y) / 2)
                self.temp_vec_field[3, j] = max_fr * self.temp_vec_field[3, j] * np.exp(-(d_x + d_y) / 2)
        elif self.place_locs[cell_num, 4] == 1:  # grid
            i_ = np.arange(0, 3)
            for j in range(self.temp_vec_field.shape[1]):
                omega = 2 * np.pi / (np.sin(np.pi / 3) * self.place_locs[cell_num, 2])
                phil = -np.pi / 6 + i_ * np.pi / 3
                kl = np.array([np.cos(phil), np.sin(phil)]).T
                S = max_fr * np.exp(self.place_locs[cell_num, 3] * (np.sum(
                    np.cos(omega * np.dot(kl, self.temp_vec_field[0:2, j] - self.place_locs[cell_num, 0:2]))) - 3) / 3)
                self.temp_vec_field[2, j] = self.temp_vec_field[2, j] * S
                self.temp_vec_field[3, j] = self.temp_vec_field[3, j] * S
        elif self.place_locs[cell_num, 4] == 2:  # border
            for j in range(self.temp_vec_field.shape[1]):
                if self.place_locs[cell_num, 0] - self.place_locs[cell_num, 2] <= self.temp_vec_field[0, j] and \
                        self.temp_vec_field[0, j] <= self.place_locs[cell_num, 0] + self.place_locs[cell_num, 2]:
                    if self.place_locs[cell_num, 1] - self.place_locs[cell_num, 3] <= self.temp_vec_field[1, j] and \
                            self.temp_vec_field[1, j] <= self.place_locs[cell_num, 1] + self.place_locs[cell_num, 3]:
                        self.temp_vec_field[2, j] = self.temp_vec_field[2, j] * max_fr
                        self.temp_vec_field[3, j] = self.temp_vec_field[3, j] * max_fr
                else:
                    self.temp_vec_field[2, j] = 0
                    self.temp_vec_field[3, j] = 0
        else:
            print("Representation unknown! Check the create_grid_positions.py")

    def plot_vector_field_stack(self, fl_suffix='', frm='pdf', plot_ind=True, cell_type=None):
        '''
        sums up the individual vector fields

        Parameters
        ----------
        fl_suffix : TYPE, optional
            DESCRIPTION. The default is ''.
        frm : TYPE, optional
            DESCRIPTION. The default is 'pdf'.
        plot_ind : bool
            DESCRIPTION. indicates if individual vectorfields would be saved

        Returns
        -------
        None.

        '''
        # TODO: These dopamine files are all the same (except for dopamine cell ID)- why save them multiple times?
        if cell_type == 'place':
            self.get_times_from_DA(dop_flname='dopamine_p-0.gdf')
        elif cell_type == 'grid':
            self.get_times_from_DA(dop_flname='dopamine_g-0.gdf')
        elif cell_type == 'border':
            self.get_times_from_DA(dop_flname='dopamine_b-0.gdf')
            self.get_init_ws()
            self.get_action_vecs()
            self.get_firing_fields(filename='borderID_pos.dat')
            self.w_range = [0, 1]
            self.times = [0]
        else:
            raise ValueError('cell_type must be provided to plot_vector_field_stack')
        #self.get_times_from_DA(dop_flname='dopamine-0.gdf')
        self.x_vect_pos = np.linspace(-1.2, 1.2, 16)
        self.y_vect_pos = np.linspace(-1.2, 1.2, 16)
        self.temp_vec_field = np.zeros([4, self.x_vect_pos.shape[0] * self.y_vect_pos.shape[0]])
        n = 0
        for j in range(self.x_vect_pos.shape[0]):
            for k in range(self.y_vect_pos.shape[0]):
                self.temp_vec_field[0, n] = self.x_vect_pos[j]
                self.temp_vec_field[1, n] = self.y_vect_pos[k]
                n += 1

        for time in self.times:
            self.stack_vec_field = np.zeros([2, self.x_vect_pos.shape[0] * self.y_vect_pos.shape[0]])
            path = os.path.join(self.fig_path, "VectorField_" + str(int(time)) + "ms")
            os.system("mkdir " + path)
            fig, ax = plt.subplots()
            vector_field = self.vector_field(time)
            for i in range(len(vector_field[0, :])):
                flname = '{}_vectorfield-weight-{:.0f}ms-{}{}.{}'.format(cell_type, time, fl_suffix, str(i), frm)
                self.vector_grid(vector_field, i, self.place_locs[i, 4])
                if self.place_locs[i, 4] != 2:
                    self.stack_vec_field = self.stack_vec_field + self.temp_vec_field[2:4, :]
                if plot_ind:
                    print('\tWeight: Plotting vector field for time {}'.format(time) +
                          ' and saving it as {} ...\n'.format(flname))
                    clr = ['b', 'orange', 'g']
                    scl = [10, 5, 2000]
                    fig, ax = plt.subplots()
                    ax.quiver(self.temp_vec_field[0, :],
                              self.temp_vec_field[1, :],
                              self.temp_vec_field[2, :],
                              self.temp_vec_field[3, :],
                              scale_units='xy',
                              angles='xy',
                              color=clr[int(self.place_locs[i, 4])],
                              scale=scl[int(self.place_locs[i, 4])])
                    if self.sim_dict['goal']['hide_goal'] == False:
                        add_goal_zone(ax, self.sim_dict)
                    ax.set_xlim((self.place_locs[:, 0].min() - .5,
                                 self.place_locs[:, 0].max() + .5))
                    ax.set_ylim((self.place_locs[:, 1].min() - .5,
                                 self.place_locs[:, 1].max() + .5))
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                    ax.set_title('t = {} ms, scale: 1:{}'.format(time, scl[int(self.place_locs[i, 4])]))
                    ax.set_aspect(1)
                    # rewrite the environment limits to start from 0
                    plt.xticks(ticks=self.h_limit, labels=self.h_label)
                    plt.yticks(ticks=self.v_limit, labels=self.v_label)
                    fig.savefig(os.path.join(path, flname), format=frm)
                    plt.close(fig)
            flname = 'average-{}-vectorfield-'.format(cell_type) + 'weight-{:.0f}ms-{}.{}'.format(time, fl_suffix, frm)
            print('\tWeight: Plotting {} stack-vector field for time {}'.format(cell_type, time) +
                  ' and saving it as {} ...\n'.format(flname))
            fig, ax = plt.subplots()
            ax.quiver(self.temp_vec_field[0, :], self.temp_vec_field[1, :],
                      self.stack_vec_field[0, :] / n, self.stack_vec_field[1, :] / n,
                      scale_units='xy', angles='xy', scale=.5)
            if self.sim_dict['goal']['hide_goal'] == False:
                add_goal_zone(ax, self.sim_dict)
            ax.set_xlim((self.place_locs[:, 0].min() - .5,
                         self.place_locs[:, 0].max() + .5))
            ax.set_ylim((self.place_locs[:, 1].min() - .5,
                         self.place_locs[:, 1].max() + .5))
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('average-vectorfield-weight-{:.0f}ms-{} scale:2:1'.format(time, fl_suffix))
            ax.set_aspect(1)
            # rewrite the environment limits to start from 0
            plt.xticks(ticks=self.h_limit, labels=self.h_label)
            plt.yticks(ticks=self.v_limit, labels=self.v_label)
            fig.savefig(os.path.join(self.fig_path, flname), format=frm)
            plt.close(fig)

    def get_times_from_DA(self, dop_flname, plot_selective=True, max_times=4):
        """
        geting time points, where the vector fields change, due to the STDP learning
        """
        print('\tWeight: Getting time points for vector fields' +
              ' from dopamine spikes stored in {} ...\n'.format(dop_flname))
        try:
            self.dopamine = SpikefileNest(self.data_path, dop_flname)
            self.dopamine.read_spike_files()
            dop_spks = self.dopamine.spk_data.get_group(0).time.values
            ids = np.hstack((np.where(np.abs(np.diff(dop_spks) - 0.1) > 1e-2)[0], -1))
            self.times = np.hstack((0, dop_spks[ids],
                                    dop_spks[ids] + 10))
        except ValueError:
            print("Reward was not found in this session!")

        max_num_trs = self.sim_dict['max_num_trs']
        self.times = self.times[0:min(len(self.times), max_num_trs)]
        if plot_selective:
            if len(self.times) > max_times:
                dummy1 = self.times[0:int((max_num_trs - max_times) / 2)]
                dummy2 = self.times[int((max_num_trs + max_times) / 2):max_num_trs]
                self.times = np.concatenate((dummy1, dummy2), axis=None)
                print(self.times)

    def vector_field(self, time):
        """
        calculates the vectorfield at the given time

        Parameters
        ----------
        time : float
            time of the simulation, where the vectorfield is demanded

        Returns
        -------
        weight_field : np.array
            vectorfield

        """
        print('\tWeight: Computing weight matrix for vector field ...')
        if time == 0:
            W = self.get_init_weight_matrix(self.init_ws)
        else:
            if not hasattr(self, 'init_w_test'):
                self.get_init_weight_matrix(self.init_ws)
            W = self.get_weight_matrix(time)
        W = Weight.normalize(W, self.w_range)
        weight_field = self.action_dirs.dot(W)
        weight_field = Weight.normalize_vec(weight_field,
                                            self.x_range,
                                            self.y_range)

#        for cell_num, cell in enumerate(self.place_locs):
#            if cell[4] == 2:
#                weight_field[0][cell_num] = 0
#                weight_field[1][cell_num] = 0
        return weight_field

    def get_update_times(self):
        self.plot_vector_field_placecells()

    def get_weight_matrix(self, T):
        print('\tWeight: Computing weight matrix for t={}ms\n'.format(T))
        data_grp = self.data[self.data.time < T].groupby(by=['pre', 'post'])
        grp_keys = data_grp.groups.keys()

        W = copy.deepcopy(self.init_w_mat)

        for id_pr, pre in enumerate(self.pre_ids):
            for id_po, post in enumerate(self.post_ids):
                if (pre, post) in grp_keys:
                    W[id_po, id_pr] = data_grp.get_group((pre, post)).weight.values[-1]

        return W

    def get_init_weight_matrix(self, W):
        print('\tWeight: Computing weight matrix for t=0 ms\n')

        self.init_w_mat = np.zeros(shape=(self.post_ids.size, self.pre_ids.size))

        for id_pr, pre in enumerate(self.pre_ids):
            for id_po, post in enumerate(self.post_ids):
                try:
                    self.init_w_mat[id_po, id_pr] = W.weight.values[
                        (W['post'] == post) & (W['pre'] == pre)][0]
                except IndexError:
                    print(f'!!!!!!!!!!!!!!!! ValueError: post:{post} pre:{pre}')
                    exit
        return self.init_w_mat

    @staticmethod
    def normalize(var, var_range):
        min_ = var_range[0]
        max_ = var_range[1]
        return (var - min_) / (max_ - min_)

    @staticmethod
    def normalize_vec(var, rangex, rangey):
        var = var / var.max()
        min_ = rangex[0]
        max_ = rangex[1]
        var[0, :] = var[0, :] * (max_ - min_)
        min_ = rangey[0]
        max_ = rangey[1]
        var[1, :] = var[1, :] * (max_ - min_)
        return var
