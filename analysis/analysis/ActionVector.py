

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class ActionVector():

    def __init__(self, data_path, filename, fig_path):

        self.file_path = os.path.join(data_path, filename)
        if not os.path.exists(self.file_path):
            print(
                'Could not find {}. Check whether the data file exists' + ' and whether the path to the file is correct!'.format(
                    filename))

        if not os.path.exists(fig_path):
            os.makedirs(fig_path)

        self.fig_path = fig_path

    def read_file(self):
        tmp_dat = pd.read_csv(self.file_path, sep='\t', names=['id', 'val'])
        self.units = tmp_dat.id.unique()
        stp_size = 0.0001
        grp_id = tmp_dat.groupby(by='id')
        self.time_vec = np.arange(0, grp_id.get_group(self.units[0])['val'].size * stp_size, stp_size)
        self.action_mat = np.zeros(shape=(self.units.size, self.time_vec.size))
        for idx, u_id in enumerate(self.units):
            print(idx, grp_id.get_group(u_id)['val'].values.size)
            self.action_mat[idx][:grp_id.get_group(u_id)['val'].values.size] = grp_id.get_group(u_id)['val'].values

    def plot(self, fig_filename):
        """
        Plots times and neurons and stores it in the directory `self.fig_path`.
        X-axis: Time
        y-axis: Neuron
        z-axis (color): Firing rate (spk/s)

        Parameters
        ----------
        fig_filename: String
          The name of the file of the plot to be stored.
          The final path will be `self.fig_path`/`fig_filename`
        """
        if not hasattr(self, 'action_mat'):
            self.read_file()

        figfile_path = os.path.join(self.fig_path, fig_filename)
        fig, ax = plt.subplots()
        im = ax.pcolor(self.time_vec, self.units, self.action_mat)
        ax.set_xlabel('Time (sample)')
        ax.set_ylabel('Neuron ID')
        plt.colorbar(im)
        fig.savefig(figfile_path + '_colorplot' + '.png', format='png')
