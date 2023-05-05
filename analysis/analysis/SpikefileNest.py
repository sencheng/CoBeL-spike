#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class SpikefileNest():
    """
    Opening .gdf file and managing the data.

    Parameters
    ----------
    data_path: String
        The location of the spike file.
    filename: String
        The name of the spike file.
    fig_path: any or None, default None
    -----------
    fig_path
    bin_width
    ms_scale
    neuron_ids
    spk_data
    max_time
    time_vec     [calc_avg_fr]
    fr_vec       [calc_avg_fr]
    fr_vec_smth  [smooth_avg_fr]
    """

    def __init__(self, data_path, filename, fig_path=None):

        print("Starting processing spiking data from {} ...\n".format(filename))

        self.file_path = os.path.join(data_path, filename)
        if not os.path.exists(self.file_path):
            self._file_path_does_not_exist_error()

        self.fig_path = fig_path
        self.bin_width = 50.0
        self.ms_scale = 1000.0

        if fig_path is not None:
            self._create_fig_path_if_not_exists()

        self.read_spike_files()

    def _file_path_does_not_exist_error(self):
        print(
            'Could not find {}. Check whether the data file exists and whether the path to the file is correct!'.format(
                self.file_path))  # TODO: Actually throw an error

    def _create_fig_path_if_not_exists(self):
        if not os.path.exists(self.fig_path):
            os.makedirs(self.fig_path)

    # TODO: Make privatespk
    def read_spike_files(self):
        print('\tspikefile_nest: Reading file: {}'.format(self.file_path))

        spk_data = pd.read_csv(self.file_path, sep='\t', names=['id', 'time'], index_col=False)
        spk_data.id = spk_data.id - spk_data.id.min()
        grp = spk_data.groupby(by='id')

        self.neuron_ids = np.sort(spk_data.id.unique())
        self.spk_data = grp
        if len(spk_data.time.values) == 0:
            raise ValueError
        self.max_time = spk_data.time.values.max()
            
        

    def calc_avg_fr(self, bin_size=1):
        """
        Calculating the average firing rate.

        Parameters
        ----------
        bin_size: int
        """

        print('\tspikefile_nest: Calculating average firing rate, bin={}ms ...\n'.format(bin_size))

        self.time_vec = np.arange(start=0., stop=self.max_time, step=bin_size)
        self.hist_edges = (self.time_vec[:-1] + self.time_vec[1:]) / 2
        self.fr_vec = np.zeros(shape=(self.neuron_ids.max() + self.neuron_ids.min() + 1, self.time_vec.size - 1))
        print(self.neuron_ids)

        for k_ in self.neuron_ids:
            tmp_spks = self.spk_data.get_group(k_).time.values
            self.fr_vec[k_, :] = np.histogram(tmp_spks, bins=self.time_vec)[0]
        self.fr_vec = self.fr_vec * 1000 / bin_size

        self.smooth_avg_fr()

    # TODO: Make private
    def smooth_avg_fr(self, win_size=20):
        """
        Smooths the average firing rate.

        Parameters
        ----------
        win_size: int
        """
        # TODO: Doesn't this require calc_avg_fr?
        print('\tspikefile_nest: Smoothing average firing rate,' + ' window={}ms ...\n'.format(win_size))

        window = np.ones(win_size)
        window = window / window.size

        self.fr_vec_smth = np.zeros_like(self.fr_vec)

        for k_ in self.neuron_ids:
            self.fr_vec_smth[k_, :] = np.convolve(self.fr_vec[k_, :], window, mode='same')

    def calc_isi(self, stop_isi=100., bin_size=1.):
        """
        Calculate the ISI distribution.

        Parameters
        ----------
        bin_size: float
        """
        print('\tspikefile_nest: Calculating ISI distribution,', ' bin={}  & max(ISI)>={}'.format(bin_size, stop_isi))

        hist_edges = np.arange(start=0., stop=stop_isi, step=bin_size)
        self.hist_edges_isi = (hist_edges[:-1] + hist_edges[1:]) / 2
        self.hist_vals_isi = np.zeros(
            shape=((self.neuron_ids.max() + self.neuron_ids.min() + 1), self.hist_edges_isi.size))

        self.cv = np.zeros(shape=(self.neuron_ids.max() + self.neuron_ids.min() + 1))

        for k_ in self.neuron_ids:
            tmp_spks = self.spk_data.get_group(k_).time.values
            print(k_, tmp_spks.sort())
            diff = np.diff(np.sort(tmp_spks))
            self.cv[k_] = stats.variation(diff)
            self.hist_vals_isi[k_, :] = np.histogram(diff, bins=hist_edges, density=True)[0]


    def plot_avg_fr(self, fig_filename, tr_times, frm='pdf'):
        """
        Calculate the average firing rate.
        Requires calc_avg_fr() to have run.
    
        Parameters
        ----------
        fig_filename: str
        tr_times: list containing dictionaries in format {trial, start_time, end_time}
        frm: str
        """
        print('\tspikefile_nest: Plotting average firing rate' + ' and saving it as {}.{}'.format(fig_filename, frm))

        if not hasattr(self, 'fr_vec'):
            self.calc_avg_fr()
        figfile_path = os.path.join(self.fig_path, fig_filename)
        os.makedirs(figfile_path, exist_ok=True)
        fig, ax = plt.subplots()
        neuron_ids = np.arange(self.neuron_ids.min(), self.neuron_ids.max() + 1)
        im = ax.pcolor(self.hist_edges, neuron_ids, self.fr_vec_smth, edgecolors='none')
        ax.set_ylabel('Neuron IDs')
        ax.set_xlabel('Time (ms)')
        plt.colorbar(im)
        for tr_time in tr_times:
            ax.set_xlim([tr_time["start_time"], tr_time["end_time"]])
            ax.set_xticks([tr_time["start_time"], tr_time["start_time"] + 0.5 *(tr_time["end_time"]-tr_time["start_time"]), tr_time["end_time"]])
            ax.set_title(f"{fig_filename} plot for trial {tr_time['trial']}")
            fig.savefig(figfile_path + f"/{fig_filename}_colorplot_tr_" + str(tr_time["trial"]) +'.{}'.format(frm), format=frm)
        ax.set_xlim([tr_times[0]["start_time"], tr_times[-1]["end_time"]])
        ax.set_xticks([tr_times[0]["start_time"], tr_times[0]["start_time"] + 0.5 *(tr_times[-1]["end_time"]-tr_times[0]["start_time"]), tr_times[-1]["end_time"]])
        ax.set_title(f"{fig_filename} plot for all trials")
        fig.savefig(figfile_path + f"/{fig_filename}_colorplot" +'.{}'.format(frm), format=frm)
        plt.close(fig)


    def plot_isi(self, fig_filename, frm='pdf'):
        """
        Plot the ISI distribution.
    
        Parameters
        ----------
        fig_filename: str
        frm: str
        """
        print('\tspikefile_nest: Plotting ISI distribution' + ' and saving it as {}.{}'.format(fig_filename, frm))

        figfile_path = os.path.join(self.fig_path, fig_filename)

        fig, ax = plt.subplots()
        ax.plot(self.hist_edges_isi, self.hist_vals_isi.T)
        ax.set_ylabel('Probability')
        ax.set_xlabel('Interspike interval (ms)')
        fig.savefig(figfile_path + '.{}'.format(frm), format=frm)

        fig, ax = plt.subplots()
        neuron_ids = np.arange(self.neuron_ids.min(), self.neuron_ids.max() + 1)
        ax.pcolormesh(self.hist_edges_isi, neuron_ids, self.hist_vals_isi)
        ax.set_ylabel('Probability')
        ax.set_xlabel('Interspike interval (ms)')
        fig.savefig(figfile_path + '_colorplot.{}'.format(frm), format=frm)
