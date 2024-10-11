#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


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
        self.data_path = data_path

        self.file_path = os.path.join(data_path, filename)
        if not os.path.exists(self.file_path):
            self._file_path_does_not_exist_error()

        self.fig_path = fig_path
        self.bin_width = 50.0
        self.ms_scale = 1000.0

        if fig_path is not None:
            self._create_fig_path_if_not_exists()

        self._read_spike_files()



    def _file_path_does_not_exist_error(self):
        print(
            'Could not find {}. Check whether the data file exists and whether '
            'the path to the file is correct!'.format(self.file_path)
        )
        raise FileNotFoundError


    def _create_fig_path_if_not_exists(self):
        if not os.path.exists(self.fig_path):
            os.makedirs(self.fig_path)


    def _read_spike_files(self):
        print('\tspikefile_nest: Reading file: {}'.format(self.file_path))

        spk_data = pd.read_csv(self.file_path, sep='\t', names=['id', 'time'], index_col=False)
        spk_data.id = spk_data.id - spk_data.id.min()
        grp = spk_data.groupby(by='id')

        self.neuron_ids = np.sort(spk_data.id.unique())
        self.spk_data = grp
        if len(spk_data.time.values) == 0:
            raise ValueError
        self.max_time = spk_data.time.values.max()


    def check_pop_size(self, filename):
        """
        Checks the given spike file against a data file containing the entire
        cell population. This is necessary for plotting populations where
        not every cell fires.

        Parameters
        ----------
        filename: string
            The filename of the file which contains the full list of cells in
            the population.

        TODO: This hasn't been extensively tested and may lead to some 
        irregularities in spike plots.
        """
        file_path = os.path.join(self.data_path, filename)
        pop_data = pd.read_csv(file_path, sep='\t', index_col=False, header=0)
        pop_data.id = pop_data.id - pop_data.id.min()
        self.neuron_ids = np.array(pop_data.id)


    def calc_avg_fr(self, bin_size=1):
        """
        Calculating the average firing rate.

        Parameters
        ----------
        bin_size: int
            The size of the bin for calculating the firing rate in milliseconds.
        """
        print('\tspikefile_nest: Calculating average firing rate, bin={}ms ...\n'.format(bin_size))

        self.time_vec = np.arange(start=0., stop=self.max_time, step=bin_size)
        self.hist_edges = (self.time_vec[:-1] + self.time_vec[1:]) / 2
        self.fr_vec = np.zeros(shape=(self.neuron_ids.max() + self.neuron_ids.min() + 1, self.time_vec.size - 1))

        for k_ in self.neuron_ids:
            try:
                tmp_spks = self.spk_data.get_group(k_).time.values
                self.fr_vec[k_, :] = np.histogram(tmp_spks, bins=self.time_vec)[0]
            except KeyError:
                continue

        self.fr_vec = self.fr_vec * 1000 / bin_size
        self._smooth_avg_fr()


    def _smooth_avg_fr(self, win_size=20):
        """
        Smooths the average firing rate.

        Parameters
        ----------
        win_size : int
            The size of the smoothing window in milliseconds.
        """
        print('\tspikefile_nest: Smoothing average firing rate, window={}ms ...\n'.format(win_size))

        window = np.ones(win_size) / win_size
        self.fr_vec_smth = np.zeros_like(self.fr_vec)

        for k_ in self.neuron_ids:
            self.fr_vec_smth[k_, :] = np.convolve(self.fr_vec[k_, :], window, mode='same')


    def plot_avg_fr(self, fig_filename, tr_times, frm='pdf'):
        """
        Plot the average firing rate.
        Requires calc_avg_fr() to have been run.

        Parameters
        ----------
        fig_filename : str
            The filename to save the figure.
        tr_times : list
            A list of dictionaries in format {trial, start_time, end_time}.
        frm : str
            The format to save the figure.
        """
        print(f'\tspikefile_nest: Plotting average firing rate and saving it as {fig_filename}.{frm}')

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
            mid_time = tr_time["start_time"] + 0.5 * (tr_time["end_time"] - tr_time["start_time"])
            ax.set_xticks([tr_time["start_time"], mid_time, tr_time["end_time"]])
            ax.set_title(f"{fig_filename} plot for trial {tr_time['trial']}")
            fig.savefig(os.path.join(figfile_path, f"{fig_filename}_colorplot_tr_{tr_time['trial']}.{frm}"), format=frm)
        
        ax.set_xlim([tr_times[0]["start_time"], tr_times[-1]["end_time"]])
        mid_time = tr_times[0]["start_time"] + 0.5 * (tr_times[-1]["end_time"] - tr_times[0]["start_time"])
        ax.set_xticks([tr_times[0]["start_time"], mid_time, tr_times[-1]["end_time"]])
        ax.set_title(f"{fig_filename} plot for all trials")
        fig.savefig(os.path.join(figfile_path, f"{fig_filename}_colorplot.{frm}"), format=frm)
        
        plt.close(fig)


    #TODO: not used but seem to work
    def calc_isi(self, stop_isi=100.0, bin_size=1.0):
        """
        Calculate the ISI distribution.

        Parameters
        ----------
        stop_isi : float, optional
            The upper limit for the ISI histogram bins. Default is 100.0.
        bin_size : float, optional
            The width of the histogram bins. Default is 1.0.
        """
        print(f'\tspikefile_nest: Calculating ISI distribution, bin={bin_size} & max(ISI)>={stop_isi}')

        hist_edges = np.arange(start=0.0, stop=stop_isi, step=bin_size)
        self.hist_edges_isi = (hist_edges[:-1] + hist_edges[1:]) / 2
        self.hist_vals_isi = np.zeros(
            shape=(self.neuron_ids.max() + self.neuron_ids.min() + 1, self.hist_edges_isi.size)
        )

        self.cv = np.zeros(shape=(self.neuron_ids.max() + self.neuron_ids.min() + 1))

        for k_ in self.neuron_ids:
            tmp_spks = self.spk_data.get_group(k_).time.values
            print(k_, np.sort(tmp_spks))
            diff = np.diff(np.sort(tmp_spks))
            self.cv[k_] = stats.variation(diff)
            self.hist_vals_isi[k_, :] = np.histogram(diff, bins=hist_edges, density=True)[0]


    #TODO: not used but seem to work
    def plot_isi(self, fig_filename, frm='pdf'):
        """
        Plot the ISI distribution.
        
        Parameters
        ----------
        fig_filename : str
            The name of the file to save the plot.
        frm : str
            The format in which to save the figure (e.g., 'pdf', 'png').
        """
        print(f'\tspikefile_nest: Plotting ISI distribution and saving it as {fig_filename}.{frm}')

        figfile_path = os.path.join(self.fig_path, fig_filename)

        # Plotting ISI distribution as a line plot
        fig, ax = plt.subplots()
        ax.plot(self.hist_edges_isi, self.hist_vals_isi.T)
        ax.set_ylabel('Probability')
        ax.set_xlabel('Interspike interval (ms)')
        fig.savefig(f'{figfile_path}.{frm}', format=frm)

        # Plotting ISI distribution as a color plot
        fig, ax = plt.subplots()
        neuron_ids = np.arange(self.neuron_ids.min(), self.neuron_ids.max() + 1)
        ax.pcolormesh(self.hist_edges_isi, neuron_ids, self.hist_vals_isi)
        ax.set_ylabel('Probability')
        ax.set_xlabel('Interspike interval (ms)')
        fig.savefig(f'{figfile_path}_colorplot.{frm}', format=frm)
