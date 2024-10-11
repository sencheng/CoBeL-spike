"""
this file plots vector fields
"""
import os
import copy
import json
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from .SpikefileNest import SpikefileNest
from .PlotHelper import add_goal_zone, add_obstacles
from matplotlib.colors import ListedColormap, BoundaryNorm

class Weight():
    """
    Represents a weight file. This is used for visualizing the connections between
    spatially distributed cell populations (i.e. place/grid) and the action cell population.
    """

    def __init__(self, data_path, cell_type, fig_path=None, times=None, quiet=False):
        if times is None:
            times = [0]

        self.cell_type = cell_type
        self.filename = f'{cell_type}-0.csv'
        self.file_path = os.path.join(data_path, self.filename)
        self.data_path = data_path
        self.fig_path = fig_path
        self.times = times

        if not quiet:
            print(f"Starting connections weights data from {self.filename} ...\n")

        if not os.path.exists(self.file_path) and not quiet:
            print(f'Could not find {self.filename}. Check whether the data file exists and whether the path to the file is correct!')

        if fig_path is not None and not os.path.exists(fig_path):
            os.makedirs(fig_path)

        self.read_sim_data()
        self.get_trial_data()

        # Used to rewrite the environment limits to start from 0
        env_limits = self.env_dict['environment']['openfield']
        self.hide_goal = self.env_dict['environment']['goal']['hide_goal']
        self.h_limit = [env_limits['xmin_position'], env_limits['xmax_position']]
        self.v_limit = [env_limits['ymin_position'], env_limits['ymax_position']]
        self.h_label = [0, self.h_limit[1] - self.h_limit[0]]
        self.v_label = [0, self.v_limit[1] - self.v_limit[0]]


    def get_trial_data(self, loc_filename='locs_time.dat', tr_filename='trials_params.dat'):
        """
        Loads time-trial information so that accurate goal zones can be
        plotted for simulations that have multiple goal zones.

        Parameters
        ----------
        loc_filename : str, optional
            The filename containing location and time data, by default 'locs_time.dat'.
        tr_filename : str, optional
            The filename containing trial parameters, by default 'trials_params.dat'.
        """
        # Load time-trial information
        tr_times = pd.read_csv(os.path.join(self.data_path, loc_filename), sep='\t', header=0)
        tr_times.drop(['x', 'y'], axis=1, inplace=True)
        
        # Extract start and end times for each trial
        self.tr_start_times = tr_times.drop_duplicates(subset='trial', keep='first').set_index('trial')
        self.tr_end_times = tr_times.drop_duplicates(subset='trial', keep='last').set_index('trial')
        
        # Load trial parameters
        self.tr_params = pd.read_csv(os.path.join(self.data_path, tr_filename), sep='\t', header=0)


    def get_trial_from_t(self, spike_time):
        """
        Determines the trial number corresponding to a given spike time.

        Parameters
        ----------
        spike_time : float
            The spike time in milliseconds.

        Returns
        -------
        int
            The trial number corresponding to the spike time.
        """
        # Convert spike time from ms to s
        spike_time /= 1000

        trial = 0
        for start_time in self.tr_start_times.time:
            start_time += 0.01  # Adding a small offset to account for precision

            if start_time < spike_time:
                trial += 1
            else:
                return trial

        return trial


    def read_sim_data(
        self, 
        data_path='../simulator', 
        sim_file='parameter_sets/current_parameter/sim_params.json', 
        env_file='parameter_sets/current_parameter/env_params.json'
        ):
        """
        Reads the simulation parameters from JSON files.

        Parameters
        ----------
        data_path : str, optional
            The directory that contains the sim_params file. The default is '../simulator'.
        sim_file : str, optional
            Filename of the sim_params file. The default is 'parameter_sets/current_parameter/sim_params.json'.
        env_file : str, optional
            Filename of the environment parameter file. The default is 'parameter_sets/current_parameter/env_params.json'.

        Returns
        -------
        None
        """
        self.sim_file_path = os.path.join(data_path, sim_file)
        with open(self.sim_file_path, "r") as f:
            self.sim_dict = json.load(f)

        self.env_file_path = os.path.join(data_path, env_file)
        with open(self.env_file_path, "r") as f:
            self.env_dict = json.load(f)


    def read_files(self, cell_type=None, quiet=False):
        """
        Reads the weight, action vector, and place field files and 
        writes them into the object.
        
        Parameters
        ----------
        cell_type : str, optional
            The desired cell population to retrieve data for. If no cell_type is specified, 
            the cell type specified in the initialization is used. The default is None.
        quiet : bool, optional
            If True, suppresses print statements. The default is False.

        Returns
        -------
        None
        """
        
        # Border cells have constant weights throughout simulations
        if cell_type == 'border':
            self.w_range = [0, 1]
            self.times = [0]
            cell_type = 'border'
        else:
            if not quiet:
                print(f'\tWeight: Reading weight file: {self.file_path}\n')
            
            if cell_type is None:
                cell_type = self.cell_type

            data = pd.read_csv(self.file_path, sep='\t', names=['pre', 'post', 'time', 'weight', 'N'])
            self.data = data.drop('N', axis=1)
            
            w_min = data.weight.min()
            w_max = data.weight.max()
            self.w_range = np.array([w_min, w_max])

        self.get_init_ws(quiet=quiet)
        self.get_action_vecs(quiet=quiet)
        self.get_firing_fields(filename=f'{cell_type}ID_pos.dat', quiet=quiet)


    def get_init_ws(self, filename='initial_weights.dat', quiet=False):
        """
        Reads the initial weights file and writes it into the object.
        
        Parameters
        ----------
        filename : str, optional
            The name of the initial weights file. The default is 'initial_weights.dat'.
        quiet : bool, optional
            If True, suppresses print statements. The default is False.
        
        Returns
        -------
        None
        """
        if not quiet:
            print('\tWeight: Getting initial weights ...\n')

        self.init_ws = pd.read_csv(
            os.path.join(self.data_path, filename), sep='\t'
        )


    def get_action_vecs(self, filename='actionID_dir.dat', quiet=False):
        """
        Reads the action vector file and writes it into the object.
        
        Parameters
        ----------
        filename : str, optional
            The name of the action vector file. The default is 'actionID_dir.dat'.
        quiet : bool, optional
            If True, suppresses print statements. The default is False.
        
        Returns
        -------
        None
        """
        if not quiet:
            print(f'\tWeight: Getting action vectors from {filename} ...\n')

        file_path = os.path.join(self.data_path, filename)
        self.action_vecs = pd.read_csv(file_path, sep='\t')

        self.post_ids = self.action_vecs.id.values
        self.action_dirs = self.action_vecs.iloc[:, 1:].to_numpy().T


    def get_firing_fields(self, filename, quiet=False):
        """
        Reads the firing field file and writes it into the object.

        Parameters
        ----------
        filename : str
            The name of the file containing the firing field data.
        quiet : bool, optional
            If True, suppresses print statements. The default is False.
        
        Returns
        -------
        None
        """
        if not quiet:
            print(f'\tWeight: Getting firing fields from {filename} ...\n')

        file_path = os.path.join(self.data_path, filename)
        self.pre_fields = pd.read_csv(file_path, sep='\t')

        self.pre_ids = self.pre_fields.id.values
        self.place_locs = self.pre_fields.iloc[:, 1:].to_numpy()

        self.x_range = np.diff(np.unique(self.place_locs[:, 0]))
        self.x_range = self.x_range[0] if self.x_range.size != 0 else 0

        self.y_range = np.diff(np.unique(self.place_locs[:, 1]))
        self.y_range = self.y_range[0] if self.y_range.size != 0 else 0

        self.x_range = np.array([0, self.x_range])
        self.y_range = np.array([0, self.y_range])


    def vector_border(self, ax, cl="k", lw_=2, ls="solid"):
        """
        Draws a border around the currently plotted cell population.
        
        This is useful when plotting grid cells, since their 'borders' will
        be a smaller portion of the environment. When calling this function for
        other populations, the border drawn will likely overlap the environment
        boundaries.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The matplotlib Axes to draw a border on.
        cl : str, optional
            Line color option. The default is "k" (black).
        lw_ : int, optional
            Line width option. The default is 2.
        ls : str, optional
            Line style option. The default is 'solid'.
        
        Returns
        -------
        None
        """
        ax.vlines(
            x=self.place_locs[:, 0].min(),
            ymin=self.place_locs[:, 1].min(),
            ymax=self.place_locs[:, 1].max(),
            color=cl,
            lw=lw_,
            linestyle=ls,
        )
        ax.vlines(
            x=self.place_locs[:, 0].max(),
            ymin=self.place_locs[:, 1].min(),
            ymax=self.place_locs[:, 1].max(),
            color=cl,
            lw=lw_,
            linestyle=ls,
        )
        ax.hlines(
            y=self.place_locs[:, 1].min(),
            xmin=self.place_locs[:, 0].min(),
            xmax=self.place_locs[:, 0].max(),
            color=cl,
            lw=lw_,
            linestyle=ls,
        )
        ax.hlines(
            y=self.place_locs[:, 1].max(),
            xmin=self.place_locs[:, 0].min(),
            xmax=self.place_locs[:, 0].max(),
            color=cl,
            lw=lw_,
            linestyle=ls,
        )


    def vector_grid(self, vector_field, cell_num, rep_type):
        """
        Calculates the vector field of one grid location in the environment.
        
        'Grid location' in this context indicates the grid of starting 
        positions for vectors being drawn in the environment, and is not 
        specifically associated with grid cells.

        Parameters
        ----------
        vector_field : np.array
            The vector field for the environment.
        cell_num : int
            The number of the cell.
        rep_type : float or int
            Type of representation: place=0, grid=1, border=2

        Returns
        -------
        None
        """
        for j in [2, 3]:
            for k in range(self.temp_vec_field.shape[1]):
                self.temp_vec_field[j, k] = vector_field[j - 2, cell_num]

        max_fr = self.place_locs[cell_num, 5]
        if self.place_locs[cell_num, 4] == 0:  # place
            for j in range(self.temp_vec_field.shape[1]):
                d_x = ((self.temp_vec_field[0, j] - self.place_locs[cell_num, 0]) / self.place_locs[cell_num, 2]) ** 2
                d_y = ((self.temp_vec_field[1, j] - self.place_locs[cell_num, 1]) / self.place_locs[cell_num, 3]) ** 2
                self.temp_vec_field[2, j] *= max_fr * np.exp(-(d_x + d_y) / 2)
                self.temp_vec_field[3, j] *= max_fr * np.exp(-(d_x + d_y) / 2)

        elif self.place_locs[cell_num, 4] == 1:  # grid
            i_ = np.arange(0, 3)
            for j in range(self.temp_vec_field.shape[1]):
                omega = 2 * np.pi / (np.sin(np.pi / 3) * self.place_locs[cell_num, 2])
                phil = -np.pi / 6 + i_ * np.pi / 3
                kl = np.array([np.cos(phil), np.sin(phil)]).T
                S = max_fr * np.exp(
                    self.place_locs[cell_num, 3] * (
                        np.sum(
                            np.cos(omega * np.dot(kl, self.temp_vec_field[0:2, j] - self.place_locs[cell_num, 0:2]))
                        ) - 3) / 3)
                self.temp_vec_field[2, j] *= S
                self.temp_vec_field[3, j] *= S

        elif self.place_locs[cell_num, 4] == 2:  # border
            for j in range(self.temp_vec_field.shape[1]):
                if (self.place_locs[cell_num, 0] - self.place_locs[cell_num, 2] <= self.temp_vec_field[0, j] <= 
                    self.place_locs[cell_num, 0] + self.place_locs[cell_num, 2]):
                    if (self.place_locs[cell_num, 1] - self.place_locs[cell_num, 3] <= self.temp_vec_field[1, j] <= 
                        self.place_locs[cell_num, 1] + self.place_locs[cell_num, 3]):
                        self.temp_vec_field[2, j] *= max_fr
                        self.temp_vec_field[3, j] *= max_fr
                    else:
                        self.temp_vec_field[2, j] = 0
                        self.temp_vec_field[3, j] = 0
                else:
                    self.temp_vec_field[2, j] = 0
                    self.temp_vec_field[3, j] = 0

        else:
            print("Representation unknown! Check the create_grid_positions.py")


    def set_cell_type(self, cell_type):
        """
        Sets the cell type and processes the appropriate data based on the cell type.

        Parameters
        ----------
        cell_type : str
            The type of cell, must be 'place', 'grid', or 'border'.
        
        Raises
        ------
        ValueError
            If an invalid cell type is provided.
        """
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
            raise ValueError('cell_type must be "place", "grid", or "border" to plot_vector_field_stack')

    
    def copy_failed_trial(self, fig, ax, current_tr, next_tr, failed_trial_path, frm):
        """
        Copies the visualization of failed trials between two consecutive trials.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure object to save.
        ax : matplotlib.axes.Axes
            The axes object where the title will be set.
        current_tr : int
            The current trial number.
        next_tr : int
            The next trial number.
        failed_trial_path : function
            A function that takes a trial number and returns a file path to save the figure.
        frm : str
            The format in which to save the figure.

        Returns
        -------
        None
        """
        if next_tr - current_tr < 2:
            return

        for tr in range(current_tr + 1, next_tr):
            ax.set_title(f"Trial: {tr}")

            fig.tight_layout()    
            fig.savefig(failed_trial_path(tr), format=frm, dpi=300)


    def plot_vector_field_stack(self, fl_suffix='', frm='pdf', plot_ind=True, cell_type=None, title=True):
        """
        Sums up the individual vector fields.

        Parameters
        ----------
        fl_suffix : str, optional
            Suffix to append to plot filenames. The default is ''.
        frm : str, optional
            Format for saved plots. The default is 'pdf'.
        plot_ind : bool
            Indicates if individual vector fields should be saved.
        cell_type : str
            Indicates which cell population to plot an averaged vector field for.
        title : bool
            Indicates whether to include a title in the plots.

        Returns
        -------
        None
        """
        self.set_cell_type(cell_type)
        self.calc_temp_vec_field()

        next_trs = [self.get_trial_from_t(time) for time in self.times]
        next_trs.append(-1)
        next_trs.pop(0)

        for time, next_tr in zip(self.times, next_trs):
            tr = self.get_trial_from_t(time)
            self.stack_vec_field = np.zeros([2, self.x_vect_pos.shape[0] * self.y_vect_pos.shape[0]])
            vector_field = self.vector_field(time)

            path = os.path.join(self.fig_path, f"VectorField_{tr}_{time}_ms")
            if not os.path.exists(path) and plot_ind:
                os.makedirs(path)

            fig, ax = plt.subplots()
            for i in range(len(vector_field[0, :])):
                flname = f'stack-{cell_type}_vectorfield-tr{tr}-{time}ms-{fl_suffix}{str(i)}.{frm}'
                self.vector_grid(vector_field, i, self.place_locs[i, 4])

                if self.place_locs[i, 4] != 2:
                    self.stack_vec_field += self.temp_vec_field[2:4, :]

                if plot_ind:
                    print(f'\tWeight: Plotting vector field for trial {tr} and saving it as {flname}\n')
                    clr = ['b', 'orange', 'g']
                    scl = 30
                    ax.quiver(
                        self.temp_vec_field[0, :],
                        self.temp_vec_field[1, :],
                        self.temp_vec_field[2, :],
                        self.temp_vec_field[3, :],
                        scale_units='xy',
                        angles='xy',
                        color=clr[int(self.place_locs[i, 4])],
                        scale=scl
                    )

                    self.format_plot(ax, f"Trial: {tr}", tr)
                    fig.tight_layout()
                    fig.savefig(os.path.join(path, flname), format=frm)

                    failed_trial_path = lambda current_tr: self._create_name(current_tr, cell_type, time, fl_suffix, i, frm)
                    self.copy_failed_trial(fig, ax, tr, next_tr, failed_trial_path, frm)

                    ax.cla()

            plt.close(fig)

            if cell_type == 'border':
                return  # Average border plot will always be empty, so there's not much point in making it

            flname = f'stack-{cell_type}-vectorfield-tr{tr}-{time}ms-{fl_suffix}.{frm}'
            print(f'\tWeight: Plotting {cell_type} stack-vector field for trial {tr} and saving it as {flname} ...\n')

            fig, ax = plt.subplots()
            ax.quiver(
                self.temp_vec_field[0, :],
                self.temp_vec_field[1, :],
                self.stack_vec_field[0, :] / self.n,
                self.stack_vec_field[1, :] / self.n,
                scale_units='xy',
                angles='xy',
                scale=1,
                color="b"
            )

            self.format_plot(ax, f"Trial: {tr}", tr)
            fig.tight_layout()
            fig.savefig(os.path.join(self.fig_path, flname), format=frm, dpi=300)

            failed_trial_path = lambda tr: os.path.join(self.fig_path, f'stack-{cell_type}-vectorfield-tr{tr}-{time}ms-{fl_suffix}.{frm}')
            self.copy_failed_trial(fig, ax, tr, next_tr, failed_trial_path, frm)

            plt.close(fig)


    def _create_name(self, tr, cell_type, time, fl_suffix, i, frm):
        """
        Creates a file path and name for saving vector field plots.

        Parameters
        ----------
        tr : int
            Trial number.
        cell_type : str
            Type of cell (e.g., place, grid, border).
        time : int or float
            Time in milliseconds.
        fl_suffix : str
            Suffix to append to the filename.
        i : int
            Index number for the file.
        frm : str
            File format (e.g., 'pdf', 'png').

        Returns
        -------
        str
            The complete file path for saving the vector field plot.
        """
        path = os.path.join(self.fig_path, f"VectorField_{tr}_{time}_ms")
        if not os.path.exists(path):
            os.makedirs(path)  # Use os.makedirs instead of os.system("mkdir ")

        return os.path.join(path, f'stack-{cell_type}_vectorfield-tr{tr}-{time}ms-{fl_suffix}{i}.{frm}')


    def format_plot(self, ax, title=None, tr=None):
        """
        Formats the plot with specific settings, such as title, goal zone, axis limits, and labels.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes object to format.
        title : str, optional
            The title for the plot. If None, no title is set. Default is None.
        tr : int, optional
            The trial number, used for adding the goal zone to the plot. Default is None.
        """
        if not self.hide_goal:
            add_goal_zone(ax, os.path.join(self.sim_dict['data_path'], 'trials_params.dat'), tr=tr)

        if title:
            ax.set_title(title)
            
        ax.set_xlim(self.h_limit[0] - 0.5, self.h_limit[1] + 0.5)
        ax.set_ylim(self.v_limit[0] - 0.5, self.v_limit[1] + 0.5)
        ax.set_xticks(ticks=self.h_limit, labels=self.h_label)
        ax.set_yticks(ticks=self.v_limit, labels=self.v_label)
        ax.set_aspect(1)


    def calc_temp_vec_field(self):
        """
        Initializes and calculates the temporary vector field.

        The vector field is initialized as a grid of points within the range of 
        -1.2 to 1.2 in both x and y directions, with a grid size of 16x16. 
        These points are stored in the temp_vec_field attribute for further calculations.
        """
        self.y_vect_pos = np.linspace(-1.2, 1.2, 16)
        self.x_vect_pos = np.linspace(-1.2, 1.2, 16)
        self.temp_vec_field = np.zeros([4, self.x_vect_pos.size * self.y_vect_pos.size])
        self.n = 0
        for j in range(self.x_vect_pos.size):
            for k in range(self.y_vect_pos.size):
                self.temp_vec_field[0, self.n] = self.x_vect_pos[j]
                self.temp_vec_field[1, self.n] = self.y_vect_pos[k]
                self.n += 1


    def calc_vector_field_stack_at_time(self, time):
        """
        Calculates the average vector field at a specific time without plotting.

        This method initializes the temporary vector field grid, calculates 
        the vector field at a given time, and accumulates the vectors into 
        a stack for averaging. It returns the temporary vector field, the 
        stacked vector field, and the number of vectors.

        Parameters
        ----------
        time : float
            The time at which to calculate the vector field.

        Returns
        -------
        temp_vec_field : numpy.ndarray
            The temporary vector field containing positions and vectors.
        stack_vec_field : numpy.ndarray
            The stacked vector field accumulated across grid points.
        n : int
            The number of vectors in the grid.
        """
        self.calc_temp_vec_field()
        self.stack_vec_field = np.zeros([2, self.x_vect_pos.size * self.y_vect_pos.size])
        vector_field = self.vector_field(time)

        for i in range(vector_field.shape[1]):
            self.vector_grid(vector_field, i, self.place_locs[i, 4])
            if self.place_locs[i, 4] != 2:
                self.stack_vec_field += self.temp_vec_field[2:4, :]

        return self.temp_vec_field, self.stack_vec_field, self.n


    def plot_vector_field_all(self, fl_suffix='', frm='pdf'):
        """
        Summarizes and plots the individual vector fields from all cell populations.

        This method calculates and sums vector fields from both place and grid 
        cell populations, then plots the resulting vector field for each time 
        point in `self.times`.

        Parameters
        ----------
        fl_suffix : str, optional
            Suffix to append to the plot filenames. The default is ''.
        frm : str, optional
            File format for saving the plots (e.g., 'pdf', 'png'). The default is 'pdf'.

        Returns
        -------
        None
        """
        self.get_times_from_DA(dop_flname='dopamine_p-0.gdf')
        self.calc_delta_w()

        for time in self.times:
            self.stack_vec_field = np.zeros([2, self.x_vect_pos.size * self.y_vect_pos.size])
            path = os.path.join(self.fig_path, f"VectorField_{int(time)}ms")
            if not os.path.exists(path):
                os.makedirs(path)
            
            # Process and accumulate vector fields for place cells
            self.read_files(cell_type='place', quiet=True)
            vector_field = self.vector_field(time)
            for i in range(vector_field.shape[1]):
                self.vector_grid(vector_field, i, self.place_locs[i, 4])
                if self.place_locs[i, 4] != 2:
                    self.stack_vec_field += self.temp_vec_field[2:4, :]
            
            # Process and accumulate vector fields for grid cells
            self.read_files(cell_type='grid', quiet=True)
            vector_field = self.vector_field(time)
            for i in range(vector_field.shape[1]):
                self.vector_grid(vector_field, i, self.place_locs[i, 4])
                if self.place_locs[i, 4] != 2:
                    self.stack_vec_field += self.temp_vec_field[2:4, :]
            
            # Generate the plot
            flname = f'complete-vectorfield-weight-{int(time)}ms-{fl_suffix}.{frm}'
            print(f'\tWeight: Plotting complete stack-vector field for time {time} and saving it as {flname}...\n')
            
            fig, ax = plt.subplots()
            ax.quiver(
                self.temp_vec_field[0, :],
                self.temp_vec_field[1, :],
                self.stack_vec_field[0, :] / self.n,
                self.stack_vec_field[1, :] / self.n,
                scale_units='xy',
                angles='xy',
                scale=1
            )
            
            self.format_plot(ax, f'complete-vectorfield-weight-{int(time)}ms-{fl_suffix}')
            fig.savefig(os.path.join(self.fig_path, flname), format=frm)
            plt.close(fig)


    def get_times_from_DA(self, dop_flname, plot_selective=False, max_times=4):
        """
        Extracts time points where vector fields change due to STDP learning, based on dopamine spikes.

        Parameters
        ----------
        dop_flname : str
            Filename of the dopamine spike data.
        plot_selective : bool, optional
            If True, limits the number of time points to `max_times`, selecting a subset for plotting. The default is False.
        max_times : int, optional
            Maximum number of time points to select if `plot_selective` is True. The default is 4.

        Returns
        -------
        None
        """
        print(f'\tWeight: Getting time points for vector fields from dopamine spikes stored in {dop_flname}\n')

        try:
            # Load dopamine spike data
            self.dopamine = SpikefileNest(self.data_path, dop_flname)
            dop_spks = self.dopamine.spk_data.get_group(0).time.values
            
            # Identify significant spike intervals
            ids = np.hstack((np.where(np.abs(np.diff(dop_spks) - 0.1) > 1e-2)[0], -1))
            max_num_trs = self.sim_dict['max_num_trs']
            
            # Calculate the relevant time points
            self.times = np.hstack((0, dop_spks[ids] + 10))
            self.times = self.times[:min(len(self.times), max_num_trs)]
        except ValueError:
            print("Reward was not found in this session!")

        if plot_selective:
            if len(self.times) > max_times:
                dummy1 = self.times[:int((max_num_trs - max_times) / 2)]
                dummy2 = self.times[int((max_num_trs + max_times) / 2):max_num_trs]
                self.times = np.concatenate((dummy1, dummy2), axis=None)


    def vector_field_batch(self, fl_suffix='', frm='pdf', cell_type='place', border_flag=False, show_obs=False, title=True):
        """
        Generates and plots vector fields for specified time points and cell types.

        Parameters
        ----------
        fl_suffix : str, optional
            Suffix to append to plot filenames. The default is ''.
        frm : str, optional
            Format for saved plots. The default is 'pdf'.
        cell_type : str, optional
            Indicates which cell population to plot (e.g., 'place', 'grid', 'border'). The default is 'place'.
        border_flag : bool, optional
            If True, adds a border to the plot. The default is False.
        show_obs : bool, optional
            If True, shows obstacles in the environment for border cells. The default is False.
        title : bool, optional
            If True, displays the title on the plot. The default is True.
        """
        # Set the cell type for the vector field
        self.set_cell_type(cell_type)

        # Calculate vector fields for all time points
        fields = [self.vector_field(time, normalize=False) for time in self.times]
        
        # Normalize vector fields and split into separate time points
        all_fields, colors = Weight.trim_outliers_zscore(np.hstack(fields))
        all_fields = Weight.normalize_vec(all_fields, self.x_range, self.y_range)
        split_fields = np.split(all_fields, len(self.times), axis=1)
        colors = np.split(colors, len(self.times), axis=1)

        # Set up colormap and normalization for vector colors
        cmap = ListedColormap(['black', 'red'])
        norm = BoundaryNorm([0, 2, 4], 2)  # Boundary values are arbitrary since each value is set explicitly

        # Create a plot for each time point
        fig, ax = plt.subplots()

        for i, field in enumerate(split_fields):
            tr = self.get_trial_from_t(self.times[i])
            flname = f'vectorfield-weight_{cell_type[0]}-{tr}-{self.times[i]}ms-{fl_suffix}.{frm}'
            print(f'\tWeight: Plotting vector field for time {tr} and saving it as {flname}\n')
            
            # Plot the vector field
            q = ax.quiver(
                self.place_locs[:, 0],
                self.place_locs[:, 1],
                field[0, :],
                field[1, :],
                colors[i],
                scale_units='xy',
                angles='xy',
                norm=norm,
                cmap=cmap
            )
            
            # Format the plot
            self.format_plot(ax, f'Trial: {tr}')
            plt.quiverkey(q, 1.2, 0.9, 0.2, 'real', color='k', labelpos='E')
            plt.quiverkey(q, 1.2, 0.8, 0.2, 'trimmed', color='r', labelpos='E')

            # Plot obstacles if required
            if cell_type == 'border' and show_obs and self.sim_dict['environment']['obstacles']['flag']:
                self.get_firing_fields(filename='obstacleID_pos.dat')
                vector_field = self.vector_field(self.times[i])  # Recalculate vector field for obstacles
                ax.quiver(
                    self.place_locs[:, 0],
                    self.place_locs[:, 1],
                    vector_field[0, :],
                    vector_field[1, :],
                    scale_units='xy',
                    angles='xy',
                    scale=1.8
                )
            
            # Add goal zones if applicable
            if not self.hide_goal:
                if tr <= self.sim_dict['max_num_trs']:
                    add_goal_zone(ax, os.path.join(self.sim_dict['data_path'], 'trials_params.dat'), tr=tr)
                else:
                    print(f'Warning: found dopamine spikes after {self.sim_dict["max_num_trs"]} trials concluded')
            
            # Add border if required
            if border_flag:
                self.vector_border(ax, ls='dotted')

            # Save the figure
            fig.savefig(os.path.join(self.fig_path, flname), format=frm)
            ax.cla()  # Clear the axes for the next plot
        
        plt.close(fig)

                
    def delta_w_helper(self, n_t=None, tr_list=None, pop='place', frm='pdf', plot_all=True, rotate_dir=None, calc_all_goals=False, save_all_vecs=True):
        """
        Repeatedly calls plot_delta_w() for multiple trial subdivisions, normalizing
        across all requested subdivisions for easy comparison.

        Parameters
        ----------
        n_t : int, optional
            Number of trials to include in each plot_delta_w call. Should be a positive integer.
        tr_list : list, optional
            List of start-end pairs to designate which trials to group in each plot_delta_w call.
        pop : str, optional
            Cell population to examine (e.g., 'place', 'grid'). Default is 'place'.
        frm : str, optional
            File format for saving plots. Default is 'pdf'.
        plot_all : bool, optional
            If True, plots an additional delta_w plot for all trials. Default is True.
        rotate_dir : tuple, optional
            If provided, plots will include additional vectors showing rotation towards the goal area.
        calc_all_goals : bool, optional
            If True, calculates a rotated vector field for each goal position for each trial pair. Requires rotate_dir.
        save_all_vecs : bool, optional
            If True, saves all vector fields. Default is True.
        """
        
        # Determine trial subdivisions
        param_list = []
        if n_t is not None and tr_list is None:
            r = self.sim_dict['max_num_trs'] % n_t
            q = self.sim_dict['max_num_trs'] // n_t
            param_list = [[1 + i * n_t, n_t * (i + 1)] for i in range(q)]
            if r != 0:
                param_list.append([q * n_t + 1, q * n_t + r])
            if plot_all:
                param_list.append([1, self.sim_dict['max_num_trs']])
            param_list = np.array(param_list)
        elif n_t is None and tr_list is not None:
            param_list = np.array(tr_list)
        else:
            raise ValueError('Delta_W plotter must receive exactly one trial selection parameter (n_t XOR tr_list)')

        # Initialize storage for field data and goal locations
        fields, times, goal_locs = [], [], []
        tr_starts, tr_ends, goal_xs, goal_ys, r_v_sum_list = [], [], [], [], []
        
        # Get unique goal locations
        all_goal_locs = self.tr_params.drop_duplicates(subset=['goal_x', 'goal_y'])[['trial_num', 'goal_x', 'goal_y']]

        # Calculate fields and times for each trial subdivision
        for pair in param_list:
            end_tr_dat = self.tr_params[self.tr_params['trial_num'] == pair[-1]]
            goal_locs.append((end_tr_dat['goal_x'].to_numpy()[0], end_tr_dat['goal_y'].to_numpy()[0]))
            field, time = self.calc_delta_w(pair[0], pair[1], pop=pop)
            fields.append(field)
            times.append(time)
        
        # Normalize and split fields for plotting
        all_fields, colors = Weight.trim_outliers_zscore(np.hstack(fields))
        all_fields = Weight.normalize_vec(all_fields, self.x_range, self.y_range)
        split_fields = np.split(all_fields, len(param_list), axis=1)
        colors = np.split(colors, len(param_list), axis=1)
        
        # Iterate over each field and plot
        for i, field in enumerate(split_fields):
            params = (param_list[i][0], param_list[i][1], times[i][0], times[i][1], colors[i])
            
            if rotate_dir is not None:
                if calc_all_goals:
                    for j, goal_dat in all_goal_locs.iterrows():
                        goal_x, goal_y = goal_dat['goal_x'], goal_dat['goal_y']
                        goal = (goal_x, goal_y)
                        rotated_field = self.rotate_vec_grid(field, goal, rotate_dir)
                        rotated_field_real = self.rotate_vec_grid(fields[i], goal, rotate_dir)
                        self.plot_delta_w(params, field, rotated_field=rotated_field, plot_tr_goal=goal_dat['trial_num'], pref=f'goal({goal_x},{goal_y})rotated-')
                        r_v_sum_list.append(np.sum(rotated_field_real, axis=1))
                        tr_starts.append(params[0])
                        tr_ends.append(params[1])
                        goal_xs.append(goal_x)
                        goal_ys.append(goal_y)
                        if save_all_vecs:
                            df = pd.DataFrame({
                                'gridpos_x': self.place_locs[:, 0],
                                'gridpos_y': self.place_locs[:, 1],
                                'vector_x': rotated_field_real[0, :],
                                'vector_y': rotated_field_real[1, :]
                            })
                            if params[0] == params[1]:
                                fl_name = f'delta_w_rotated_tr{params[0]}_goal({goal_x},{goal_y}).dat'
                            else:
                                fl_name = f'delta_w_rotated_tr{params[0]}-{params[1]}_goal({goal_x},{goal_y}).dat'
                            df.to_csv(os.path.join(self.fig_path, fl_name), sep='\t')
                else:
                    rotated_field = self.rotate_vec_grid(field, goal_locs[i], rotate_dir)
                    rotated_field_real = self.rotate_vec_grid(fields[i], goal_locs[i], rotate_dir)
                    self.plot_delta_w(params, field, rotated_field=rotated_field, pref='rotated-')
                    r_v_sum_list.append(np.sum(rotated_field_real, axis=1))
                    tr_starts.append(params[0])
                    tr_ends.append(params[1])
                    goal_xs.append(goal_locs[i][0])
                    goal_ys.append(goal_locs[i][1])
                    if save_all_vecs:
                        df = pd.DataFrame({
                            'gridpos_x': self.place_locs[:, 0],
                            'gridpos_y': self.place_locs[:, 1],
                            'vector_x': rotated_field_real[0, :],
                            'vector_y': rotated_field_real[1, :]
                        })
                        if params[0] == params[1]:
                            fl_name = f'delta_w_rotated_tr{params[0]}.dat'
                        else:
                            fl_name = f'delta_w_rotated_tr{params[0]}-{params[1]}.dat'
                        df.to_csv(os.path.join(self.fig_path, fl_name), sep='\t')
            else:
                self.plot_delta_w(params, field)
        
        # Save vector sums if rotation is applied
        if rotate_dir is not None:
            r_v_sum_list = np.array(r_v_sum_list)
            df = pd.DataFrame({
                'trial_start': tr_starts,
                'trial_end': tr_ends,
                'goal_x': goal_xs,
                'goal_y': goal_ys,
                'vector_sum_x': r_v_sum_list[:, 0],
                'vector_sum_y': r_v_sum_list[:, 1]
            })
            fl_name = 'rotated_vector_sums.dat'
            if calc_all_goals:
                fl_name = 'all_goals_' + fl_name
            df.to_csv(os.path.join(self.fig_path, fl_name), sep='\t')


    def rotate_vec_grid(self, delta_vecs, goal_loc, target_dir=(0, 1)):
        """
        Rotates the given vectors so that they align with a specified target direction
        relative to a goal location.

        Parameters
        ----------
        delta_vecs : np.array
            Array of 2D vectors to be rotated. Shape is (2, N), where N is the number of vectors.
        goal_loc : tuple
            The goal location (x, y) to which the vectors should be aligned.
        target_dir : tuple, optional
            The direction (dx, dy) that the vectors should be rotated towards. Default is (0, 1) (i.e., upwards).

        Returns
        -------
        np.array
            The rotated vectors as a 2D array with shape (2, N).
        """
        # Zip the x and y components of the vectors together
        zipped = list(zip(delta_vecs[0, :], delta_vecs[1, :]))
        rotated_vecs = np.empty_like(zipped)  # Create an array to hold the rotated vectors

        # Iterate over each vector and rotate it towards the target direction
        for i, vec in enumerate(zipped):
            # Calculate the angle from the vector to the goal
            angle_to_goal = np.arctan2(goal_loc[1] - self.place_locs[i, 1], goal_loc[0] - self.place_locs[i, 0])
            # Calculate the angle difference between the target direction and the goal direction
            angle = np.arctan2(target_dir[1], target_dir[0]) - angle_to_goal
            
            # Create the rotation matrix
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            # Apply the rotation to the vector
            rotated_vecs[i] = np.dot(rotation_matrix, vec)

        # Unzip the rotated vectors back into x and y components
        x, y = zip(*rotated_vecs)
        
        return np.array([list(x), list(y)])


    def plot_delta_w(self, params, field, plot_tr_goal=None, rotated_field=None, pop='place', frm='pdf', pref=''):
        """
        Plots a single delta_w plot, using params given by delta_w_helper.

        Parameters
        ----------
        params : tuple
            Contains start trial, end trial, start time, end time, and colors for vectors.
        field : np.array
            The vector field to be plotted.
        plot_tr_goal : int, optional
            The trial number for which the goal zone should be plotted. Defaults to the end trial.
        rotated_field : np.array, optional
            An optional field of rotated vectors to be plotted alongside the original vectors.
        pop : str, optional
            The population type (e.g., 'place', 'grid', 'border'). Defaults to 'place'.
        frm : str, optional
            The file format for saving the plot (e.g., 'pdf', 'png'). Defaults to 'pdf'.
        pref : str, optional
            A prefix to add to the filename. Defaults to ''.

        Returns
        -------
        rotated_vec_sum : np.array
            The sum of the rotated vectors if `rotated_field` is provided, otherwise 0.
        """
        start, end, start_t, end_t, colors = params
        if plot_tr_goal is None:
            plot_tr_goal = end
        rotated_vec_sum = 0 

        cmap = ListedColormap(['black', 'red'])
        norm = BoundaryNorm([0, 2, 4], 2)  # Boundary values are arbitrary since each value is set explicitly

        if start == end:
            flname = '{}{}_vectorfield-delta_tr{}.{}'.format(pref, pop, start, frm)
            print('\tWeight: Plotting vector field delta for trial {}'.format(start) +
                ' and saving it as {} ...\n'.format(flname))
            title = fr'$ \Delta w _{{tr~{start}}}: {start_t:.2f}s-{end_t:.2f}s$'
        else:
            flname = '{}{}_vectorfield-delta_tr{}-{}.{}'.format(pref, pop, start, end, frm)
            print('\tWeight: Plotting vector field delta for trials {} - {}'.format(start, end) +
                ' and saving it as {} ...\n'.format(flname))
            title = fr'$ \Delta w _{{[tr~{start}, tr~{end}]}}: {start_t:.2f}s-{end_t:.2f}s$'

        fig, ax = plt.subplots()

        # Plot the main vector field
        q = ax.quiver(self.place_locs[:, 0],
                    self.place_locs[:, 1],
                    field[0, :],
                    field[1, :],
                    colors,
                    scale_units='xy',
                    angles='xy',
                    scale=0.5,
                    norm=norm,
                    cmap=cmap)

        # Plot the rotated vector field if provided
        if rotated_field is not None:
            qr = ax.quiver(self.place_locs[:, 0],
                        self.place_locs[:, 1],
                        rotated_field[0, :],
                        rotated_field[1, :],
                        scale_units='xy',
                        angles='xy',
                        scale=0.5,
                        color='grey')
            plt.quiverkey(qr, 1.2, .7, .2, 'rotated', color='grey', labelpos='E')
            rotated_vec_sum = np.sum(rotated_field, axis=1)
        
        # Format the plot with title, labels, and other elements
        self.format_plot(ax, title)
        plt.quiverkey(q, 1.2, .9, .2, 'real', color='k', labelpos='E')
        plt.quiverkey(q, 1.2, .8, .2, 'trimmed', color='r', labelpos='E')

        # Add goal zone and obstacles if they exist
        if not self.hide_goal:
            add_goal_zone(ax, os.path.join(self.sim_dict['data_path'], 'trials_params.dat'), tr=plot_tr_goal)

        if self.env_dict['environment']['obstacles']['flag']:
            add_obstacles(ax, self.env_dict)

        # Save the figure
        fig.savefig(os.path.join(self.fig_path, flname), format=frm)
        plt.close(fig)

        return rotated_vec_sum


    def calc_delta_w(self, start=1, end=None, pop='place'):
        """
        Calculates the change in the vector field across trials.

        Parameters
        ----------
        start : int
            The starting trial number.
        end : int
            The ending trial number. If not provided, it defaults to the last trial.
        pop : str
            The cell population type (e.g., 'place', 'grid', 'border'). Defaults to 'place'.
        
        Returns
        -------
        delta_field : np.array
            The difference between the vector fields at the start and end of the specified trial range.
        (start_t, end_t) : tuple
            The start and end times corresponding to the start and end trials.
        """
        # Read the files related to the specified cell population type
        self.read_files(cell_type=pop)
        
        # Set the end trial to the last trial if it's not provided or exceeds the max number of trials
        if end is None or end > self.sim_dict['max_num_trs']:
            end = self.sim_dict['max_num_trs']
        
        # Ensure the start trial is at least 1
        if start < 1:
            start = 1

        # Get the start and end times for the specified trials
        start_t = self.tr_start_times.at[start, 'time']
        end_t = self.tr_end_times.at[end, 'time']
        
        # Calculate the vector field at the start and end times
        start_field = self.vector_field(start_t * 1000, normalize=False)
        end_field = self.vector_field(end_t * 1000, normalize=False)
        
        # Calculate the difference between the end and start vector fields
        delta_field = end_field - start_field
        
        # Return the delta field and the start and end times
        return delta_field, (start_t, end_t)


    def vector_field(self, time, normalize=True):
        """
        Calculates the vector field at the given time.

        Parameters
        ----------
        time : float
            The time in the simulation for which the vector field is required.
        normalize : bool, optional
            If True, the resulting vector field is normalized. Defaults to True.

        Returns
        -------
        weight_field : np.array
            The calculated vector field.
        """
        print('\tWeight: Computing weight matrix for vector field ...')

        # If time is 0, get the initial weight matrix
        if time == 0:
            W = self.get_init_weight_matrix(self.init_ws)
        else:
            # If the initial weight matrix is not already available, compute it
            if not hasattr(self, 'init_w_mat'):
                self.get_init_weight_matrix(self.init_ws)
            # Get the weight matrix at the given time
            W = self.get_weight_matrix(time)

        # Normalize the weight matrix based on the range of weights
        W = Weight.normalize(W, self.w_range)

        # Calculate the weight field (vector field) by dotting the action directions with the weight matrix
        weight_field = self.action_dirs.dot(W)

        # If normalization is requested, normalize the weight field
        if normalize:
            weight_field = Weight.normalize_vec(weight_field, self.x_range, self.y_range)

        return weight_field


    def get_weight_matrix(self, T):
        """
        Computes the weight matrix at a given time T.

        Parameters
        ----------
        T : float
            The time in milliseconds at which to compute the weight matrix.

        Returns
        -------
        W : np.array
            The weight matrix at time T.
        """
        # Inform the user that the weight matrix is being computed for the given time
        print('\tWeight: Computing weight matrix for t={}ms\n'.format(T))

        # Group data by 'pre' and 'post' neurons, considering only weights up to time T
        data_grp = self.data[self.data.time < T].groupby(by=['pre', 'post'])

        # Retrieve the unique (pre, post) neuron pairs
        grp_keys = data_grp.groups.keys()

        # Start with the initial weight matrix as a baseline
        W = copy.deepcopy(self.init_w_mat)

        # Loop over all pre-synaptic neurons
        for id_pr, pre in enumerate(self.pre_ids):
            # Loop over all post-synaptic neurons
            for id_po, post in enumerate(self.post_ids):
                # If there's an entry for the (pre, post) pair, update the weight matrix
                if (pre, post) in grp_keys:
                    # Get the last weight value before time T for this (pre, post) pair
                    W[id_po, id_pr] = data_grp.get_group((pre, post)).weight.values[-1]

        # Return the updated weight matrix
        return W


    def get_init_weight_matrix(self, W):
        """
        Initializes the weight matrix at t=0 ms using the initial weights provided.

        Parameters
        ----------
        W : DataFrame
            A DataFrame containing the initial weights with 'pre' and 'post' neuron identifiers.

        Returns
        -------
        self.init_w_mat : np.array
            The initialized weight matrix at t=0 ms.
        """
        # Inform the user that the initial weight matrix is being computed
        print('\tWeight: Computing weight matrix for t=0 ms\n')

        # Initialize the weight matrix with zeros, shaped according to the size of post and pre neuron populations
        self.init_w_mat = np.zeros(shape=(self.post_ids.size, self.pre_ids.size))

        # Loop over all pre-synaptic neurons
        for id_pr, pre in enumerate(self.pre_ids):
            # Loop over all post-synaptic neurons
            for id_po, post in enumerate(self.post_ids):
                try:
                    # Assign the weight value to the matrix for the corresponding (post, pre) neuron pair
                    self.init_w_mat[id_po, id_pr] = W.weight.values[
                        (W['post'] == post) & (W['pre'] == pre)][0]
                except IndexError:
                    # If there is an IndexError (no matching weight found), print an error message and exit
                    print(f'!!!!!!!!!!!!!!!! ValueError: post:{post} pre:{pre}')
                    exit  # Exit the program upon encountering the error

        # Return the initialized weight matrix
        return self.init_w_mat


    def get_data_smth(self, bins):
        """
        Smooths the data by binning the weights over specified time intervals and calculates the mean weight for each bin.

        Parameters
        ----------
        bins : array-like
            The time intervals used to bin the data. Each bin represents a time range over which weights will be averaged.
            
        Returns
        -------
        None
        """
        # Create a deep copy of the data to avoid modifying the original dataset
        self.data_smth = self.data.copy(deep=True)

        # Extend the bins array to ensure that the final bin covers all data points
        bins = np.append(bins, [2*bins[-1] - bins[-2]])
        bin_labels = bins[:-1]

        # Create a new column 'time_bin' in the data, assigning each time point to a bin
        self.data_smth['time_bin'] = pd.cut(self.data_smth['time'], bins=bins, labels=bin_labels, right=False)   

        # Group the data by 'post' and 'time_bin', then calculate the mean weight for each group
        summed_weights = self.data_smth.groupby(['post', 'time_bin'])['weight'].mean().reset_index()

        # Pivot the data so that each 'post' neuron has its own row and each time bin is a column
        pivoted = summed_weights.pivot(index='post', columns='time_bin', values='weight')

        # Fill any missing values (NaN) with 0 and convert the pivoted DataFrame to a NumPy array
        self.data_smth = pivoted.fillna(0).values


    def get_data_range(self, bins):
        """
        Calculates the global minimum (vmin) and maximum (vmax) values of smoothed data 
        across multiple time bins. These values can be used for consistent color scaling 
        in visualizations.
        
        Parameters
        ----------
        bins : list of arrays
            A list of arrays, where each array represents the bin edges for a particular 
            time interval over which to calculate the smoothed data.
            
        Returns
        -------
        None
        """
        # Initialize vmin and vmax with extreme values
        self.vmin = np.inf
        self.vmax = -np.inf  # Changed to -np.inf for consistency
        
        # Iterate over each set of bins provided
        for bin in bins:
            # Create a deep copy of the data to avoid modifying the original dataset
            data_smth = self.data.copy(deep=True)

            # Extend the current bin array to ensure the final bin covers all data points
            bin = np.append(bin, [2 * bin[-1] - bin[-2]])
            bin_labels = bin[:-1]

            # Bin the time data and calculate the mean weight for each bin
            data_smth['time_bin'] = pd.cut(data_smth['time'], bins=bin, labels=bin_labels, right=False)   
            summed_weights = data_smth.groupby(['post', 'time_bin'])['weight'].mean().reset_index()

            # Pivot the data so that each 'post' neuron has its own row and each time bin is a column
            pivoted = summed_weights.pivot(index='post', columns='time_bin', values='weight')
            data_smth = pivoted.fillna(0).values

            # Update the global minimum and maximum values if necessary
            if np.min(data_smth) < self.vmin:
                self.vmin = np.min(data_smth)
            
            if np.max(data_smth) > self.vmax:
                self.vmax = np.max(data_smth)



    # detect outliers using the Z-score
    @staticmethod
    def trim_outliers_zscore(data, threshold=3):
        """
        Identifies and scales down outliers in the dataset based on z-scores.

        Parameters
        ----------
        data : np.array
            The input data array where each column represents a vector.
        threshold : float, optional
            The z-score threshold above which a data point is considered an outlier. 
            The default is 3.

        Returns
        -------
        np.array
            The transformed data with outliers scaled down.
        np.array
            An array indicating the color coding for each vector, where outliers are marked differently.
        """

        # Transpose the data to work with rows as vectors for easier processing
        transform = data.copy().T

        # Calculate the magnitude of each vector (Euclidean norm)
        magnitudes = np.linalg.norm(transform, axis=1)

        # Compute z-scores of the magnitudes
        z_scores = stats.zscore(magnitudes)

        # Identify indices of outliers based on the z-score threshold
        indices = np.nonzero(np.abs(z_scores) > threshold)

        # Identify non-outlier magnitudes to calculate the scaling factor
        non_outliers = magnitudes[np.where(np.abs(z_scores) <= threshold)]
        scaling_factor = np.max(non_outliers)

        # Scale down the outlier vectors while preserving their direction
        transform[indices] *= scaling_factor / magnitudes[indices][:, np.newaxis]

        # Create a color array to mark outliers differently (1 for non-outliers, 3 for outliers)
        colors = np.full_like(transform[:, 0], 1.0, dtype=float).reshape(-1, 1)
        np.put(colors, indices, 3.0)

        # Return the transformed data (back to its original orientation) and the color information
        return transform.T, colors.T


    @staticmethod
    def normalize(var, var_range):
        """
        Normalizes the input variable to a range between 0 and 1.

        Parameters
        ----------
        var : np.array or float
            The input variable or array of variables to be normalized.
        var_range : tuple
            A tuple containing the minimum and maximum values (min_, max_) 
            that define the range for normalization.

        Returns
        -------
        np.array or float
            The normalized variable or array, scaled to the range [0, 1].
        """

        min_ = var_range[0]
        max_ = var_range[1]

        return (var - min_) / (max_ - min_)


    @staticmethod
    def normalize_vec(var, rangex, rangey):
        """
        Normalizes a 2D vector array to specific x and y ranges.

        Parameters
        ----------
        var : np.array
            A 2D numpy array where `var[0, :]` represents the x-components 
            and `var[1, :]` represents the y-components of the vectors.
        rangex : tuple
            A tuple containing the minimum and maximum values (min_, max_) 
            that define the range for the x-components.
        rangey : tuple
            A tuple containing the minimum and maximum values (min_, max_) 
            that define the range for the y-components.

        Returns
        -------
        np.array
            The normalized 2D vector array.
        """

        # Normalize the entire vector array by dividing by its maximum value
        var = var / var.max()

        # Normalize the x-components
        min_ = rangex[0]
        max_ = rangex[1]
        var[0, :] = var[0, :] * (max_ - min_)

        # Normalize the y-components
        min_ = rangey[0]
        max_ = rangey[1]
        var[1, :] = var[1, :] * (max_ - min_)

        return var



    # TODO: very similiar to vector_field_batch
    def plot_vector_field(self, fl_suffix='', frm='pdf', cell_type='place', 
                      border_flag=False, show_obs=False):
        """
        Plots the vector feedforward weight of each cell of the desired type 
        and represents an arrow at the topological position of that cell.

        Parameters
        ----------
        fl_suffix : string
            Suffix to append to saved file name. The default is ''.
        frm : string 
            Format of the saved file. The default is 'pdf'.
        cell_type : string
            The population to draw a vector field for. The default is 'place'.
        border_flag : bool
            Option determining whether or not a border should be drawn around 
            the vector field. This may not be the same borders as those defined 
            by the environment boundaries - see vector_border().
        show_obs : bool
            Option determining whether or not to draw vectors for obstacle cells.

        Returns
        -------
        None.
        """
        self.read_files(cell_type=cell_type)
        scale = 0
        if cell_type == 'place':
            self.get_times_from_DA(dop_flname='dopamine_p-0.gdf')
            scale = 0.5
        elif cell_type == 'grid':
            self.get_times_from_DA(dop_flname='dopamine_g-0.gdf')
            scale = 0.5
        elif cell_type == 'border':
            self.get_times_from_DA(dop_flname='dopamine_b-0.gdf')
            self.get_init_ws()
            self.get_action_vecs()
            self.get_firing_fields(filename='borderID_pos.dat')
            self.w_range = [0, 1]
            self.times = [0]
            scale = 1.8
        else:
            raise ValueError('cell_type must be provided to plot_vector_field')
        
        for time in self.times:
            flname = 'vectorfield-weight_{}-{:.0f}ms-{}.{}'.format(
                cell_type[0], time, fl_suffix, frm)
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
                    scale=scale)
            
            self.format_plot(ax, 't = {} ms'.format(time))
            ax.set_xlabel('x')
            ax.set_ylabel('y')

            if cell_type == 'border' and show_obs:
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
            
            if not self.hide_goal:
                tr = self.get_trial_from_t(time)
                if tr <= self.sim_dict['max_num_trs']:
                    add_goal_zone(ax, os.path.join(self.sim_dict['data_path'], 
                                                'trials_params.dat'), tr=tr)
                else:
                    print(f'Warning: found dopamine spikes after {self.sim_dict["max_num_trs"]} trials concluded')
            
            if border_flag:
                self.vector_border(ax, ls='dotted')
            
            fig.savefig(os.path.join(self.fig_path, flname), format=frm)
            plt.close(fig)
