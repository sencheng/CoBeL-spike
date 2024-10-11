import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis import PositionFile as pos
from .PlotHelper import read_trials_params, tr_reward


class BehavioralPerformance:
    """Computes some measures about the performance of the agent."""

    def __init__(
        self,
        path_dict={'nest_data_path': '', 'param_path': ''},
        flname_dict={
            'tr_time_pos': '',
            'param_file': {'sim': '', 'net': ''}
        },
        fig_path='',
        rec_all_beh_data=True,
        beh_data_flname='complete_beh_data.dat',
        dtw_opt_flname='DTW_optimal_path.dat',
        reset_summary_file=False,
        perf_params=None
    ):

        with open(os.path.join(path_dict['param_path'], flname_dict['param_file']['sim'])) as par_f:
            self.sim_params = json.load(par_f)

        with open(os.path.join(path_dict['param_path'], flname_dict['param_file']['net'])) as par_f:
            self.net_params = json.load(par_f)

        with open(os.path.join(path_dict['param_path'], flname_dict['param_file']['env'])) as par_f:
            self.env_params = json.load(par_f)

        self.data = pd.read_csv(
            os.path.join(path_dict['nest_data_path'], flname_dict['tr_time_pos']),
            sep='\t'
        )

        self.fig_path = fig_path
        self.perf_params = perf_params

        if rec_all_beh_data:
            master_rng_seed = str(self.sim_params['master_rng_seed'])
            data_dir = self.sim_params['data_path']
            main_data_dir = os.path.join(*data_dir.split('/')[0:-1])
            fig_dir = os.path.join(main_data_dir, 'fig-' + master_rng_seed)
            beh_data_flpath = os.path.join(fig_dir, beh_data_flname)
            print(f'\n writing summary data in {beh_data_flpath}')
            file_exists = os.path.exists(beh_data_flpath)
            if file_exists and reset_summary_file:
                os.remove(beh_data_flpath)
                file_exists = False

            self.log_file = open(beh_data_flpath, 'a')  # creates the text file
            if not file_exists:
                self.write_header()

        self.trials_params = read_trials_params(
            os.path.join(self.sim_params['data_path'], 'trials_params.dat')
        )
        self.group_data()

        if self.sim_params['max_num_trs'] > self.num_trials:
            print('\nInsufficient number of trials ...\n')
        elif self.sim_params['max_num_trs'] < self.num_trials:
            self.num_trials = self.sim_params['max_num_trs']
            self.tr_ids = np.arange(1, self.num_trials + 1)

        calc_all_goals = True
        if calc_all_goals:
            # This creates a dataframe matching all trials to all unique goal zones
            goal_list = self.trials_params.drop_duplicates(
                subset=['goal_shape', 'goal_size1', 'goal_size2', 'goal_x', 'goal_y']
            ).drop(columns='trial_num')
            goal_per_simulation = len(goal_list)
        else:
            goal_per_simulation = 1

        self.trial = np.zeros(self.num_trials * goal_per_simulation)
        self.traj_length = np.zeros(self.num_trials * goal_per_simulation)
        self.tr_duration = np.zeros(self.num_trials * goal_per_simulation)
        self.goal_x = np.zeros(self.num_trials * goal_per_simulation)
        self.goal_y = np.zeros(self.num_trials * goal_per_simulation)
        self.reward_rad1 = np.zeros(self.num_trials * goal_per_simulation)
        self.reward_rad2 = np.zeros(self.num_trials * goal_per_simulation)
        self.prox_min = np.zeros(self.num_trials * goal_per_simulation)
        self.prox_mean = np.zeros(self.num_trials * goal_per_simulation)

        i = 0
        for tr in self.tr_ids:
            tmp = self.grp_data.get_group(tr).to_numpy()[:, 1:]

            tr_duration = (tmp[-1, 0] - tmp[0, 0]) * 1000
            traj_length = np.sum(np.sqrt(np.diff(tmp[:, 1]) ** 2 + np.diff(tmp[:, 2]) ** 2))

            if calc_all_goals:
                tr_reward_dict = goal_list
            else:
                tr_reward_dict, _ = tr_reward(tr, self.trials_params)
                tr_reward_dict = pd.DataFrame([tr_reward_dict])

            for _, row in tr_reward_dict.iterrows():
                self.trial[i] = tr
                self.tr_duration[i] = tr_duration
                self.traj_length[i] = traj_length
                self.reward_rad1[i] = row['goal_size1']
                self.reward_rad2[i] = row['goal_size2']
                self.goal_x[i] = row['goal_x']
                self.goal_y[i] = row['goal_y']

                self.prox_min[i], self.prox_mean[i] = self.calc_prox(tmp, row)
                i += 1

        dtw_opt_flpath = os.path.join(fig_dir, dtw_opt_flname)
        if 'DTW_opt' in self.perf_params:
            if os.path.exists(dtw_opt_flpath):
                df = pd.read_csv(dtw_opt_flpath, sep='\t')
                self.DTW_opt = df['DTW'].to_numpy()
            else:
                print('DTW file not found. Calculating DTW...')
                pos_obj = pos.PositionFile(
                    data_path=self.sim_params['data_path'],
                    filename='locs_time.dat',
                    fig_path=self.fig_path
                )
                pos_obj.read_pos_file()
                self.DTW_opt = pos_obj.calc_DTW_optimal(formats=[])['DTW'].to_numpy()
        else:
            self.DTW_opt = None

        self.param_dict = {
            'seed': self.sim_params['master_rng_seed'],
            'trial': [int(tr) for tr in self.trial],
            'p_ncols': self.net_params['place']['cells_prop']['p_ncols'],
            'p_sigma': self.net_params['place']['cells_prop']['p_row_sigma'],
            'A_plus': self.net_params['place']['syn_params']['A_plus'],
            'A_minus': self.net_params['place']['syn_params']['A_minus'],
            'p_max_fr': self.net_params['place']['cells_prop']['max_fr'],
            'tr_duration': self.tr_duration,
            'traj_length': self.traj_length,
            'proximity_min': self.prox_min,
            'proximity_mean': self.prox_mean,
            'DTW_opt': self.DTW_opt,
            'goal_size1': self.reward_rad1,
            'goal_x': self.goal_x,
            'goal_y': self.goal_y,
            'num_goal': goal_per_simulation
        }

    def calc_prox(self, tr_group_dat, tr_reward_dict):
        obs_dict = self.env_params["environment"]["obstacles"]
        obstacle_list = []
        goal_center = (tr_reward_dict['goal_x'], tr_reward_dict['goal_y'])
        if obs_dict["flag"]:
            for center, vert, horiz in zip(
                obs_dict["centers"], obs_dict["vert_lengths"], obs_dict["horiz_lengths"]
            ):
                delta_y = vert / 2.0
                delta_x = horiz / 2.0
                ll = (center[0] - delta_x, center[1] - delta_y)  # lower left
                lr = (center[0] + delta_x, center[1] - delta_y)  # lower right
                ur = (center[0] + delta_x, center[1] + delta_y)  # upper right
                ul = (center[0] - delta_x, center[1] + delta_y)  # upper left
                obstacle_list.append([ll, lr, ur, ul])

        prox_arr = np.zeros(len(tr_group_dat))
        for i_r, row in enumerate(tr_group_dat):
            pos = (row[1], row[2])
            for obs in obstacle_list:
                for i_o, _ in enumerate(obs):
                    obs_segment = (obs[i_o], obs[i_o - 1])
                    if self._intersect(obs_segment[0], obs_segment[1], pos, goal_center):
                        prox_arr[i_r] = np.nan
            if not np.isnan(prox_arr[i_r]):
                prox_arr[i_r] = np.linalg.norm(row[1:] - goal_center) - tr_reward_dict['goal_size1']

        prox_arr = np.linalg.norm(tr_group_dat[:, 1:] - goal_center, axis=1) - tr_reward_dict['goal_size1']
        prox_arr = [max(0, x) for x in prox_arr]

        return np.nanmin(prox_arr), np.nanmean(prox_arr)

    @staticmethod
    def _ccw(A, B, C):
        """Checks if three given points are arranged in a counter-clockwise order."""
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    @staticmethod
    def _intersect(A, B, C, D):
        """Checks if two line segments AB and CD intersect."""
        return (BehavioralPerformance._ccw(A, C, D) != BehavioralPerformance._ccw(B, C, D) and
                BehavioralPerformance._ccw(A, B, C) != BehavioralPerformance._ccw(A, B, D))

    def group_data(self):
        """Groups the data by trials in a pandas DataFrame format."""
        self.grp_data = self.data.groupby(by='trial')
        self.num_trials = self.grp_data.ngroups
        self.tr_ids = list(self.grp_data.groups.keys())

    def get_performance(self):
        """Computes the performance of the agent as trial duration and
        trajectory length as a function of trials.
        """
        for i, tr in enumerate(self.trial):
            if hasattr(self, 'log_file'):
                self.write_data(tr, i)

        if hasattr(self, 'log_file'):
            self.log_file.close()

    def plot_performance(self, perf_plots=None, fl_suffix='', frm='pdf'):
        """Plots the computed performance.

        Parameters
        ----------
        perf_plots : list
            List of measures to plot. Defaults to every possible measure.
            Note that DTW takes more time to calculate than other measures.
        fl_suffix : str, optional
            String to add to the end of the file, e.g., to distinguish outcomes
            of different simulations. Default is an empty string.
        frm : str, optional
            Format of the output figure. Default is 'pdf'.

        Returns
        -------
        None.

        """
        if perf_plots is None:
            perf_plots = ['tr_duration', 'traj_length', 'DTW_opt']

        if not hasattr(self, 'tr_duration'):
            self.get_performance()

        fig, ax = plt.subplots(nrows=len(perf_plots), ncols=1, sharex=True)

        for i, plot in enumerate(perf_plots):
            if plot not in self.perf_params:
                raise ValueError(
                    f'{plot} passed as parameter to plotting function without being calculated. '
                    'Ensure all desired plots are also given in perf_params.'
                )
            if plot == 'tr_duration':
                n = int(len(self.tr_duration) / len(self.tr_ids))
                ax[i].bar(self.tr_ids, self.tr_duration[::n])
                ax[i].set_ylabel('Trial duration (ms)')
            elif plot == 'traj_length':
                n = int(len(self.traj_length) / len(self.tr_ids))
                ax[i].bar(self.tr_ids, self.traj_length[::n])
                ax[i].set_ylabel('Trajectory length (a.u.)')
            elif plot == 'DTW_opt':
                n = int(len(self.DTW_opt) / len(self.tr_ids))
                ax[i].bar(self.tr_ids, self.DTW_opt[::n])
                ax[i].set_ylabel('DTW vs optimal path (a.u.)')

        ax[-1].set_xlabel('Trial #')
        plt.tight_layout()
        fig.savefig(
            os.path.join(self.fig_path, f'behavior_performance{fl_suffix}.{frm}'),
            format=frm
        )
        plt.close(fig)

    def write_data(self, tr, i):
        """Writes/appends the performance data and the parameters used
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
        for param in self.perf_params:
            try:
                self.log_file.write(f'{self.param_dict[param][i]}\t')
            except TypeError:
                self.log_file.write(f'{self.param_dict[param]}\t')

        self.log_file.write('\n')

    def write_header(self):
        """Writes the header of the performance data file."""
        for param in self.perf_params:
            self.log_file.write(f'{param}\t')
        self.log_file.write('\n')
