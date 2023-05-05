#!/usr/bin/env python3

import os
import json

import pandas as pd
import numpy as np


class BehavioralPerformanceComparison():

    def __init__(self, data_path, flname='data_paths.json'):
        with open(os.path.join(data_path, flname)) as fl:
            paths_dict = json.load(fl)

        self.paths = paths_dict['data_path']

    def read_behavior_analysis_files(self, flname='behvavior_data.txt'):
        self.all_bh_data = pd.DataFrame()
        self.agent_ids = np.zeros(len(self.paths))

        for i, path in enumerate(self.paths):
            agent_id = int(path.split('tr')[-1])
            self.agent_ids[i] = agent_id

            tmp = pd.read_csv(os.path.join(path, flname))
            tmp = tmp.drop(labels=['Unnamed: 0'], axis=1)
            tmp = tmp.astype({'trial': int})
            tmp.insert(loc=0, column='agent',
                       value=agent_id * np.ones(tmp.shape[0]).astype(int))

            self.all_bh_data = pd.concat((self.all_bh_data, tmp.copy()), axis=0)

        self.grp_bh_data = self.all_bh_data.groupby(by='agent')

    def get_success_stats(self):
        self.num_success = np.zeros_like(self.agent_ids)
        self.med_time_to_reward = np.zeros_like(self.agent_ids)
        self.med_dist_to_reward = np.zeros_like(self.agent_ids)
        self.mean_time_to_reward = np.zeros_like(self.agent_ids)
        self.mean_dist_to_reward = np.zeros_like(self.agent_ids)

        for i, a_id in enumerate(self.agent_ids):
            self.num_success[i] = self.grp_bh_data.get_group(a_id).trial.max()
            self.med_time_to_reward[i] = self.grp_bh_data.get_group(a_id).duration.median()
            self.med_dist_to_reward[i] = self.grp_bh_data.get_group(a_id).distance.median()
            self.mean_time_to_reward[i] = self.grp_bh_data.get_group(a_id).duration.mean()
            self.mean_dist_to_reward[i] = self.grp_bh_data.get_group(a_id).distance.mean()

        self.total_num_success = self.num_success.sum()

        return self.agent_ids, self.num_success

    def plot_success(self, ax, label, var, sort=False, sort_by=None):

        # if not hasattr(self, 'num_success'):
        #     self.get_success_stats()

        if sort & (sort_by is None):
            Var = np.sort(var)[::-1]
        elif sort & (sort_by is not None):
            Var = var[np.argsort(sort_by)[::-1]]
        else:
            Var = var

        ax.plot(self.agent_ids, Var, label=label)
