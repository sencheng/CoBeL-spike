#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 16:33:16 2023

Helper functions for making analysis plots.

@author: Ray Black
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_trials_params(trial_path):
    return pd.read_csv(trial_path, sep="\t")


def tr_reward(tr, trials_params):
    try:
        if tr:
            trial_dummy = trials_params.loc[trials_params['trial_num'] == tr]
            start_x = trial_dummy['start_x'].values[0]
            start_y = trial_dummy['start_y'].values[0]

            goal_x = trial_dummy['goal_x'].values[0]
            goal_y = trial_dummy['goal_y'].values[0]
            goal_position = np.array([goal_x, goal_y])
            goal_shape = trial_dummy['goal_shape'].values[0]
            goal_size1 = trial_dummy['goal_size1'].values[0]
            goal_size2 = trial_dummy['goal_size2'].values[0]
        else:
            start_x, start_y = 0, 0
            goal_x, goal_y = 0, 0
            goal_shape, goal_size1, goal_size2 = None, 0, 0
    except IndexError:
        print(f'Trial reward data not found for trial {tr}!')
        raise IndexError

    tr_reward_dict = {
        'start_x': start_x,
        'start_y': start_y,
        'goal_x': goal_x,
        'goal_y': goal_y,
        'goal_shape': goal_shape,
        'goal_size1': goal_size1,
        'goal_size2': goal_size2,
    }

    trials_params = trials_params.loc[trials_params['trial_num'] <= tr]
    trials_params = trials_params.drop(['trial_num'], axis=1).drop_duplicates()

    start_x = trials_params['start_x'].to_numpy()
    start_y = trials_params['start_y'].to_numpy()

    goal_x = trials_params['goal_x'].to_numpy()
    goal_y = trials_params['goal_y'].to_numpy()
    goal_shape = trials_params['goal_shape'].to_numpy()
    goal_size1 = trials_params['goal_size1'].to_numpy()
    goal_size2 = trials_params['goal_size2'].to_numpy()

    all_reward_dict = {
        'start_x': start_x,
        'start_y': start_y,
        'goal_x': goal_x,
        'goal_y': goal_y,
        'goal_shape': goal_shape,
        'goal_size1': goal_size1,
        'goal_size2': goal_size2,
    }

    return tr_reward_dict, all_reward_dict


def add_goal_zone(ax, trial_path, tr=1, c='purple', lw=1):
    """
    Adds a visual representation of the reward zone to a given ax.
    
    Parameters:
    -----------
    ax : matplotlib axis
        The axis to add the reward zone to.
    trial_path : str
        The path to the trial parameters file.
    tr : int, optional
        The trial of the associated plot (default is 1).
    c : str, optional
        The color of the goal zone (default is 'purple').
    lw : int, optional
        The line width of the goal zone outline (default is 1).
    """
    trials_params = read_trials_params(trial_path)
    tr_reward_dict, all_reward_dict = tr_reward(tr, trials_params)

    if tr_reward_dict['goal_shape'] == 'round':
        goal_zone = plt.Circle(
            (tr_reward_dict['goal_x'], tr_reward_dict['goal_y']),
            tr_reward_dict['goal_size1'],
            color=c,
            alpha=1,
            fill=False,
            linewidth=4,
            label='current goal zone'
        )
        ax.add_patch(goal_zone)
    elif tr_reward_dict['goal_shape'] == 'rect':
        goal_x = tr_reward_dict['goal_x']
        goal_y = tr_reward_dict['goal_y']
        delta_x = tr_reward_dict['goal_size1']
        delta_y = tr_reward_dict['goal_size2']

        ll = (goal_x - delta_x, goal_y - delta_y)  # lower left
        lr = (goal_x + delta_x, goal_y - delta_y)  # lower right
        ur = (goal_x + delta_x, goal_y + delta_y)  # upper right
        ul = (goal_x - delta_x, goal_y + delta_y)  # upper left

        goal_zone = plt.Polygon(
            [ll, lr, ur, ul],
            closed=True,
            color=c,
            alpha=0.1,
            label='Goal zone'
        )
        ax.add_patch(goal_zone)

    for i in range(len(all_reward_dict['goal_x'])):
        if all_reward_dict['goal_shape'][i] == 'round':
            goal_zone = plt.Circle(
                (all_reward_dict['goal_x'][i], all_reward_dict['goal_y'][i]),
                all_reward_dict['goal_size1'][i],
                color=c,
                alpha=1,
                fill=False,
                linewidth=2,
                linestyle='--',
                label='previous goal zone' if i == 0 else None
            )
        elif all_reward_dict['goal_shape'][i] == 'rect':
            goal_x = all_reward_dict['goal_x'][i]
            goal_y = all_reward_dict['goal_y'][i]
            delta_x = all_reward_dict['goal_size1'][i]
            delta_y = all_reward_dict['goal_size2'][i]

            ll = (goal_x - delta_x, goal_y - delta_y)  # lower left
            lr = (goal_x + delta_x, goal_y - delta_y)  # lower right
            ur = (goal_x + delta_x, goal_y + delta_y)  # upper right
            ul = (goal_x - delta_x, goal_y + delta_y)  # upper left

            goal_zone = plt.Polygon(
                [ll, lr, ur, ul],
                closed=True,
                fill=False,
                color=c,
                alpha=0.1,
                label=None
            )
        ax.add_patch(goal_zone)


def add_all_goal_zones(ax, sim_dict, highlight=[], c='crimson'):
    """
    Adds all goal zones without overlapping identical goal zones.

    Parameters:
    -----------
    ax : matplotlib axis
        The axis to add the goal zones to.
    sim_dict : dict
        Simulation parameters dictionary including reward zone parameters.
    highlight : list, optional
        List of goal zones to highlight.
    c : str, optional
        The color of the goal zones (default is 'crimson').
    """
    trials_params = read_trials_params(os.path.join(sim_dict['data_path'], 'trials_params.dat'))
    trials_params.drop_duplicates(subset=['goal_rad', 'goal_x', 'goal_y'], inplace=True)

    for tr in trials_params.values:
        if tr[0] in highlight:
            goal_zone = plt.Circle(
                (tr[2], tr[3]),  # x, y values of goal center
                tr[1],           # goal radius
                color=c,
                alpha=0.25,
                fill=True,
                label='reward zone'
            )
        else:
            goal_zone = plt.Circle(
                (tr[2], tr[3]),  # x, y values of goal center
                tr[1],           # goal radius
                color=c,
                alpha=1,
                fill=False,
                lw=2,
                label='reward zone'
            )

        x, y = goal_zone.get_center()
        radius = goal_zone.get_radius()
        x_boundaries = ax.get_xlim()
        y_boundaries = ax.get_ylim()

        if (
            x - radius >= x_boundaries[0] and
            x + radius <= x_boundaries[1] and
            y - radius >= y_boundaries[0] and
            y + radius <= y_boundaries[1]
        ):
            ax.add_patch(goal_zone)


def add_obstacles(ax, env_dict, c='k'):
    """
    Adds visual representations of obstacles to a given ax.
    
    Parameters:
    -----------
    ax : matplotlib axis
        The axis to add the obstacles to.
    env_dict : dict
        Environment dictionary containing obstacle parameters.
    c : str, optional
        The color of the obstacles (default is 'k' for black).
    """
    obs_dict = env_dict["environment"]["obstacles"]
    if obs_dict["flag"]:
        for center, vert, horiz in zip(obs_dict["centers"], obs_dict["vert_lengths"], obs_dict["horiz_lengths"]):
            delta_y = vert / 2.  # Get the length and width
            delta_x = horiz / 2.  # as distances from the center point

            ll = (center[0] - delta_x, center[1] - delta_y)  # lower left
            lr = (center[0] + delta_x, center[1] - delta_y)  # lower right
            ur = (center[0] + delta_x, center[1] + delta_y)  # upper right
            ul = (center[0] - delta_x, center[1] + delta_y)  # upper left

            obs_actual = plt.Polygon(
                [ll, lr, ur, ul],
                closed=True,
                color=c,
                zorder=1.1  # zorder should put obstacle patches above other patches
            )
            ax.add_patch(obs_actual)


def add_tmaze(ax, env_dict):
    """
    Adds a visual representation of a T-maze to a given ax.
    
    Parameters:
    -----------
    ax : matplotlib axis
        The axis to add the T-maze to.
    env_dict : dict
        Environment dictionary containing T-maze parameters.
    """
    maze_params = env_dict["environment"]["tmaze"]
    x_min = maze_params['xmin_position']
    x_max = maze_params['xmax_position']
    y_min = maze_params['ymin_position']
    y_max = maze_params['ymax_position']
    gaw = maze_params['goal_arm_width']
    cw = maze_params['corridor_width']

    ax.hlines(y_max - gaw, x_min, -cw / 2)
    ax.hlines(y_max - gaw, cw / 2, x_max)
    ax.vlines(-cw / 2, y_min, y_max - gaw)
    ax.vlines(cw / 2, y_min, y_max - gaw)
