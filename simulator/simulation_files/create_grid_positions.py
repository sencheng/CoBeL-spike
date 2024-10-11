#!/usr/bin/env python3

"""
This file generates the grid_pos.json file.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../analysis/analysis/')
from PlotHelper import add_goal_zone

# File paths
in_network_params = "parameter_sets/current_parameter/network_params_spikingnet.json"
in_sim_params = "parameter_sets/current_parameter/sim_params.json"
env_params_path = "parameter_sets/current_parameter/env_params.json"
out_all_cells = "grid_pos.json"

# Read the parameter files
with open(in_network_params, "r") as f:
    net_dict = json.load(f)
with open(in_sim_params, "r") as f:
    sim_dict = json.load(f)
with open(env_params_path, "r") as f:
    env_dict = json.load(f)

master_rng_seed = str(sim_dict['master_rng_seed'])
data_dir = sim_dict['data_path']
main_data_dir = os.path.join(*data_dir.split('/')[:-1])
fig_dir = os.path.join(main_data_dir, f'fig-{master_rng_seed}')
fig_path = f'{fig_dir}/grid_poss.png'

env_type = env_dict['sim_env']
env_params = env_dict['environment'][env_type]
sim_obs_dict = env_dict['environment']['obstacles']

# Outer boundaries for the environment
xmin_pos, xmax_pos = env_params['xmin_position'], env_params['xmax_position']
ymin_pos, ymax_pos = env_params['ymin_position'], env_params['ymax_position']
outer_xlim = np.array([xmin_pos, xmax_pos])
outer_ylim = np.array([ymin_pos, ymax_pos])

# References to parameters for the different cell types
try:
    place = net_dict['place']
except KeyError:
    place = {'num_neurons': 0}

try:
    grid = net_dict['grid']
except KeyError:
    grid = {'num_neurons': 0}

try:
    border = net_dict['border']
except KeyError:
    border = {'num_neurons': 0}

try:
    obstacle = net_dict['obstacle']
except KeyError:
    obstacle = {'num_neurons': 0}

try:
    noise = net_dict['noise']
except KeyError:
    noise = {'num_neurons': 0}

# Lists to store all the different types of cells
pos, place_pos, grid_pos, border_pos, obs_pos, noise_pos = [], [], [], [], [], []

def grid_in_env(dummy: list) -> list:
    """Ensure that the grid points are inside the environment."""
    x_range, y_range = outer_xlim[1] - outer_xlim[0], outer_ylim[1] - outer_ylim[0]
    dummy[0] = ((dummy[0] - outer_xlim[0]) % x_range) + outer_xlim[0]
    dummy[1] = ((dummy[1] - outer_ylim[0]) % y_range) + outer_ylim[0]


# Generate place cells
if place["num_neurons"]:
    p_neurons_x = place['cells_prop']['p_nrows']
    p_neurons_y = place['cells_prop']['p_ncols']
    p_sig_x = place['cells_prop']['p_row_sigma']
    p_sig_y = place['cells_prop']['p_col_sigma']
    p_rep_index = place['cells_prop']['rep_index']
    p_fr = place['cells_prop']['max_fr']
    p_width = place['spatial_prop']['width']
    p_height = place['spatial_prop']['height']

    for x in np.linspace(-p_width / 2, p_width / 2, p_neurons_x):
        shift = p_height / 2 if p_neurons_y == 1 else 0.0

        for y in np.linspace(p_height / 2, -p_height / 2, p_neurons_y):
            dummy = [float(x), float(y - shift), float(p_sig_x), float(p_sig_y), p_rep_index, p_fr]
            place_pos.append(dummy)

# Generate grid cells
if grid["num_neurons"]:
    g_nrows = grid['cells_prop']['g_nrows']
    g_ncols = grid['cells_prop']['g_ncols']
    g_kappa = grid['cells_prop']['g_kappa']
    g_lambda = grid['cells_prop']['g_lambda']
    g_rep_index = grid['cells_prop']['rep_index']
    g_fr = grid['cells_prop']['max_fr']

    x_range, y_range = outer_xlim[1] - outer_xlim[0], outer_ylim[1] - outer_ylim[0]
    n_grids = len(g_nrows)
    grid_start = zip(np.random.rand(n_grids) * x_range - outer_xlim[0], np.random.rand(n_grids) * y_range - outer_ylim[0])

    for start, row, col, lam, kap, fr in zip(grid_start, g_nrows, g_ncols, g_lambda, g_kappa, g_fr):
        for RS, y in enumerate(np.linspace(start[1], start[1] + lam * np.cos(np.pi / 6), row, endpoint=False), start=1):
            for x in np.linspace(start[0], start[0] + lam, col, endpoint=False):
                x_offset = (lam / 2) / row
                dummy = [x + (x_offset * RS), y, kap, lam, 1, fr]
                if 'disorder' in grid.keys():
                    dis = grid['disorder']
                    np.random.seed(sim_dict['master_rnd_seed'] + RS)
                    dummy[0] += np.random.uniform(-dis, dis, 2)
                grid_in_env(dummy)
                grid_pos.append(dummy)

# Generate border cells
if border["num_neurons"] and border['cells_prop']['flag']:
    xmid_pos, ymid_pos = (xmin_pos + xmax_pos) / 2, (ymin_pos + ymax_pos) / 2
    b_width = border['cells_prop']['width']
    b_rep_index = border['cells_prop']['rep_index']
    b_fr = border['cells_prop']['max_fr']

    if env_type == 'openfield':
        border_pos.extend([
            [xmid_pos, ymax_pos, xmax_pos - b_width, b_width, b_rep_index, b_fr],
            [xmax_pos, ymax_pos, b_width, b_width, b_rep_index, b_fr],
            [xmax_pos, ymid_pos, b_width, ymax_pos - b_width, b_rep_index, b_fr],
            [xmax_pos, ymin_pos, b_width, b_width, b_rep_index, b_fr],
            [xmid_pos, ymin_pos, xmax_pos - b_width, b_width, b_rep_index, b_fr],
            [xmin_pos, ymin_pos, b_width, b_width, b_rep_index, b_fr],
            [xmin_pos, ymid_pos, b_width, ymax_pos - b_width, b_rep_index, b_fr],
            [xmin_pos, ymax_pos, b_width, b_width, b_rep_index, b_fr]
        ])
    elif env_type == 'tmaze':
        gaw = env_params['goal_arm_width']
        cw = env_params['corridor_width']
        y_width = ymax_pos - ymin_pos
        block = xmax_pos - (cw / 2)
        border_pos.extend([
            [xmid_pos, ymax_pos, ymax_pos - b_width, b_width, b_rep_index, b_fr],
            [xmax_pos, ymax_pos, b_width, b_width, b_rep_index, b_fr],
            [xmax_pos, ymax_pos - (gaw / 2), b_width, gaw / 2 - b_width, b_rep_index, b_fr],
            [xmax_pos, ymax_pos - gaw, b_width, b_width, b_rep_index, b_fr],
            [xmax_pos - block / 2 - b_width / 2, ymax_pos - gaw, (block - b_width) / 2, b_width, b_rep_index, b_fr],
            [xmax_pos - block, ymid_pos - gaw / 2 + b_width / 2, b_width, (y_width - gaw) / 2 - b_width / 2, b_rep_index, b_fr],
            [xmid_pos + cw / 2, ymin_pos, b_width, b_width, b_rep_index, b_fr],
            [xmid_pos, ymin_pos, cw / 2 - b_width, b_width, b_rep_index, b_fr],
            [xmid_pos - cw / 2, ymin_pos, b_width, b_width, b_rep_index, b_fr],
            [xmin_pos + block, ymid_pos - gaw / 2 + b_width / 2, b_width, (y_width - gaw) / 2 - b_width / 2, b_rep_index, b_fr],
            [xmin_pos + block / 2 + b_width / 2, ymax_pos - gaw, (block - b_width) / 2, b_width, b_rep_index, b_fr],
            [xmin_pos, ymax_pos - gaw, b_width, b_width, b_rep_index, b_fr],
            [xmin_pos, ymax_pos - (gaw / 2), b_width, gaw / 2 - b_width, b_rep_index, b_fr],
            [xmin_pos, ymax_pos, b_width, b_width, b_rep_index, b_fr]
        ])
    else:
        raise NotImplementedError("Only 'openfield' and 'tmaze' environments are supported.")

# Generate obstacle cells
if sim_obs_dict['flag']:
    ob_width = obstacle['cells_prop']['width']
    ob_rep_index = obstacle['cells_prop']['rep_index']
    ob_fr = obstacle['cells_prop']['max_fr']
    for center, vert, horiz in zip(sim_obs_dict["centers"], sim_obs_dict["vert_lengths"], sim_obs_dict["horiz_lengths"]):
        delta_y = vert / 2.0
        delta_x = horiz / 2.0
        x = float(center[0])
        y = float(center[1])
        obs_pos.extend([
            [x, y + delta_y, delta_x, ob_width, ob_rep_index, ob_fr],
            [x + delta_x, y, ob_width, delta_y, ob_rep_index, ob_fr],
            [x, y - delta_y, delta_x, ob_width, ob_rep_index, ob_fr],
            [x - delta_x, y, ob_width, delta_y, ob_rep_index, ob_fr]
        ])

# Generate noise cells
if noise["num_neurons"]:
    noise_target_list = list(range(noise["num_neurons"]))
    noise_connect_list = [0] * noise["num_neurons"]
    noise_active_index = [noise['cells_prop']['start_trial']] * noise["num_neurons"]
    noise_active_time_dummy = [0] * noise["num_neurons"]
    noise_index_list = [noise['cells_prop']['rep_index']] * noise["num_neurons"]
    noise_fr_list = [noise['cells_prop']['max_fr']] * noise["num_neurons"]
    noise_pos = [[noise_target_list[i], noise_connect_list[i], noise_active_index[i], 
                  noise_active_time_dummy[i], noise_index_list[i], noise_fr_list[i]] for i in range(len(noise_target_list))]
   
    
                
# Combine populations into complete list
pos = pos + place_pos
pos = pos + grid_pos
pos = pos + border_pos
pos = pos + obs_pos
pos = pos + noise_pos

# Combine populations into a complete list
pos = pos + place_pos + grid_pos + border_pos + obs_pos + noise_pos

# Write the data from all the cells to a file
with open(out_all_cells, "w") as f:
    json.dump(pos, f)

# Extract positions for plotting
place_posx, place_posy = [p[0] for p in place_pos], [p[1] for p in place_pos]
grid_posx, grid_posy = [g[0] for g in grid_pos], [g[1] for g in grid_pos]
border_posx, border_posy = [b[0] for b in border_pos], [b[1] for b in border_pos]
obs_posx, obs_posy = [o[0] for o in obs_pos], [o[1] for o in obs_pos]

# Visualize all the cells in a plot
fig, ax = plt.subplots()
ax.scatter(place_posx, place_posy, label=f'{len(place_pos)} Place Cells', marker='.')
ax.scatter(grid_posx, grid_posy, label=f'{len(grid_pos)} Grid Cells', marker='*')
ax.scatter(border_posx, border_posy, label=f'{len(border_pos)} Border Cells', marker='o', color='m')
ax.scatter(obs_posx, obs_posy, label=f'{len(obs_pos)} Obstacle-Border Cells', marker='s', color='r')
ax.scatter(sim_dict['trial_params']['start_x'], sim_dict['trial_params']['start_y'], label='Starting position(s)', color='g', marker='o')
ax.scatter(1.4, 1.4, label='Noise generator', color='cyan', marker='x')

add_goal_zone(ax, os.path.join('.', 'parameter_sets/current_parameter/trials_params.dat'))

ax.set_title('Location of cells')
ax.set_aspect(1)

ax.hlines(outer_ylim[0], outer_xlim[0], outer_xlim[1], color='grey', label='environment boundary')
ax.hlines(outer_ylim[1], outer_xlim[0], outer_xlim[1], color='grey')
ax.vlines(outer_xlim[0], outer_ylim[0], outer_ylim[1], color='grey')
ax.vlines(outer_xlim[1], outer_ylim[0], outer_ylim[1], color='grey')

if env_type == 'tmaze':
    gaw = env_params['goal_arm_width']
    cw = env_params['corridor_width']
    ax.hlines(ymax_pos - gaw, xmin_pos, -cw / 2)
    ax.hlines(ymax_pos - gaw, cw / 2, xmax_pos)
    ax.vlines(-cw / 2, ymin_pos, ymax_pos - gaw)
    ax.vlines(cw / 2, ymin_pos, ymax_pos - gaw)

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.savefig(fig_path)
