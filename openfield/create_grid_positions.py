"""
this file makes the grid_pos.json file
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt

in_network_params = "network_params_spikingnet.json"
in_sim_params = "sim_params.json"
out_all_cells = "grid_pos.json"

# Read the parameter files
with open(in_network_params, "r") as f:
    net_dict = json.load(f)
with open(in_sim_params, "r") as f:
    sim_dict = json.load(f)

master_rng_seed = str(sim_dict['master_rng_seed'])
data_dir = sim_dict['data_path']
main_data_dir = os.path.join(*data_dir.split('/')[0:-1])
fig_dir = os.path.join(main_data_dir, f'fig-{master_rng_seed}')
fig_path = f'{fig_dir}/grid_poss.png'

openfield_env = sim_dict['environment']['openfield']
sim_obs_dict = sim_dict['environment']['obstacles']

# Boundaries for the openfield enviornment
xmin_pos, xmax_pos = openfield_env['xmin_position'], openfield_env['xmax_position']
ymin_pos, ymax_pos = openfield_env['ymin_position'], openfield_env['ymax_position']
openfield_xlim = np.array([xmin_pos, xmax_pos])
openfield_ylim = np.array([ymin_pos, ymax_pos])

# References to parameters for the different cell types
#place, grid, border = net_dict['place'], net_dict['grid'], net_dict['border']
try:
    place = net_dict['place']
except KeyError:
    place = {'num_neurons':0}
try:
    grid = net_dict['grid']
except KeyError:
    grid = {'num_neurons':0}
try:
    border = net_dict['border']
except KeyError:
    border = {'num_neurons':0}
try:
    obstacle = net_dict['obstacle']
except KeyError:
    obstacle = {'num_neurons':0}

# Lists to store all the different types of cells
pos, place_pos, grid_pos, border_pos, obs_pos = [], [], [], [], []

# If a grid point is outside the enviornment let its position wrap around
# essentially making the enviornment into a torus
#     ((x - xmin_pos) mod x_range)  gives a value in Z_n, n being the length/ width of the enviornment
#     this value is then shifted by the lower boundary to adapt the point to the boundaries of
#     the enviornment. This is done for x and y.
def grid_in_env(dummy: list) -> list:
    x_range, y_range = openfield_xlim[1] - openfield_xlim[0], openfield_ylim[1] - openfield_ylim[0]
    dummy[0] = ((dummy[0] - openfield_xlim[0]) % x_range) + openfield_xlim[0]
    dummy[1] = ((dummy[1] - openfield_ylim[0]) % y_range) + openfield_ylim[0]


# Check that the amount of place-cells is greater than 0
if place["num_neurons"]:
    p_neurons_x, p_neurons_y = place['cells_prop']['p_nrows'], place['cells_prop']['p_ncols']
    p_sig_x, p_sig_y = place['cells_prop']['p_row_sigma'], place['cells_prop']['p_col_sigma']
    p_rep_index = place['cells_prop']['rep_index']
    p_fr = place['cells_prop']['max_fr']
    p_width, p_height = place['spatial_prop']['width'], place['spatial_prop']['height']

    # create an evenly spaced amount of place cells over the entire enviornment
    for x in np.linspace(- p_width / 2, p_width / 2, p_neurons_x):
        for y in np.linspace(p_height / 2, - p_height / 2, p_neurons_y):
            dummy = [float(x), float(y), float(p_sig_x), float(p_sig_y), p_rep_index, p_fr]
            #pos.append(dummy)       # add to list of all cells
            place_pos.append(dummy) # add to list of place cells

# Check that the amount of grid cells is greater than 0
if grid["num_neurons"]:
    # data for the grids is given in lists, each position represents a grid
    g_nrows, g_ncols = grid['cells_prop']['g_nrows'], grid['cells_prop']['g_ncols']
    g_kappa = grid['cells_prop']['g_kappa']
    g_lambda = grid['cells_prop']['g_lambda']
    g_rep_index = grid['cells_prop']['rep_index']
    g_fr = grid['cells_prop']['max_fr']
    
    # generate a list of random coordinates inside the enviornment, these will be the start position for
    # each grid
    x_range, y_range = openfield_xlim[1] - openfield_xlim[0], openfield_ylim[1] - openfield_ylim[0]
    n_grids = len(g_nrows)
    grid_start = zip(np.random.rand(n_grids) * x_range - openfield_xlim[0], np.random.rand(n_grids) * y_range - openfield_ylim[0])
    
    grid_start = zip(np.random.rand(n_grids) * x_range, np.random.rand(n_grids) * y_range)

    grid_start = zip(-1+0*np.random.rand(n_grids) * 0*x_range, -1+0*np.random.rand(n_grids) * y_range)

    # Iterate though the given parameters one grid at a time
    for start, row, col, lam, kap, fr in zip(grid_start, g_nrows, g_ncols, g_lambda, g_kappa, g_fr):
        # generate the points for that grid
        for RS, y in enumerate(np.linspace(start[1], start[1] + lam * np.cos(np.pi/6), row, endpoint=False), start=1):
            for x in np.linspace(start[0], start[0] + lam, col, endpoint=False):
                x_offset = (lam / 2) / row
                dummy = [x + (x_offset * RS), y, kap, lam, 1, fr]
                # if disorder is desired then add some randomness
                if 'disorder' in grid.keys():
                    dis = grid['disorder']
                    np.random.seed(sim_dict['master_rnd_seed'] + RS)
                    dummy[0] += np.random.unifrom(-dis, dis, 2)
                grid_in_env(dummy)
                #pos.append(dummy)       # add to list of all cells
                grid_pos.append(dummy)  # add to list of grid cells

        np.random.RandomState(sim_dict['master_rng_seed'])

# Check that the amount of border cells is greater then 0 (i.e. equal to 8)
if border["num_neurons"]:
    xmid_pos, ymid_pos = (xmin_pos + xmax_pos) / 2, (ymin_pos + ymax_pos) / 2
    b_flag = border['cells_prop']['flag']
    b_width = border['cells_prop']['width']
    b_rep_index = border['cells_prop']['rep_index']
    b_fr = border['cells_prop']['max_fr']
    
    # If the flag is set to True hence border cells are desired,
    # then create bordercells at the corners and the centers of the edges of 
    # the environment and add them to the list of border cells
    # By convention, border cells should be added in a !clock-wise! order starting from the top
    if b_flag:
        border_pos.append([xmid_pos, ymax_pos, ymax_pos - b_width, b_width, b_rep_index, b_fr])
        border_pos.append([xmax_pos, ymax_pos, b_width, b_width, b_rep_index, b_fr])
        border_pos.append([xmax_pos, ymid_pos, b_width, xmax_pos - b_width, b_rep_index, b_fr]) 
        border_pos.append([xmax_pos, ymin_pos, b_width, b_width, b_rep_index, b_fr])
        border_pos.append([xmid_pos, ymin_pos, ymax_pos - b_width, b_width, b_rep_index, b_fr])
        border_pos.append([xmin_pos, ymin_pos, b_width, b_width, b_rep_index, b_fr])
        border_pos.append([xmin_pos, ymid_pos, b_width, xmax_pos - b_width, b_rep_index, b_fr])
        border_pos.append([xmin_pos, ymax_pos, b_width, b_width, b_rep_index, b_fr])


    
if sim_obs_dict['flag']:
    ob_width = obstacle['cells_prop']['width'] # firing field width
    ob_rep_index = obstacle['cells_prop']['rep_index']
    ob_fr = obstacle['cells_prop']['max_fr']
    for center, vert, horiz in zip(sim_obs_dict["centers"], sim_obs_dict["vert_lengths"], sim_obs_dict["horiz_lengths"]):
        delta_y = vert / 2.   # Get the length and width 
        delta_x = horiz / 2.  # as distances from the center point
        x = float(center[0])
        y = float(center[1])
        # The order these are added is important! These must also be given
        # in !clock-wise! order starting from the top
        obs_pos.append([x, y + delta_y, delta_x, ob_width, ob_rep_index, ob_fr]) # top
        obs_pos.append([x + delta_x, y, ob_width, delta_y, ob_rep_index, ob_fr]) # right
        obs_pos.append([x, y - delta_y, delta_x, ob_width, ob_rep_index, ob_fr]) # bottom
        obs_pos.append([x - delta_x, y, ob_width, delta_y, ob_rep_index, ob_fr]) # left
        border_x = (x - delta_x, x + delta_x)
        border_y = (y - delta_y, y + delta_y)
        
        # This removes place cells which overlap with the obstacle
        # TODO: This breaks borders and obstacles! Need to find alternative way to remove these cells
#        del_list = []
#        for p in place_pos:
#            if x - delta_x < p[0] and p[0] < x + delta_x:
#                if y - delta_y < p[1] and p[1] < y + delta_y:
#                    print('REMOVING CELL {} AT COORDINATES: {}, {}'.format(place_pos.index(p), p[0], p[1]))
#                    del_list.append(p)
#        for d in del_list:
#            place_pos.remove(d)
        
# Combine populations into complete list
pos = pos + place_pos
pos = pos + grid_pos
pos = pos + border_pos
pos = pos + obs_pos

# write the data from all the cells to a file
with open(out_all_cells, "w") as f:
    json.dump(pos, f)

# x and y positions of the place cells
place_posx, place_posy = [p[0] for p in place_pos], [p[1] for p in place_pos]

# x and y positions of the grid cells
grid_posx, grid_posy = [g[0] for g in grid_pos], [g[1] for g in grid_pos]

# x and y positions of the border cells
border_posx, border_posy = [b[0] for b in border_pos], [b[1] for b in border_pos]

# x and y positions of the border cells
obs_posx, obs_posy = [o[0] for o in obs_pos], [o[1] for o in obs_pos]

# visualize all the cells in a plot
plt.figure(figsize=(18, 12), dpi=100)
plt.scatter(place_posx, place_posy, label=f'{len(place_pos)} Place Cells', marker='.')
plt.scatter(grid_posx, grid_posy, label=f'{len(grid_pos)} Grid Cells', marker='*')
plt.scatter(border_posx, border_posy, label=f'{len(border_pos)} Border Cells', marker='o')
plt.scatter(obs_posx, obs_posy, label=f'{len(obs_pos)} Obstacle-Border Cells', marker='s')
plt.title('Location of cells')

if sim_dict['sim_env'] == 'openfield':
    plt.hlines(openfield_ylim[0], openfield_xlim[0], openfield_xlim[1], color='grey', label='environment')
    plt.hlines(openfield_ylim[1], openfield_xlim[0], openfield_xlim[1], color='grey')
    plt.vlines(openfield_xlim[0], openfield_ylim[0], openfield_ylim[1], color='grey')
    plt.vlines(openfield_xlim[1], openfield_ylim[0], openfield_ylim[1], color='grey')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig(fig_path)
#plt.show()
#print(pos)
