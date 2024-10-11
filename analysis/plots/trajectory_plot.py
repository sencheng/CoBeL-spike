import sys
sys.path.append("..")

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from analysis import PositionFile as pos
from misc import set_params_plots, return_trajectory_plot_params, get_formats, get_figure_size_multiplier



def createTitle(net_config, base_net_config):
    a_minus = net_config['place']['syn_params']['A_minus']
    sigma_pc = net_config['place']['cells_prop']['p_row_sigma']
    N_pc = net_config['place']['cells_prop']['p_nrows']
    omega_nc = net_config['noise']['cells_prop']['max_fr'] if 'noise' in net_config else 0

    base_a_minus = base_net_config['place']['syn_params']['A_minus']
    base_sigma_pc = base_net_config['place']['cells_prop']['p_row_sigma']
    base_N_pc = base_net_config['place']['cells_prop']['p_nrows']
    base_omega_nc = base_net_config['noise']['cells_prop']['max_fr'] if 'noise' in base_net_config else 0

    check_numbers = lambda num1, num2: (num1 < 0 and num2 < 0) or (num1 > 0 and num2 > 0)

    if a_minus < 0 and check_numbers(a_minus, base_a_minus):
        a_minus_string = "Symmetric STDP"
    elif a_minus > 0 and check_numbers(a_minus, base_a_minus):
        a_minus_string = "Asymmetric STDP"
    elif a_minus < 0 and not check_numbers(a_minus, base_a_minus):
        a_minus_string = f"$\\bf{{Symmetric}}$" + " " + f"$\\bf{{STDP}}$"
    elif a_minus > 0 and not check_numbers(a_minus, base_a_minus):
        a_minus_string = f"$\\bf{{Asymmetric}}$" + " " + f"$\\bf{{STDP}}$"

    sigma_pc_string = f'$\\mathbf{{\sigma_{{PC}} = {sigma_pc}~m}}$' if sigma_pc != base_sigma_pc else f'$\sigma_{{PC}} = {sigma_pc}~m$'
    N_pc_string = f'$\\mathbf{{ N_{{PC}} = {N_pc}^2}}$' if N_pc != base_N_pc else  f'$N_{{PC}} = {N_pc}^2$'
    omega_nc_string = f'$\\mathbf{{ \Omega_{{NC}}={omega_nc}}}$' if omega_nc != base_omega_nc else f'$\Omega_{{NC}}={omega_nc}$'

    return a_minus_string + '\n' + sigma_pc_string + ', ' + N_pc_string + ', ' + omega_nc_string


def addABA(ax, j):
    """Currently only works with plots of the size [3, 3]
    """
    if j==0:
        ax.set_ylabel('A', rotation=0, color="red", fontsize=25)
    elif j==3:
        ax.set_ylabel('B', rotation=0, color="blue", fontsize=25)
    elif j==6:
        ax.set_ylabel('A', rotation=0, color="green", fontsize=25)


add_simulation_letter = True
add_ABA = True # Currently only works with plots of the size [3, 3]
shape = [2, 2]
plots = [
    {
        "data_path": "../../data/test/agent25/",
        "fig_path": "../../data/test/fig-25",
        "subplot_ids": [[1, 5, 10],
                        [11, 15, 20],
                        [21, 25, 30]]
    },
    {
        "data_path": "../../data/test/agent30/",
        "fig_path": "../../data/test/fig-30",
        "subplot_ids": [[1, 5, 10],
                        [11, 15, 20],
                        [21, 25, 30]]
    },
    {
        "data_path": "../../data/test/agent34/",
        "fig_path": "../../data/test/fig-34",
        "subplot_ids": [[1, 5, 10],
                        [11, 15, 20],
                        [21, 25, 30]]
    },
    {
        "data_path": "../../data/test/agent38/",
        "fig_path": "../../data/test/fig-38",
        "subplot_ids": [[1, 5, 10],
                        [11, 15, 20],
                        [21, 25, 30]]
    }
]

set_params_plots()
linewidth, markersize, trial_titlesize, titlesize, padding_title, legendsize, label_fontsize = return_trajectory_plot_params()

fig = plt.figure(figsize=(shape[0]*3*get_figure_size_multiplier(), shape[1]*3*get_figure_size_multiplier()))
outer = gridspec.GridSpec(*shape, wspace=0.3, hspace=0.3)

for i in range(np.prod(shape)):
    sim_path = os.path.join(plots[i]['data_path'], "sim_params.json")
    with open(sim_path, 'r') as fl:
        sim_config = json.load(fl)

    net_path = os.path.join(plots[i]['data_path'], "network_params_spikingnet.json")
    with open(net_path, 'r') as fl:
        net_config = json.load(fl)
    
    base_net_path = os.path.join(plots[0]['data_path'], "network_params_spikingnet.json")
    with open(base_net_path) as fl:
        base_net_config = json.load(fl)
    
    
    title = createTitle(net_config, base_net_config)
    pos_obj = pos(data_path=plots[i]['data_path'], filename='locs_time.dat', fig_path=plots[i]['fig_path'])
    pos_obj.read_pos_file()
    pos_obj.set_xy_lims()
    
    ax_outer = fig.add_subplot(outer[i])
    ax_outer.set_title(title, fontsize=titlesize, pad=padding_title)
    ax_outer.axis('off')

    subplot_ids = np.array(plots[i]["subplot_ids"])
    inner = gridspec.GridSpecFromSubplotSpec(*subplot_ids.shape, subplot_spec=outer[i], wspace=0.25, hspace=0.25)

    for j, tr in enumerate(subplot_ids.flatten()):
        ax = plt.Subplot(fig, inner[j])
        
        loc = None
        if j == len(subplot_ids.flatten()) - 1:
            loc = "best"
        
        if add_ABA:
            addABA(ax, j)
        
        if j == 0 and add_simulation_letter:
            ax.text(-0.09, 1.6, chr(i + 65), horizontalalignment='left', verticalalignment='top', fontsize=label_fontsize, transform=ax.transAxes, weight='bold')
        
        pos_obj._create_trajectory(
            tr=tr,
            ax=ax,
            title=f"Trial: {tr}",
            show_obstacle=True,
            legend_loc=loc,
            linewidth=linewidth,
            markersize=markersize,
            titlesize=trial_titlesize,
            legendsize=legendsize
        )
        fig.add_subplot(ax)

for frm in get_formats():
    fig.savefig(f"../../data/trajectories_different_trials.{frm}")
    