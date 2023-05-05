"""
This file analyses and visualizes the raw results of the simulation
correct one!
"""
import os
import json
import numpy as np
from analysis import SpikefileNest as spk
from analysis import PositionFile as pos
from analysis import ActionVector as act
from analysis import Weight as W

from analysis import MultipleAnalysis as MA
from analysis import BehavioralPerformance as BP
from analysis import RepresentationVisualization as RV


PLOT_FIRINGMAP = False
PLOT_RATPOS = True
PLOT_PERFORMANCE = False
ANIMATION = False
BORDER = False
PLOT_AVR_FR_PLACE = False
PLOT_AVR_FR_ACTION = False
PLOT_VEC_FIELD_PLACE = False
PLOT_VEC_FIELD = True
PLOT_VEC_FIELD_STACK_PLACE = False
PLOT_VEC_FIELD_GRID = False
PLOT_VEC_FIELD_STACK_GRID = False
PLOT_VEC_FIELD_BORDER = True
PLOT_VEC_FIELD_ALL = False


# System call
os.system("")


# Class of different styles
class style():
    BLACK = '\033[30#    python ../analysis_border/analysis_test.py    m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'


with open('../openfield/sim_params.json', 'r') as fl:
    net_dict = json.load(fl)

seed = net_dict['master_rng_seed']
fig_p = os.path.join(*net_dict['data_path'].split('/')[0:-1], 'fig-{}/'.format(seed))

pos_obj = pos(data_path=net_dict['data_path'], filename='locs_time.dat', fig_path=fig_p,
                      sim_file='sim_params.json')
tr_times = pos_obj.get_times_from_pos_file()


# if seed == 1:
#     PLOT_FIRINGMAP = True


def print_report(vis_name, error_type):
    """
    Prints the status of the visualisation
    """
    if error_type == "done":
        print("\nThe visualisation of the {} is done without error!\n".format(vis_name))
    elif error_type == "error":
        print(style.YELLOW)
        print("\nThe visualisation of the {} is not possible due to an error!\n".format(vis_name))
        print(style.RESET)
    elif error_type == "not demanded!":
        print("\nThe visualisation of the {} is not demanded!\n".format(vis_name))


VIS_NAME = "'firingmap(s)'"
if PLOT_FIRINGMAP:
    try:
        repres_vis = RV(os.path.join(*net_dict['data_path'].split('/')[0:-1]), flname='grid_pos.json', fig_path=fig_p,
                        cal_covarage=True, firing_rate_quota=0.75,
                        param_file={'sim': 'sim_params.json', 'net': 'network_params_spikingnet.json'}, resolution=50)
        repres_vis.plot_firing_maps(frm='png', plot_inds=True, show_obs=True)
        print_report(VIS_NAME, "done")
    except ValueError:
        print_report(VIS_NAME, "error")
else:
    print_report(VIS_NAME, "not demanded!")

VIS_NAME = "'trajectories'"
if PLOT_RATPOS:
    try:
        pos_obj.read_pos_file()
        pos_obj.plot_rat_pos(formats=['png'], title=False, legend_loc='out')
        print("Trajectories are plotted!")
    except ValueError:
        print_report(VIS_NAME, "error")
else:
    print_report(VIS_NAME, "not demanded!")

VIS_NAME = "'performance'"
if PLOT_PERFORMANCE:
    try:
        beh_obj = BP({'nest_data_path': net_dict['data_path'],
                      'param_path': os.path.join(*net_dict['data_path'].split('/')[0:-1])},
                     {'tr_time_pos': 'locs_time.dat',
                      'param_file': {'sim': 'sim_params.json', 'net': 'network_params_spikingnet.json'}},
                     fig_path=fig_p, reset_summary_file=False)#####,
                     ####what to plot = ['tr_dur','traj_len'])
        beh_obj.get_performance()
        beh_obj.plot_performance()
        print_report(VIS_NAME, "done")
    except ValueError:
        print_report(VIS_NAME, "error")
else:
    print_report(VIS_NAME, "not demanded!")

VIS_NAME = "'animation'"
if ANIMATION:
    #try:
        animation = MA( path_dict={'nest_data_path': net_dict['data_path'], 
                                  'other_data_path': net_dict['data_path']},
                        flname_dict=[['loc', 'action', 'place']
                                    ,['weight', 'grid', 'border']],
                        fig_path=fig_p, dpi=200)
        animation.animate_tr_by_tr_loc(frame_dur=20, summary=False)
        #animation.plot_rat_pos_tr_by_tr_loc()
        print_report(VIS_NAME, "done")
#    except ValueError:
#        print_report(VIS_NAME, "error")
else:
    print_report(VIS_NAME, "not demanded!")

VIS_NAME = "'border'"
if BORDER:
    try:
        spk_obj = spk(data_path=net_dict['data_path'], filename='border-0.gdf', fig_path=fig_p)
        spk_obj.plot_avg_fr(fig_filename='border', tr_times=tr_times, frm='png')
        print_report(VIS_NAME, "done")
    except ValueError:
        print_report(VIS_NAME, "error")

VIS_NAME = "'average firing rate of place/grid cells'"
if PLOT_AVR_FR_PLACE:
    try:
        spk_obj = spk(data_path=net_dict['data_path'], filename='place-0.gdf', fig_path=fig_p)
        spk_obj.plot_avg_fr(fig_filename='place', tr_times=tr_times, frm='png')
        print_report(VIS_NAME, "done")
    except ValueError:
        print_report(VIS_NAME, "error")
else:
    print_report(VIS_NAME, "not demanded!")

VIS_NAME = "'average firing rate of action neurons'"
if PLOT_AVR_FR_ACTION:
    try:
        spk_obj = spk(data_path=net_dict['data_path'], filename='action-0.gdf', fig_path=fig_p)
        spk_obj.plot_avg_fr(fig_filename='action',tr_times=tr_times, frm='png')
        print_report(VIS_NAME, "done")
    except ValueError:
        print_report(VIS_NAME, "error")
else:
    print_report(VIS_NAME, "not demanded!")

VIS_NAME = "'vector field (place-single)'"
if PLOT_VEC_FIELD_PLACE:
#    try:
        w_obj = W(data_path=net_dict['data_path'], filename='place-0.csv', fig_path=fig_p, times=np.array([0.0]))
        w_obj.read_files()
        w_obj.plot_vector_field_placecells(frm='png')
        print_report(VIS_NAME, "done")
#    except ValueError:
#        print_report(VIS_NAME, "error")
else:
    print_report(VIS_NAME, "not demanded!")

VIS_NAME = "'vector field (place-stack)'"
if PLOT_VEC_FIELD_STACK_PLACE:
#    try:
        w_obj = W(data_path=net_dict['data_path'], filename='place-0.csv', fig_path=fig_p, times=np.array([0.0]))
        w_obj.read_files()
        w_obj.plot_vector_field_stack(frm='png', cell_type='place', plot_ind=False)
        print_report(VIS_NAME, "done")
#    except ValueError:
#        print_report(VIS_NAME, "error")
else:
    print_report(VIS_NAME, "not demanded!")


VIS_NAME = "'vector field (grid-single)'" 
if PLOT_VEC_FIELD_GRID:
    try:
        w_obj = W(data_path=net_dict['data_path'], filename='grid-0.csv', fig_path=fig_p, times=np.array([0.0]))
        w_obj.read_files()
        w_obj.plot_vector_field_gridcells(frm='png')
        print_report(VIS_NAME, "done")
    except ValueError:
        print_report(VIS_NAME, "error")
else:
    print_report(VIS_NAME, "not demanded!")

VIS_NAME = "'vector field (grid-stack)'"
if PLOT_VEC_FIELD_STACK_GRID:
    try:
        w_obj = W(data_path=net_dict['data_path'], filename='grid-0.csv', fig_path=fig_p, times=np.array([0.0]))
        w_obj.read_files()
        w_obj.plot_vector_field_stack(frm='png', cell_type='grid', plot_ind=False)
        print_report(VIS_NAME, "done")
    except ValueError:
        print_report(VIS_NAME, "error")
else:
    print_report(VIS_NAME, "not demanded!")
    
VIS_NAME = "'vector field (border)'"
if PLOT_VEC_FIELD_BORDER:
    try:
        w_obj = W(data_path=net_dict['data_path'], filename='border-0.csv', fig_path=fig_p, times=np.array([0.0]))
#        w_obj.plot_vector_field_bordercells(frm='png', show_obs=True)
        w_obj.plot_vector_field_stack(frm='png', cell_type='border', plot_ind=True)
        print_report(VIS_NAME, "done")
    except ValueError:
        print_report(VIS_NAME, "error")
else:
    print_report(VIS_NAME, "not demanded!")

VIS_NAME = "'vector field (all)'"
if PLOT_VEC_FIELD_ALL:
    try:
        w_obj = W(data_path=net_dict['data_path'], filename='place-0.csv', fig_path=fig_p, times=np.array([0.0]))
        w_obj.read_files()
        w_obj.plot_vector_field_stack(frm='png', plot_ind=False)
        print_report(VIS_NAME, "done")
    except ValueError:
        print_report(VIS_NAME, "error")
else:
    print_report(VIS_NAME, "not demanded!")


# os.system("rm -rf "+net_dict['data_path'])
print("DONE!!!!")
