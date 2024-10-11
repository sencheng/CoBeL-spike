import os
import json
import argparse
import numpy as np
from analysis import SpikefileNest as spk
from analysis import PositionFile as pos
from analysis import Weight as W
from analysis import MultipleAnalysis as MA
from analysis import BehavioralPerformance as BP
from analysis import RepresentationVisualization as RV


"""
Adding a new analysis case:
    1. Add the analysis case to simulator/parameter_sets/current_parameters/analysis_config.json and simulator/parameter_sets/originial_params/original_analysis_config.json
    2. Make sure that the analysis case has a flag 'active' to (de)activate the case (see other analysis case for reference)
    3. The parameters of the analysis case should match to the parameters of the called function (e.g. no typos)
    4. If other parameters are needed, remove them before passing the to the function error_block (see config['ANIMATION']["active"] in this file for reference)
    5. If the parameters match with the parameters needed for the function, the function and the dictionay with the parameters can be given to error_block (removes 'active' flag)
"""


def print_report(vis_name, error_type, error=None):
    YELLOW = '\033[33m'
    RESET = '\033[0m'
    if error_type == "done":
        print(f"\nThe visualization of the {vis_name} is done without error!\n")
    elif error_type == "error":
        print(YELLOW)
        print(f"\nThe visualization of the {vis_name} is not possible due to an error!\n")
        if error is not None:
            print(f"{type(error)} : {error}")
        print()
        print(RESET)
    elif error_type == "invalid":
        print(YELLOW)
        print(f"\nThe visualization of the {vis_name} is not possible (only one cell population was found in current simulation)\n")
        print(RESET)
    elif error_type == "not demanded!":
        print(f"\nThe visualization of the {vis_name} is not demanded!\n")


def error_block(analysis_case, f, param_dict):
    try:
        param_dict.pop('active')
    except KeyError:
        pass

    try:
        out = f(**param_dict)
        if analysis_case is not None:
            print_report(analysis_case, "done")
        return out
    except ValueError as e:
        if analysis_case is None:
            print_report(f.__name__, "error", e)
        else:
            print_report(analysis_case, "error", e)
        return None


def create_pos_obj(sim_config, fig_p, config):
    if (
        config["TRAJECTORY"]["active"] or
        config["TRAJECTORY_SUBPLOT1"]["active"] or
        config["TRAJECTORY_SUBPLOT2"]["active"] or
        config["TRAJECTORY_GIF"]["active"] or
        config["TRAJECTORY_3D"]["active"] or
        config['AVR_FR']["active"] or
        config['OCCUPANCY']['active'] or
        config['INTERGOAL_OCC']['active'] or
        config['DTW']['active'] or
        config['DTW_OPT']['active'] or
        config['DTW_ALL']['active']
    ):
        print("Create pos obj")
        pos_obj = pos(
            data_path=sim_config['data_path'],
            filename='locs_time.dat',
            fig_path=fig_p
        )
        pos_obj.read_pos_file()
        return pos_obj
    return None


def create_weight_obj(sim_config, fig_p, config):
    if (
        config['VEC_FIELD']["active"] or
        config['VEC_FIELD_STACK']['active'] or
        config['VEC_FIELD_COMPLETE']['active'] or
        config['WEIGHT_CHANGE']['active'] or
        config['WEIGHT_CHANGE_ROTATED']['active']
    ):
        print("Create weight obj")
        w_obj = W(
            data_path=sim_config['data_path'],
            cell_type=config['w_obj']['cell_type'],
            fig_path=fig_p,
            times=np.array([0.0]),
            quiet=True
        )
        w_obj.read_files(
            cell_type=config['w_obj']['cell_type'],
            quiet=True
        )
        return w_obj
    return None


def create_beh_obj(sim_config, fig_p, config):
    if config['PERFORMANCE']["active"]:
        beh_obj = BP(
            {
                'nest_data_path': sim_config['data_path'],
                'param_path': sim_config['data_path']
            },
            {
                'tr_time_pos': 'locs_time.dat',
                'param_file': {
                    'sim': 'sim_params.json',
                    'net': 'network_params_spikingnet.json',
                    'env': 'env_params.json'
                }
            },
            fig_path=fig_p,
            reset_summary_file=True,
            perf_params=config['beh_obj']['perf_params']
        )
        beh_obj.get_performance()
        return beh_obj
    return None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, help='Path to the data directory from simulator')
    parser.add_argument('--num', type=int, help='master_rng_seed in sim_params.json')

    args = parser.parse_args()

    if (args.path) and (args.num is not None):
        sim_config = {
            'data_path': args.path,
            'master_rng_seed': args.num
        }
    else:
        with open('../simulator/parameter_sets/current_parameter/sim_params.json', 'r') as fl:
            sim_config = json.load(fl)
    
    with open('../simulator/parameter_sets/current_parameter/analysis_config.json', 'r') as f:
        config = json.load(f)

    fig_p = os.path.join(
        *sim_config['data_path'].split('/')[0:-1],
        'fig-{}/'.format(sim_config['master_rng_seed'])
    )
    pos_obj = create_pos_obj(sim_config, fig_p, config)
    w_obj = create_weight_obj(sim_config, fig_p, config)
    beh_obj = create_beh_obj(sim_config, fig_p, config)

    print(f"fig_path is: {fig_p}")

    if config["TRAJECTORY"]["active"]:
        error_block("TRAJECTORY", pos_obj.plot_rat_pos, config["TRAJECTORY"])

    if config["TRAJECTORY_SUBPLOT1"]["active"]:
        error_block("TRAJECTORY_SUBPLOT1", pos_obj.subplot_rat_pos1, config["TRAJECTORY_SUBPLOT1"])

    if config["TRAJECTORY_SUBPLOT2"]["active"]:
        error_block("TRAJECTORY_SUBPLOT2", pos_obj.subplot_rat_pos2, config["TRAJECTORY_SUBPLOT2"])
    
    if config["TRAJECTORY_GIF"]["active"]:
        error_block("TRAJECTORY_GIF", pos_obj.animate_rat_pos, config["TRAJECTORY_GIF"])

    if config["TRAJECTORY_3D"]["active"]:
        error_block("TRAJECTORY_3D", pos_obj.plot_rat_pos_3d, config["TRAJECTORY_3D"])
    
    if config['FIRINGMAP']["active"]:
        if 'data_path' not in sim_config:
            raise ValueError("Data path not found in sim_dict.")
        
        params_dict = {
            "data_path": sim_config['data_path'],
            "flname": 'grid_pos.json',
            "fig_path": fig_p,
            "cal_covarage": config['FIRINGMAP']['cal_coverage'],
            "firing_rate_quota": config['FIRINGMAP']['firing_rate_quota'],
            "param_file": {
                'sim': 'sim_params.json',
                'net': 'network_params_spikingnet.json',
                'env': 'env_params.json'
            },
            "resolution": config['FIRINGMAP']['resolution']
        } 
        
        repres_vis = error_block(None, RV, params_dict)

        config["FIRINGMAP"].pop("cal_coverage")
        config["FIRINGMAP"].pop("firing_rate_quota")
        config["FIRINGMAP"].pop("resolution")

        error_block('FIRINGMAP', repres_vis.plot_firing_maps, config['FIRINGMAP'])

    if config['AVR_FR']["active"]:
        config['AVR_FR']["tr_times"] = pos_obj.get_times_from_pos_file() 
        pops = config['AVR_FR'].pop('pop')

        for pop in pops:
            filename = f"{pop.lower()}-0.gdf"
            spk_obj = error_block(
                None,
                spk,
                {
                    "data_path": sim_config['data_path'],
                    "filename": filename,
                    "fig_path": fig_p
                }
            )
            config['AVR_FR']['fig_filename'] = pop
            error_block('AVR_FR', spk_obj.plot_avg_fr, config['AVR_FR'])            
    
    if config['ANIMATION']["active"]:
        animation = error_block(
            None,
            MA,
            {
                "path": sim_config['data_path'],
                "flname_dict": config['ANIMATION']["flname"],
                "fig_path": fig_p
            }
        )
        config['ANIMATION'].pop("flname")
        error_block('ANIMATION', animation.animate_tr_by_tr_loc, config['ANIMATION'])

    if config['VEC_FIELD']["active"]:
        pops = config['VEC_FIELD']['cell_type']
        for pop in pops:
            config['VEC_FIELD']['cell_type'] = pop
            error_block('VEC_FIELD', w_obj.vector_field_batch, config['VEC_FIELD'])

    if config['OCCUPANCY']['active']:
        error_block('OCCUPANCY', pos_obj.plot_occupancy, config['OCCUPANCY'])

    if config['INTERGOAL_OCC']['active']:
        error_block('INTERGOAL_OCC', pos_obj.plot_intergoal_occupancy, config['INTERGOAL_OCC'])

    if config['VEC_FIELD_STACK']['active']:
        pops = config['VEC_FIELD_STACK']['cell_type']
        for pop in pops:
            config['VEC_FIELD_STACK']['cell_type'] = pop
            error_block('VEC_FIELD_STACK', w_obj.plot_vector_field_stack, config['VEC_FIELD_STACK'])

    # TODO: check, not working
    if config['VEC_FIELD_COMPLETE']['active']:
        error_block('VEC_FIELD_COMPLETE', w_obj.plot_vector_field_all, config['VEC_FIELD_COMPLETE'])

    if config['WEIGHT_CHANGE']['active']:
        pops = config['WEIGHT_CHANGE']['pop']
        for pop in pops:
            config['WEIGHT_CHANGE']['pop'] = pop
            error_block('WEIGHT_CHANGE', w_obj.delta_w_helper, config['WEIGHT_CHANGE'])
    
    if config['WEIGHT_CHANGE_ROTATED']['active']:
        pops = config['WEIGHT_CHANGE_ROTATED']['pop']
        for pop in pops:
            config['WEIGHT_CHANGE_ROTATED']['pop'] = pop
            error_block('WEIGHT_CHANGE_ROTATED', w_obj.delta_w_helper, config['WEIGHT_CHANGE_ROTATED'])

    if config['DTW']['active']:
        error_block('DTW', pos_obj.calc_DTW, config['DTW'])

    if config['DTW_OPT']['active']:
        error_block('DTW_OPT', pos_obj.calc_DTW_optimal, config['DTW_OPT'])
    
    if config['DTW_ALL']['active']:
        error_block('DTW_ALL', pos_obj.calc_DTW_all, config['DTW_ALL'])

    if config['PERFORMANCE']["active"]:
        error_block('PERFORMANCE',  beh_obj.plot_performance, config['PERFORMANCE'])


if __name__ == "__main__":
    main()