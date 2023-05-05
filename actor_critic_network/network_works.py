#!/usr/bin/env python3
"""
Implementation of Brzonko et al. Elife (2017) using NEST.
The code structure is adapted from Microcircuit example
on NEST website authored by Hendrik Rothe, Hannah Bos,
Sacha van Albada; May 2016

Code author: Mohammad M. Nejad
"""

import nest
import nest.topology as topp
import numpy as np
import os
import pandas as pd
from copy import deepcopy
import json
import pprint as pr


class Network:
    """ Handles the setup of the network and
    provides functions to connect the network and devices.

    Arguments
    ---------
    sim_dict
        dictionary containing all parameters specific to the simulation
        such as the directory the data is stored in and the seeds
        (see: sim_params.json)
    net_dict
         dictionary containing all parameters specific to the neurons
         and the network (see: network_params.json)

    Keyword Arguments
    -----------------
    stim_dict (Doesn't apply here, should change accordingly!)
        dictionary containing all parameter specific to the stimulus
        (see: stimulus_params.py)
    """

    def __init__(self, sim_dict, net_dict):

        self.sim_dict = deepcopy(sim_dict)
        self.net_dict = deepcopy(net_dict)
        self.rem_zero_pop()
        self._create_dir_('data_path')
        # *:Iterating over all contents of the following expression (here is list).
        fig_path = os.path.join(*sim_dict['data_path'].split('/')[:-1], 'fig')
        self.sim_dict['fig_path'] = fig_path
        self._create_dir_('fig_path')
        self.sim_interval = [0]
        self.curr_pos = np.array([0., 0.])
        self.rat_pos = [self.curr_pos]
        self.rat_pos_all = [self.curr_pos]
        self.epsilon = 1e-4
        self._last_music_ch_id = 0

    def _istest_(self, flag=False):

        if flag:
            self.data_path = self.sim_dict['data_path'] + '_test'
            self.fig_path = self.sim_dict['fig_path'] + '_test'
        else:
            self.data_path = self.sim_dict['data_path']
            self.fig_path = self.sim_dict['fig_path']

    def _create_dir_(self, name):

        for l in ['', '_test']:
            if os.path.isdir(self.sim_dict[name] + l):
                print('Data path already exist!\n')
            else:
                os.makedirs(self.sim_dict[name] + l, exist_ok=True)
            print('Storing simualtion data in %s' % (self.sim_dict[name] + l))

    def _remove_existing_gdffiles_(self):

        done = os.system('rm ' + os.path.join(self.data_path, '*.gdf'))
        if done == 0:
            print('Existing files from previous simulations are deleted!')

    def setup_nest(self):

        """ Hands kernel parameters to the NEST-kernel.

        Resets the NEST-kernel and passes parameters to it.
        The number of seeds for the NEST-kernel is computed, based on the
        total number of MPI processes and threads of each.

        """
        nest.ResetKernel()
        master_seed = self.sim_dict['master_rng_seed']
        if nest.Rank() == 0:
            print('Master seed: %i ' % master_seed)
        nest.SetKernelStatus({'local_num_threads': self.sim_dict['local_num_threads']})
        N_tp = nest.GetKernelStatus(['total_num_virtual_procs'])[0]
        if nest.Rank() == 0:
            print('Number of total processes: %i' % N_tp)
        rng_seeds = list(range(master_seed + 1 + N_tp, master_seed + 1 + (2 * N_tp)))
        grng_seed = master_seed + N_tp
        if nest.Rank() == 0:
            print('Seeds for random number generators of virtual processes: %r' % rng_seeds)
            print('Global random number generator seed: %i' % grng_seed)
        self.pyrngs = [np.random.RandomState(s) for s in list(range(master_seed, master_seed + N_tp))]
        self.sim_resolution = self.sim_dict['sim_resolution']
        kernel_dict = {'resolution': self.sim_resolution, 'grng_seed': grng_seed, 'rng_seeds': rng_seeds,
                       'overwrite_files': self.sim_dict['overwrite_files'], 'print_time': self.sim_dict['print_time'], }
        nest.SetKernelStatus(kernel_dict)

        if not self.sim_dict['info_message']:
            nest.set_verbosity('M_INFO')

    def rem_zero_pop(self):
        zero_list = []
        for pop in self.net_dict.keys():
            if self.net_dict[pop]['num_neurons'] == 0:
                zero_list.append(pop)
        for pop in zero_list:
            del self.net_dict[pop]

    def cell_center_list(self, grid_file_path, pop):

        """ Reads properties of spatial selective neurons and store the 
        coordinate of the centers of the coding space and other properties.
        These properties are firing rate, type of cell and etc.
        """
        rep_index = self.net_dict[pop]["cells_prop"]["rep_index"]
        self.cell_centers = []
        self.cell_props = []
        with open(grid_file_path, 'r') as fl:
            dummy = json.load(fl)
        for i in range(len(dummy)):
            if dummy[i][4] == rep_index:
                self.cell_centers.append(dummy[i][0:2])
                self.cell_props.append(dummy[i][2:])

    def create_pop(self, pop):

        par_rnd_dic = self.net_dict[pop]['mdl_par_rand']

        if self.net_dict[pop]['arange_spatial']:
            topp_dic, pop_ids = self.create_topological_pop(pop)
        else:
            topp_dic = {}
            pop_ids = nest.Create(self.net_dict[pop]['model'], self.net_dict[pop]['num_neurons'],
                                  params=self.net_dict[pop]['model_params'])
        self.pop_dict[pop]['prop_dic'] = topp_dic
        self.pop_dict[pop]['ids'] = pop_ids

        if par_rnd_dic['flag']:
            par_def = nest.GetStatus([pop_ids[0]], par_rnd_dic['par'])[0]
            dev_from_def = par_rnd_dic['dev_from_def']

            par_rnd = self.pyrngs[0].uniform(par_def - dev_from_def, par_def + dev_from_def, size=len(pop_ids))
            nest.SetStatus(pop_ids, par_rnd_dic['par'], par_rnd)

        if '->' in self.net_dict[pop]['targets']:
            self.setup_neuromodulation(pop)

    def setup_neuromodulation(self, pop):

        pre_pop = self.net_dict[pop]['targets'].split('->')[0]
        orig_synmodel = deepcopy(self.net_dict[pre_pop]['trg_conn']['syn_spec']['model'])

        if 'dopamine' in orig_synmodel:
            vol_trans = nest.Create('volume_transmitter', 1)
            nest.Connect(self.pop_dict[pop]['ids'], vol_trans)
            new_synmode = orig_synmodel + '_with_vt'
            nest.CopyModel(orig_synmodel, new_synmode, {'vt': vol_trans[0]})
            nest.SetDefaults(new_synmode, self.net_dict[pre_pop]['syn_params'])
            self.net_dict[pre_pop]['trg_conn']['syn_spec']['model'] = new_synmode

    def assign_border_cells(self, topp_info, pop_ids):

        '''
        makes a list of the border based on cell IDs

        Parameters
        ----------
        topp_info : <class 'dict'>
            topological positions of cells in nest
        pop_ids : <class 'tuple'> 
            ID of all place, grid and border cells

        Returns
        -------
        None.

        '''
        self.border_id_list = []
        for i, cell in enumerate(self.cell_props):
            if cell[2] == 2:
                pass  # self.border_id_list.append(pop_ids[i])

    def create_topological_pop(self, pop):
        '''
        assigns topological coordinates to the cells of the populations

        Parameters
        ----------
        pop : class 'str'
            currently is place
            TODO: there should be three different populations

        Returns
        -------
        topp_info : <class 'dict'>
            topological positions of cells in nest
        pop_ids : <class 'tuple'> 
            ID of all place, grid and border cells

        '''

        self.cell_center_list('../openfield/grid_pos.json', pop)
        rep_index = self.net_dict[pop]["cells_prop"]["rep_index"]
        layer_dict = {'extent': [2.4 + self.epsilon / 2, 2.4 + self.epsilon / 2], 'positions': self.cell_centers,
                      'center': [0.0, 0.0], 'elements': self.net_dict[pop]['model']}
        layer = topp.CreateLayer(layer_dict)
        pop_ids = nest.GetNodes(layer)[0]

        locations = np.array(topp.GetPosition(pop_ids))
        topp_info = dict(layer_id=layer, positions=locations)

        self.xmin = locations[:, 0].min()
        self.xmax = locations[:, 0].max()

        self.ymin = locations[:, 1].min()
        self.ymax = locations[:, 1].max()

        self._write_id_pos_file(pop_ids, locations, pop)
        #        self.assign_border_cells(topp_info, pop_ids)

        return topp_info, pop_ids

    def _write_id_pos_file(self, ids, pos, pop):

        f_id = open(os.path.join(self.data_path, pop + 'ID_pos.dat'), 'w')
        f_id.write('id\tx\ty\tTauxLamBox\tTauyKapBoy\trep_type\tmax_fr\n')

        for idx, nid in enumerate(ids):
            f_id.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(nid, pos[idx, 0], pos[idx, 1], self.cell_props[idx][0],
                                                             self.cell_props[idx][1], self.cell_props[idx][2],
                                                             self.cell_props[idx][3]))

        f_id.close()

    def create_input(self, pop):

        if self.net_dict[pop]['ext_inputs']:

            inp = self.net_dict[pop]['ext_inputs']

            if self.net_dict[pop]['inp_conn']['conn_spec']['rule'] == 'one_to_one':
                inp_id = nest.Create(inp['model'], len(self.pop_dict[pop]['ids']), params=inp['model_params'])
            else:
                inp_id = nest.Create(inp['model'], 1, params=inp['model_params'])

            self.pop_dict[pop]['input_id'] = inp_id

            if 'music' in inp['model']:
                for i, n in enumerate(inp_id):
                    nest.SetStatus([n], {'music_channel': i})

    def update_placecells_fr(self):

        spatial_sig = 0.4
        avg_fr = self.net_dict['place']['spatial_prop']['max_fr']

        ids = self.pop_dict['place']['input_id']
        pos = self.pop_dict['place']['prop_dic']['positions']

        fr_vec = np.array(pos) - np.array(self.curr_pos).reshape(1, 2).repeat(len(ids), axis=0)
        fr_vec = avg_fr * np.exp(-(fr_vec ** 2).sum(axis=1) / (spatial_sig ** 2))

        for idx, val in enumerate(fr_vec):
            nest.SetStatus([ids[idx]], {'rate': val})

    def connect_inputs(self, pop):

        if self.net_dict[pop]['ext_inputs']:
            nest.Connect(self.pop_dict[pop]['input_id'], self.pop_dict[pop]['ids'], **self.net_dict[pop]['inp_conn'])

    def assign_action(self):

        a0 = self.net_dict['action']['orientation_sel_dic']['mov_step']
        N = len(self.pop_dict['action']['ids'])
        dir_angle = np.linspace(0, np.pi * 2 * (1 - 1 / N), N)
        action = np.vstack((np.sin(dir_angle), np.cos(dir_angle)))
        self.pop_dict['action']['prop_dic']['action_vec'] = action
        self.pop_dict['action']['prop_dic']['action_phase'] = dir_angle

        self._write_id_phase_to_file()

    def _write_id_phase_to_file(self):

        fl_id = open(os.path.join(self.data_path, 'actionID_dir.dat'), 'w')
        fl_id.write('id\tx\ty\n')

        action_dir = self.pop_dict['action']['prop_dic']['action_vec']

        for idx, n_id in enumerate(self.pop_dict['action']['ids']):
            fl_id.write('{}\t{}\t{}\n'.format(n_id, action_dir[0, idx], action_dir[1, idx]))

        fl_id.close()

    def connect_populations(self, pop):

        par_rnd_dic = self.net_dict[pop]['trg_conn_rnd']
        if par_rnd_dic['flag']:
            self.net_dict[pop]['trg_conn']['syn_spec']['weight'] = {'distribution': par_rnd_dic['dist'], 'mu': float(
                self.net_dict[pop]['trg_conn']['syn_spec']['weight']), 'sigma': par_rnd_dic['std']}

        trg = self.net_dict[pop]['targets']
        if isinstance(trg, list):
            for trgs in trg:
                print('Not implemented!')
                break
        elif trg and (not '->' in trg):

            nest.Connect(self.pop_dict[pop]['ids'], self.pop_dict[trg]['ids'], **self.net_dict[pop]['trg_conn'])

        if pop == "border" or pop == "action":
            nest.GetConnections(self.pop_dict[pop]["ids"])

    def connect_populations_fortest(self, pop, ac_id):

        par_rnd_dic = self.net_dict[pop]['trg_conn_rnd']
        if par_rnd_dic['flag']:
            self.net_dict[pop]['trg_conn']['syn_spec']['weight'] = {'distribution': par_rnd_dic['dist'], 'mu': float(
                self.net_dict[pop]['trg_conn']['syn_spec']['weight']), 'sigma': par_rnd_dic['std']}
        trg = self.net_dict[pop]['targets']

        if isinstance(trg, list):
            for trgs in trg:
                print('Not implemented!')
                break
        elif trg:
            if pop == 'place':
                nest.Connect(self.pop_dict[pop]['ids'], [self.pop_dict[trg]['ids'][ac_id]],
                             **self.net_dict[pop]['trg_conn'])
            else:
                nest.Connect(self.pop_dict[pop]['ids'], self.pop_dict[trg]['ids'], **self.net_dict[pop]['trg_conn'])

    def compute_act_pop_wmat(self, scale=1):

        print('Connection weights got stronger %f folds.' % scale)

        psi = self.net_dict['action']['orientation_sel_dic']['exp_coef']
        winh = self.net_dict['action']['orientation_sel_dic']['winh']
        wexc = self.net_dict['action']['orientation_sel_dic']['wexc']
        N = len(self.pop_dict['action']['ids'])

        dir_angle = self.pop_dict['action']['prop_dic']['action_phase'].reshape(-1, 1)
        w_phase_dep = np.exp(psi * np.cos(dir_angle.repeat(N, axis=1) - dir_angle.T.repeat(N, axis=0)))
        w_phase_dep[np.arange(N), np.arange(N)] = 0
        w_phase_dep = w_phase_dep / w_phase_dep.sum(axis=1)[0]
        self.w_mat_action = winh / N + wexc * w_phase_dep
        self.w_mat_action[np.arange(N), np.arange(N)] = 0
        self.w_mat_action = self.w_mat_action * scale

    def set_act_pop_wmat(self):

        act_gids = self.pop_dict['action']['ids']
        min_act_gids = min(act_gids)

        conns = nest.GetConnections(self.pop_dict['action']['ids'], self.pop_dict['action']['ids'])
        for c in conns:
            pre = c[0] - min_act_gids
            post = c[1] - min_act_gids
            nest.SetStatus([c], {'weight': self.w_mat_action[post, pre]})

    def remove_boundary_conn(self, src='border', trg='action'):

        self.rmv_bnd_conn_finder(src, trg, 'xmax', np.pi / 2)
        self.rmv_bnd_conn_finder(src, trg, 'xmin', 3 * np.pi / 2)
        self.rmv_bnd_conn_finder(src, trg, 'ymax', 0)
        self.rmv_bnd_conn_finder(src, trg, 'ymin', np.pi)
        
    def remove_obstacle_conn(self, src, trg):

        obs_dict = self.sim_dict['obstacles']
        for center, length, width in zip(obs_dict["centers"], obs_dict["lengths"], obs_dict["widths"]):
            xmin = center[0] - (width / 2)
            xmax = center[0] + (width / 2)
            ymin = center[1] - (length / 2)
            ymax = center[1] + (length / 2)
            self.remove_obs_conn_finder(src, trg, xmin, np.pi / 2, 0)
            self.remove_obs_conn_finder(src, trg, xmax, 3 * np.pi / 2, 0)
            self.remove_obs_conn_finder(src, trg, ymin, 0, 1)
            self.remove_obs_conn_finder(src, trg, ymax, np.pi, 1)
            
    def remove_obs_conn_finder(self, src, trg, pos, phase, axis):
        src_id = np.array(self.pop_dict[src]['ids'])
        sel_pos = np.abs(self.pop_dict[src]['prop_dic']['positions'][:, axis] - pos) < self.epsilon
        sel_src_id = src_id[sel_pos]
        
        if phase == 0:
            sel_dir = (np.abs(self.pop_dict[trg]['prop_dic']['action_phase'] - phase) < np.pi / 2) | (
                    np.abs(self.pop_dict[trg]['prop_dic']['action_phase'] - phase) > 3 * np.pi / 2)
            sel_trg_id = np.array(self.pop_dict[trg]['ids'])[sel_dir]
        else:
            sel_dir = np.abs(self.pop_dict[trg]['prop_dic']['action_phase'] - phase) < np.pi / 2
            sel_trg_id = np.array(self.pop_dict[trg]['ids'])[sel_dir]
            
        conn_tmp = nest.GetConnections(sel_src_id.tolist(), sel_trg_id.tolist())
        nest.SetStatus(conn_tmp, {'weight': 0.0})

    def rmv_bnd_conn_finder(self, src, trg, pos, phase):

        '''
        Finds connections according to the given criteria in order to project 
        feed forward weights at the boundary areas in place cells into the env.
        Inputs:
            scr: string 
                Keyword of the source region
            trg: string
                Keyword of the traget region
            pos: string
                Boundary
            phase: Direction to which that the connections should be weakened.
            This function should be rewritten based on dictionary structure
            rather than using attributes (xmax, xmin,...).
        '''
        src_id = np.array(self.pop_dict[src]['ids'])
        '''
        if pos[0] == 'x':
            if pos[-1] == 'x':
                sel_pos = np.abs(self.pop_dict[src]['prop_dic']['positions'][:, 0] - self.xmax) < self.epsilon
                sel_src_id = src_id[sel_pos]
            else:
                sel_pos = np.abs(self.pop_dict[src]['prop_dic']['positions'][:, 0] - self.xmin) < self.epsilon
                sel_src_id = src_id[sel_pos]
        else:
            if pos[-1] == 'x':
                sel_pos = np.abs(self.pop_dict[src]['prop_dic']['positions'][:, 1] - self.ymax) < self.epsilon
                sel_src_id = src_id[sel_pos]
            else:
                sel_pos = np.abs(self.pop_dict[src]['prop_dic']['positions'][:, 1] - self.ymin) < self.epsilon
                sel_src_id = src_id[sel_pos]
        '''
        #TODO: This dict is used to test connection removal by referencing IDs. It will need to be updated!       
        test_dict_border_ids = {'xmax':[888, 891, 892, 893],
                                'xmin':[885, 889, 890, 894],
                                'ymax':[886, 890, 892, 895],
                                'ymin':[887, 889, 891]}
        
        tmp = np.array(test_dict_border_ids[pos])
        sel_pos = np.isin(src_id,tmp)
        print('sel_pos = {}'.format(sel_pos))
        sel_src_id = src_id[sel_pos]
        
        # !!! This is where generic obstacles can have their connections defined. this code is sufficient IF:
        # 1. the correct phase can be calculated from a given obstacle
        # 2. the correct border/obstacle cells can be found
        # 3. the corner case for cyclical boundaries (phase == 0 case below)

        if phase == 0:
            sel_dir = (np.abs(self.pop_dict[trg]['prop_dic']['action_phase'] - phase) < np.pi / 2) | (
                    np.abs(self.pop_dict[trg]['prop_dic']['action_phase'] - phase) > 3 * np.pi / 2)
            sel_trg_id = np.array(self.pop_dict[trg]['ids'])[sel_dir]
        else:
            sel_dir = np.abs(self.pop_dict[trg]['prop_dic']['action_phase'] - phase) < np.pi / 2
            sel_trg_id = np.array(self.pop_dict[trg]['ids'])[sel_dir]

        conn_tmp = nest.GetConnections(sel_src_id.tolist(), sel_trg_id.tolist())
        nest.SetStatus(conn_tmp, {'weight': 0.0})

    def remove_boundary_conn_border(self, src='border', trg='action'):
        '''
        projects the weights into the environment of the border cells, if 
        exist
        '''

        self.rmv_bnd_conn_finder_border(src, trg, 'xmax', np.pi / 2)
        self.rmv_bnd_conn_finder_border(src, trg, 'xmin', 3 * np.pi / 2)
        self.rmv_bnd_conn_finder_border(src, trg, 'ymax', 0)
        self.rmv_bnd_conn_finder_border(src, trg, 'ymin', np.pi)

    def rmv_bnd_conn_finder_border(self, src, trg, pos, phase):
        '''
        Finds connections according to the given criteria in order to project 
        feed forward weights of the border cells into the env.
        Inputs:
            scr: string 
                Keyword of the source region
            trg: string
                Keyword of the traget region
            pos: string
                Boundary
            phase: Direction to which that the connections should be weakened.
            This function should be rewritten based on dictionary structure
            rather than using attributes (xmax, xmin,...).
        '''
        bor_id = np.array(self.border_id_list)
        src_id = np.array(self.pop_dict[src]['ids'])

        if pos[0] == 'x':
            if pos[-1] == 'x':
                sel_pos = np.abs(self.pop_dict[src]['prop_dic']['positions'][:, 0] - self.xmax) < 1e-5
                sel_src_id = src_id[sel_pos]
            else:
                sel_pos = np.abs(self.pop_dict[src]['prop_dic']['positions'][:, 0] - self.xmin) < 1e-5
                sel_src_id = src_id[sel_pos]
        else:
            if pos[-1] == 'x':
                sel_pos = np.abs(self.pop_dict[src]['prop_dic']['positions'][:, 1] - self.ymax) < 1e-5
                sel_src_id = src_id[sel_pos]
            else:
                sel_pos = np.abs(self.pop_dict[src]['prop_dic']['positions'][:, 1] - self.ymin) < 1e-5
                sel_src_id = src_id[sel_pos]

        sel_src_id = np.intersect1d(bor_id, sel_src_id)

        if phase == 0:
            sel_dir = (np.abs(self.pop_dict[trg]['prop_dic']['conne'] - phase) < np.pi / 2) | (
                    np.abs(self.pop_dict[trg]['prop_dic']['action_phase'] - phase) > 3 * np.pi / 2)
            sel_trg_id = np.array(self.pop_dict[trg]['ids'])[sel_dir]
        else:
            sel_dir = np.abs(self.pop_dict[trg]['prop_dic']['action_phase'] - phase) < np.pi / 2
            sel_trg_id = np.array(self.pop_dict[trg]['ids'])[sel_dir]

        conn_tmp = nest.GetConnections(sel_src_id.tolist(), sel_trg_id.tolist())
        nest.SetStatus(conn_tmp, {'weight': 0.0})

    def connect_devices(self, pop):
        self.pop_dict[pop]['recorder'] = {}
        sinks = self.net_dict[pop]['sinks']
        for s_ in sinks.keys():
            if 'music' in s_:
                s_id = nest.Create(s_, params=sinks[s_]['params'])
                for i, n in enumerate(self.pop_dict[pop]['ids']):
                    nest.Connect([n], s_id, 'one_to_one', {'music_channel': i})
            elif 'spike' in s_:
                if 'params' in sinks[s_]:
                    p_dic = sinks[s_]['params']
                else:
                    p_dic = {'withgid': True, 'withtime': True, 'to_memory': False, 'to_file': True,
                             'use_gid_in_filename': False, 'label': os.path.join(self.data_path, pop)}
                s_id = nest.Create(s_, params=p_dic)
                nest.Connect(self.pop_dict[pop]['ids'], s_id)

            elif 'weight' in s_:
                if 'params' in sinks[s_]:
                    w_rec = nest.Create(s_, params=sinks[s_]['params'])
                else:
                    p_dic = {'withgid': True, 'withtime': True, 'to_memory': False, 'to_file': True,
                             'use_gid_in_filename': False, 'label': os.path.join(self.data_path, pop)}
                s_id = nest.Create(s_, params=p_dic)
                if pop != "action":
                    syn_name = self.net_dict[pop]['trg_conn']['syn_spec']['model'] + "_" + pop
                    nest.CopyModel(self.net_dict[pop]['trg_conn']['syn_spec']['model'], syn_name,
                                   {'weight_recorder': s_id[0]})
                    self.net_dict[pop]['trg_conn']['syn_spec']['model'] = syn_name
            self.pop_dict[pop]['recorder'] = {s_: deepcopy(s_id)}
        if pop == "border" or pop == "action":
            nest.GetConnections(self.pop_dict[pop]["ids"])

    def save_initial_weights(self):

        f_id = open(os.path.join(self.data_path, 'initial_weights.dat'), 'w')
        f_id.write('pre\tpost\tweight\n')
        conns = nest.GetConnections()
        for row in conns:
            stat_tmp = nest.GetStatus([row])[0]
            f_id.write('{}\t{}\t{}\n'.format(stat_tmp['source'], stat_tmp['target'], stat_tmp['weight']))
        f_id.close()

    def del_zero_pops(self):
        del_list = []
        for pop in self.net_dict.keys():
            if self.net_dict[pop]['num_neurons'] == 0:
                del_list.append(self.net_dict[pop])

    def create_network(self):
        self._istest_(False)
        self._remove_existing_gdffiles_()
        self.pop_dict = {}
        self.del_zero_pops()
        for pop in self.net_dict.keys():
            # Creating populations of neurons
            self.pop_dict[pop] = {}
            self.create_pop(pop)
            self.create_input(pop)

        if pop == 'action':
            # Assigning actions vectors to action neurons
            self.assign_action()
        for pop in self.net_dict.keys():
            # self.connect_populations_fortest(pop, 0)
            self.connect_devices(pop)
            # Connecting populations to populations
            self.connect_populations(pop)
            self.connect_inputs(pop)
            if pop == 'action':
                self.compute_act_pop_wmat(scale=7)
                self.set_act_pop_wmat()
            if pop == 'border':  # or pop == 'place':
                print("Removing " + pop + "->action connections on the env. boundry ...")
                self.remove_boundary_conn(pop, 'action')
                
        if hasattr(self, 'border_id_list'):  # and self.net_dict['border_cell']['flag']:
            if len(self.border_id_list) != 0:
                print("Removing border->action connections on the env. boundry ...")
            #                self.remove_boundary_conn_border()
            else:
                print("border->action connections on the env. boundry remained unchanged!")

    def setup(self):

        self.setup_nest()
        self.create_network()
        self.save_initial_weights()
