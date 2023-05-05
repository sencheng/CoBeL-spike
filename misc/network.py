#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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


class Network:
    """ Handles the setup of the network parameters and
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

        self._create_dir_('data_path')

        # *:Iterating over all contents of the following expression (here is list).
        fig_path = os.path.join(*sim_dict['data_path'].split('/')[:-1], 'fig')
        self.sim_dict['fig_path'] = fig_path

        self._create_dir_('fig_path')

        self.sim_interval = [0]

        self.curr_pos = np.array([0., 0.])

        self.rat_pos = [self.curr_pos]
        self.rat_pos_all = [self.curr_pos]

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
        """ Hands parameters to the NEST-kernel.

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

    def create_topological_pop(self, pop):

        spatial_dict = self.net_dict[pop]['spatial_prop']
        layer_dict = {'extent': [float(spatial_dict['width']), float(spatial_dict['height'])],
                      'rows': spatial_dict['nrows'], 'columns': spatial_dict['ncols'],
                      'elements': self.net_dict[pop]['model']}
        layer = topp.CreateLayer(layer_dict)
        pop_ids = nest.GetNodes(layer)[0]
        locations = np.array(topp.GetPosition(pop_ids))
        topp_info = dict(layer_id=layer, positions=locations)

        self.xmin = locations[:, 0].min()
        self.xmax = locations[:, 0].max()

        self.ymin = locations[:, 1].min()
        self.ymax = locations[:, 1].max()

        return topp_info, pop_ids

    def create_input(self, pop):

        if self.net_dict[pop]['ext_inputs']:

            inp = self.net_dict[pop]['ext_inputs']

            if self.net_dict[pop]['inp_conn']['conn_spec']['rule'] == 'one_to_one':
                inp_id = nest.Create(inp, len(self.pop_dict[pop]['ids']))

            else:
                inp_id = nest.Create(inp, 1)

            self.pop_dict[pop]['input_id'] = inp_id

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
        action = a0 * np.vstack((np.sin(dir_angle), np.cos(dir_angle)))
        self.pop_dict['action']['prop_dic']['action_vec'] = action
        self.pop_dict['action']['prop_dic']['action_phase'] = dir_angle

    def connect_populations(self, pop):

        #        # ADDED FOR TESTING
        #
        #        trg = self.net_dict[pop]['targets']
        #        src = pop
        #
        #        if src == 'place':
        #            src_ids = self.pop_dict[src]['input_id']
        #        else:
        #            src_ids = self.pop_dict[src]['ids']

        par_rnd_dic = self.net_dict[pop]['trg_conn_rnd']
        if par_rnd_dic['flag']:
            #            syn_dic = self.net_dict[pop]['trg_conn']['syn_spec']
            #            syn_dic['weight'] = {'distribution': par_rnd_dic['dist'],
            #                                 'mu': float(syn_dic['weight']),
            #                                 'sigma': par_rnd_dic['std']}

            self.net_dict[pop]['trg_conn']['syn_spec']['weight'] = {'distribution': par_rnd_dic['dist'], 'mu': float(
                self.net_dict[pop]['trg_conn']['syn_spec']['weight']), 'sigma': par_rnd_dic['std']}

        #            self.net_dict[pop]['trg_conn']['syn_spec']['weight'] = \
        #            {'distribution': 'uniform',
        #             'low': float(self.net_dict[pop]['trg_conn']['syn_spec']['weight'])-par_rnd_dic['std'],
        #             'high': float(self.net_dict[pop]['trg_conn']['syn_spec']['weight'])+par_rnd_dic['std']}

        trg = self.net_dict[pop]['targets']

        if isinstance(trg, list):
            for trgs in trg:
                print('Not implemented!')
                break
        #                nest.Connect(self.pop_dict[src]['ids'],
        #                             self.pop_dict[trgs]['ids'])
        elif trg:
            #            #REMOVED FOR TESTING
            nest.Connect(self.pop_dict[pop]['ids'], self.pop_dict[trg]['ids'], **self.net_dict[pop]['trg_conn'])

    #            #ADDED FOR TESTING
    #            nest.Connect(src_ids,
    #                         self.pop_dict[trg]['ids'],
    #                         **self.net_dict[src]['trg_conn'])

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

    def remove_boundary_conn(self, src='place', trg='action'):

        self.rmv_bnd_conn_finder(src, trg, 'xmax', np.pi / 2)
        self.rmv_bnd_conn_finder(src, trg, 'xmin', 3 * np.pi / 2)
        self.rmv_bnd_conn_finder(src, trg, 'ymax', 0)
        self.rmv_bnd_conn_finder(src, trg, 'ymin', np.pi)

    def rmv_bnd_conn_finder(self, src, trg, pos, phase):

        '''
        Finds connections according to the given criteria.
        Inputs:
            scr: Keyword of the source region
            trg: Keyword of the traget region
            pos: Boundary
            phase: Direction to which that the connections should be weakened.
            This function should be rewritten based on dictionary structure
            rather than using attributes (xmax, xmin,...).
        '''
        src_id = np.array(self.pop_dict[src]['ids'])
        #        src_id = src_id - src_id.min()

        if pos[0] == 'x':
            if pos[-1] == 'x':
                sel_pos = np.abs(self.pop_dict[src]['prop_dic']['positions'][:, 0] - self.xmax) < 1e-4
                sel_src_id = src_id[sel_pos]
            else:
                sel_pos = np.abs(self.pop_dict[src]['prop_dic']['positions'][:, 0] - self.xmin) < 1e-4
                sel_src_id = src_id[sel_pos]
        else:
            if pos[-1] == 'x':
                sel_pos = np.abs(self.pop_dict[src]['prop_dic']['positions'][:, 1] - self.ymax) < 1e-6
                sel_src_id = src_id[sel_pos]
            else:
                sel_pos = np.abs(self.pop_dict[src]['prop_dic']['positions'][:, 1] - self.ymin) < 1e-4
                sel_src_id = src_id[sel_pos]
        # The following part can be improved ...
        if phase == 0:
            sel_dir = (np.abs(self.pop_dict[trg]['prop_dic']['action_phase'] - phase) < np.pi / 2) | (
                        np.abs(self.pop_dict[trg]['prop_dic']['action_phase'] - phase) > 3 * np.pi / 2)
            sel_trg_id = np.array(self.pop_dict[trg]['ids'])[sel_dir]
        else:
            sel_dir = np.abs(self.pop_dict[trg]['prop_dic']['action_phase'] - phase) < np.pi / 2
            sel_trg_id = np.array(self.pop_dict[trg]['ids'])[sel_dir]

        conn_tmp = nest.GetConnections(sel_src_id.tolist(), sel_trg_id.tolist())
        nest.SetStatus(conn_tmp, {'weight': 0.0})

    #    def connect_action_neurons(self):
    #
    #        act_gids = self.pop_dict['action']['ids']
    #        nest.Connect(act_gids, act_gids)
    #        conns = nest.GetStatus(act_gids)
    #
    #        min_act_gids = min(act_gids)
    #
    #        for c in conns:
    #            pre = c[0] - min_act_gids - 1
    #            post = c[1] - min_act_gids - 1
    #            nest.SetStatus([c], self.w_mat_action[post, pre])
    #        if not hasattr(self, 'w_mat_action'):
    #            self.get_action_neurons_wmat()
    #
    #        fig, ax = plt.subplots()
    #        w_image = ax.imshow(self.w_mat_action, cmap='viridis')
    #        bar_h = plt.colorbar(w_image, ax=ax)
    #        bar_h.set_label('Weight', rotation=270)
    #        ax.set_xlabel('Presynaptic neurons\' id')
    #        ax.set_ylabel('Postsynaptic neurons\' id')
    #        ax.set_title('Action neurons')
    #        fig.savefig(os.path.join(test_path, 'weight_mat_action.pdf'),
    #                    format='pdf')

    def connect_devices(self, pop):

        recorder_type = 'spike_detector'

        #        for pop in self.net_dict.keys():

        if not ('recorder' in self.pop_dict[pop]):
            self.pop_dict[pop]['recorder'] = {}

        if pop == 'action':
            to_mem = True
        else:
            to_mem = False

        spk_det_dic = {'withgid': True, 'withtime': True, 'to_memory': to_mem, 'to_file': True,
                       'use_gid_in_filename': False, 'label': os.path.join(self.data_path, pop)}

        spk_det = nest.Create(recorder_type, params=spk_det_dic)
        self.pop_dict[pop]['recorder'] = {recorder_type: spk_det}
        nest.Connect(self.pop_dict[pop]['ids'], spk_det)

    def test_action_neurons(self, inp_w=100.0, inp_fr=200.0, act_w_scale=7.0):

        self._istest_(True)
        self._remove_existing_gdffiles_()

        self.setup_nest()

        pop = 'action'
        inp = 'poisson_generator'

        self.pop_dict = {pop: {}}

        self.create_pop(pop)
        self.connect_populations(pop)

        self.assign_action()
        self.compute_act_pop_wmat(scale=act_w_scale)
        self.w_mat_action = self.w_mat_action
        self.set_act_pop_wmat()
        inp_id = nest.Create(inp, 1, params={'rate': float(inp_fr)})
        #        spk_det = nest.Create('spike_detector',
        #                              params={'to_memory': False,
        #                                      'to_file': True,
        #                                      'withgid': True,
        #                                      'withtime': True,
        #                                      'label': os.path.join(
        #                                              self.sim_dict['data_path_test'],
        #                                              pop)})

        nest.Connect(inp_id, self.pop_dict[pop]['ids'], syn_spec={'weight': float(inp_w)})

        self.connect_devices(pop)

    def test_action_neurons_multinp(self, inp_w=100.0, inp_fr=200.0, act_w_scale=7.0):
        inhom_fr = True
        num_nrs = 121
        self._istest_(True)
        self._remove_existing_gdffiles_()

        self.setup_nest()

        pop = 'action'
        inp = 'poisson_generator'

        self.pop_dict = {pop: {}}

        self.create_pop(pop)
        self.connect_populations(pop)

        self.assign_action()
        self.compute_act_pop_wmat(scale=act_w_scale)
        self.w_mat_action = self.w_mat_action
        self.set_act_pop_wmat()

        if inhom_fr:
            inp_ids = np.arange(num_nrs)
            inp_fr = inp_fr * np.exp(-inp_ids * 0.1)

        if type(inp_fr) == 'int':
            inp_fr = float(inp_fr)

        inp_id = nest.Create(inp, num_nrs)

        for idx, i_id in enumerate(inp_id):
            nest.SetStatus([i_id], params={'rate': inp_fr[idx]})
        #        spk_det = nest.Create('spike_detector',
        #                              params={'to_memory': False,
        #                                      'to_file': True,
        #                                      'withgid': True,
        #                                      'withtime': True,
        #                                      'label': os.path.join(
        #                                              self.sim_dict['data_path_test'],
        #                                              pop)})

        nest.Connect(inp_id, self.pop_dict[pop]['ids'], 'all_to_all', syn_spec={'weight': inp_w})

        self.connect_devices(pop)

        self.pop_dict[pop]['input_id'] = inp_id

    def create_network(self):

        self._istest_(False)
        self._remove_existing_gdffiles_()

        self.pop_dict = {}

        #        for idx, pop in enumerate(self.net_dict['populations']):
        for pop in self.net_dict.keys():
            # Creating populations of neurons

            self.pop_dict[pop] = {}
            #            if isinstance(self.num_neurons[idx], list):

            # Creating inputs

            self.create_pop(pop)

            self.create_input(pop)

        # Assigning actions vectors to action neurons

        self.assign_action()

        for pop in self.net_dict.keys():
            # Connecting populations to populations

            self.connect_populations(pop)

            self.connect_devices(pop)

            self.connect_inputs(pop)

            if pop == 'action':
                self.compute_act_pop_wmat(scale=7)
                self.set_act_pop_wmat()

        # Update place cells' firing rate

        self.update_placecells_fr()

        self.remove_boundary_conn()

    def navigate(self):

        ac_ids = self.pop_dict['action']['ids']
        spk_det = self.pop_dict['action']['recorder']['spike_detector']
        spk_ids = nest.GetStatus(spk_det)[0]['events']['senders']
        spk_tms = nest.GetStatus(spk_det)[0]['events']['times']
        spk_ids = spk_ids[(spk_tms >= self.sim_interval[-2]) & (spk_tms < self.sim_interval[-1])]
        spk_tms = spk_tms[(spk_tms >= self.sim_interval[-2]) & (spk_tms < self.sim_interval[-1])]
        if spk_ids.any():
            spk_ids = spk_ids - min(ac_ids)

            spk_df = pd.DataFrame(np.hstack((np.array(spk_ids).reshape(-1, 1), np.array(spk_tms).reshape(-1, 1))),
                                  columns=['id', 'time'])
            cnt_tmp = spk_df.groupby(by='id').count()
            cnt_id = cnt_tmp.index.values.astype(int)
            cnt = cnt_tmp.values
            action_vec = self.pop_dict['action']['prop_dic']['action_vec']
            #            cnt_id = np.array([0])
            #            cnt = np.array([5])
            avg_vec_sum = np.dot(action_vec[:, cnt_id], cnt) / len(ac_ids)
            avg_vec_sum = avg_vec_sum.reshape(self.curr_pos.shape)

            #        self.curr_pos = self.curr_pos + avg_vec_sum
            self.is_in_field(avg_vec_sum)
            self.rat_pos.append(self.curr_pos + avg_vec_sum)
            self.rat_pos_all.append(self.curr_pos + avg_vec_sum)
        else:
            self.rat_pos_all.append(self.curr_pos)

        self.update_placecells_fr()

    def is_in_field(self, avg_vec_sum):
        curr_pos = self.curr_pos + avg_vec_sum
        if (curr_pos[0] > self.xmax or curr_pos[0] < self.xmin or curr_pos[1] > self.ymax or curr_pos[1] < self.ymin):
            self.bounce_back(curr_pos)
        else:
            self.curr_pos = curr_pos

    def bounce_back(self, curr_pos):
        a0 = 0.01
        if curr_pos[0] > self.xmax:
            ac_phase = 3 * np.pi / 2
        elif curr_pos[0] < self.xmin:
            ac_phase = np.pi / 2
        elif curr_pos[1] < self.ymin:
            ac_phase = 0
        elif curr_pos[1] > self.ymax:
            ac_phase = np.pi
        bounce_vec = np.array([np.sin(ac_phase), np.cos(ac_phase)])
        self.curr_pos = self.curr_pos + a0 * bounce_vec

    def setup(self):

        self.setup_nest()
        self.create_network()

    def simulate(self, time):

        nest.Simulate(float(time))
        last_simtime = self.sim_interval[-1]
        self.sim_interval.append(time + last_simtime)


class Incomplete_implementation_error():
    def __init__(self, value):
        self.value = '{} hasn\'t been implemeted yet!'.format(value)

    def __str__(self):
        return repr(self.value)
