#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 22:14:28 2019

@author: Mohammad
"""

import numpy as np

try:
    import nest
    import nest.raster_plot
    import nest.voltage_trace
    import nest.topology as topp
except:
    print('Nest has already been impoted')
import matplotlib.pyplot as plt
import os
import pandas as pd
import shelve

nest.ResetKernel()
nest.SetKernelStatus({'overwrite_files': True})


def connection_nest():
    pre_id = nest.Create('iaf_psc_exp', 4)

    nest.Connect(pre_id, pre_id)
    conn = nest.GetConnections(pre_id)
    W = np.random.uniform(size=(4, 4))
    W[np.arange(4), np.arange(4)] = 0

    for i in conn:
        pr = i[0] - 1
        po = i[1] - 1
        nest.SetStatus([i], {'weight': W[po, pr]})


def action_weight():
    psi = 20
    winh = -300
    wexc = 100
    N = 40
    dir_angle = np.linspace(0, 2 * np.pi * (1 - 1 / N), N)

    dir_angle = dir_angle.reshape(-1, 1)
    w_phase_dep = np.exp(psi * np.cos(dir_angle.repeat(N, axis=1) - dir_angle.T.repeat(N, axis=0)))
    w_phase_dep[np.arange(N), np.arange(N)] = 0
    w_phase_dep = w_phase_dep / w_phase_dep.sum(axis=1)[0]
    w_mat_action = winh / N + wexc * w_phase_dep
    w_mat_action[np.arange(N), np.arange(N)] = 0

    test_path = './test-figs'

    fig, ax = plt.subplots(figsize=(5, 4))
    w_image = ax.imshow(w_mat_action, cmap='viridis')
    bar_h = plt.colorbar(w_image, ax=ax)
    bar_h.set_label('Weight', rotation=270)
    ax.set_xlabel('Presynaptic neurons\' id')
    ax.set_ylabel('Postsynaptic neurons\' id')
    ax.set_title('Action neurons')
    fig.savefig(os.path.join(test_path, 'weight_mat_action.pdf'), format='pdf')


def action_neurons_reponse():
    nest.ResetKernel()

    psi = 20
    winh = -300
    wexc = 100
    N = 40
    pg_fr = 200.
    dir_angle = np.linspace(0, 2 * np.pi * (1 - 1 / N), N)

    dir_angle = dir_angle.reshape(-1, 1)
    w_phase_dep = np.exp(psi * np.cos(dir_angle.repeat(N, axis=1) - dir_angle.T.repeat(N, axis=0)))
    w_phase_dep[np.arange(N), np.arange(N)] = 0
    w_phase_dep = w_phase_dep / w_phase_dep.sum(axis=1)[0]
    w_mat_action = winh / N + wexc * w_phase_dep
    w_mat_action = w_mat_action * 7
    w_mat_action[np.arange(N), np.arange(N)] = 0

    test_path = './test-figs'

    act_neurons = nest.Create('iaf_psc_alpha', N, params={'tau_syn_ex': 5., 'tau_syn_in': 5.})
    nest.Connect(act_neurons, act_neurons, 'all_to_all')
    conn = nest.GetConnections(act_neurons)

    max_act_id = max(act_neurons)
    min_act_id = min(act_neurons)

    for i in conn:
        pr = i[0] - min_act_id - 1
        po = i[1] - min_act_id - 1
        nest.SetStatus([i], {'weight': w_mat_action[po, pr]})

    pg = nest.Create('poisson_generator', 1)

    nest.SetStatus(pg, {'rate': pg_fr})

    nest.Connect(pg, act_neurons, 'all_to_all', syn_spec={'weight': 100.0})

    spk_det = nest.Create('spike_detector', params={'to_memory': True, 'to_file': False})

    #    voltmeter = nest.Create('voltmeter', params={'withgid': True})

    nest.Connect(act_neurons, spk_det)
    #    nest.Connect(voltmeter, act_neurons)

    nest.Simulate(10000.)
    nest.raster_plot.from_device(spk_det)
    plt.savefig('action-direction-competition.pdf', format='pdf')


#    nest.voltage_trace.from_device(voltmeter)

def placecells_position_fr():
    #    nest.ResetKernel()

    layer_dict = {'extent': [4., 4.], 'rows': 11, 'columns': 11, 'elements': 'parrot_neuron'}
    layer = topp.CreateLayer(layer_dict)
    pop_ids = nest.GetNodes(layer)[0]
    pos = np.array(topp.GetPosition(pop_ids))
    #    topp_info = dict(layer_id=layer,
    #                     positions=locations)

    ids = nest.Create('poisson_generator', len(pop_ids))

    nest.Connect(ids, pop_ids, 'one_to_one')

    spk_det = nest.Create('spike_detector', params={'to_memory': False, 'to_file': True, 'label': 'EC'})

    nest.Connect(pop_ids, spk_det)

    spatial_sig = 0.4  # Should be corrected
    curr_pos = (0., 0.)
    avg_fr = 400.

    fr_vec = np.array(pos) - np.array(curr_pos).reshape(1, 2).repeat(len(ids), axis=0)
    fr_vec = avg_fr * np.exp(-(fr_vec ** 2).sum(axis=1) / (spatial_sig ** 2))

    for idx, val in enumerate(fr_vec):
        nest.SetStatus([ids[idx]], {'rate': val})

    nest.Simulate(1000.)

    im_vec = np.zeros((len(ids), 1))

    spk_data = pd.read_csv('spike_detector-244-0.gdf', sep='\t', names=['id', 'times', 'none'])
    spk_data = spk_data.drop(columns='none')
    spk_data_cnt = spk_data.groupby(by='id').count()

    im_vec[spk_data_cnt.index.values - min(pop_ids) - 1] = spk_data_cnt.values

    im_mat = im_vec.reshape(11, 11).T

    fig, ax = plt.subplots()
    image = ax.imshow(im_mat, cmap='viridis')
    bar_h = plt.colorbar(image, ax=ax)
    plt.show()


#    nest.raster_plot.from_device(spk_det)

def test_single_neuron_psp():
    neuron = nest.Create('iaf_psc_alpha', 1, params={'tau_syn_ex': 5.})
    spk_gen = nest.Create('spike_generator', 1)
    voltmeter = nest.Create('voltmeter', params={'withgid': True, 'withtime': True})
    nest.Connect(spk_gen, neuron, syn_spec={'weight': 100.0})
    nest.Connect(voltmeter, neuron)

    nest.SetStatus(spk_gen, {'spike_times': [500.0], 'spike_weights': [1.0]})

    nest.Simulate(1000.)
    nest.voltage_trace.from_device(voltmeter)
    Vm = nest.GetStatus(voltmeter)[0]['events']['V_m']
    min_v = Vm.min()
    max_v = Vm.max()
    print('The voltage difference is: %f' % (max_v - min_v))


def stdp_test():
    neuron_1 = nest.Create('iaf_psc_alpha', 1)
    neuron_2 = nest.Create('iaf_psc_alpha', 1)
    w_rec = nest.Create('weight_recorder', 1)
    dc_curr_1 = nest.Create('dc_generator', 1, {'amplitude': 400., 'start': 10.})
    dc_curr_2 = nest.Create('dc_generator', 1, {'amplitude': 400., 'start': 0.})
    spk_gen = nest.Create('spike_generator', 1, {'spike_times': np.arange(100.0, 1000.0, 50.0)})
    # nest.Connect(spk_gen, neuron_1, syn_spec={'weight':2000, 'delay': 1.0})
    # nest.Connect(spk_gen, neuron_2, syn_spec={'weight':2000, 'delay': 11.0})
    nest.Connect(dc_curr_1, neuron_1)
    nest.Connect(dc_curr_2, neuron_2)
    spk_det = nest.Create('spike_detector', 1, {'to_file': True})
    nest.Connect(neuron_1, spk_det)
    nest.Connect(neuron_2, spk_det)
    nest.CopyModel('stdp_synapse', 'stdp_synapse_rec', {'weight_recorder': w_rec[0], 'weight': 1.0, 'Wmax': 2.0})
    #    nest.CopyModel('stdp_triplet_synapse', 'stdp_synapse_rec', {'weight_recorder': w_rec[0],
    #                                                                'weight': 1.0})
    nest.Connect(neuron_1, neuron_2, syn_spec={'model': 'stdp_synapse_rec'})
    nest.Simulate(10000.0)
    spks = nest.GetStatus(spk_det)[0]['events']
    ids = spks['senders']
    diff = spks['times'][ids == 2] - spks['times'][ids == 1]
    weights = nest.GetStatus(w_rec)[0]['events']
    fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True)
    ax[0].plot(spks['times'][ids == 1], spks['senders'][ids == 1], '|', markersize=200, color='red', label='pre')
    ax[0].plot(spks['times'][ids == 2], spks['senders'][ids == 2], '|', markersize=200, color='blue', label='post')
    ax[0].set_yticks([1, 2])
    ax[1].plot(weights['times'], weights['weights'])
    ax[2].plot(weights['times'][:-1], np.diff(weights['weights']))
    ax[3].plot(weights['times'], diff)

    # ax[0].legend()    
    ax[1].set_ylabel('Weight')
    ax[2].set_ylabel('Weight changes')
    print(weights)
    print('parameters dict:', nest.GetDefaults('stdp_synapse_rec'))
    plt.show()


def calc_stdp_window():
    dt = np.arange(-50, 51, 1)
    ISI = []
    W = []
    for d in dt:
        nest.ResetKernel()
        nest.CopyModel('iaf_psc_alpha', 'iaf_psc_alpha_here', params={'C_m': 200.})
        neuron_1 = nest.Create('iaf_psc_alpha_here', 1)
        neuron_2 = nest.Create('iaf_psc_alpha_here', 1, params={'tau_minus': 20.})
        w_rec = nest.Create('weight_recorder', 1)
        dc_curr_1 = nest.Create('dc_generator', 1, {'amplitude': 350., 'start': 10.})
        dc_curr_2 = nest.Create('dc_generator', 1, {'amplitude': 350., 'start': 0.})
        spk_gen1 = nest.Create('spike_generator', 1, {'spike_times': np.array([200., 500.])})
        spk_gen2 = nest.Create('spike_generator', 1, {'spike_times': np.array([200 + d, 900.])})
        nest.Connect(spk_gen1, neuron_1, syn_spec={'weight': 2000})
        nest.Connect(spk_gen2, neuron_2, syn_spec={'weight': 2000})
        # nest.Connect(dc_curr_1, neuron_1)
        # nest.Connect(dc_curr_2, neuron_2)
        spk_det = nest.Create('spike_detector', 1, {'to_file': False})
        nest.Connect(neuron_1, spk_det)
        nest.Connect(neuron_2, spk_det)
        nest.CopyModel('stdp_synapse', 'stdp_synapse_rec',
                       {'weight_recorder': w_rec[0], 'weight': 1.0, 'Wmax': 2.0, 'tau_plus': 20., 'alpha': -1.,
                        'lambda': -0.01})
        #    nest.CopyModel('stdp_triplet_synapse', 'stdp_synapse_rec', {'weight_recorder': w_rec[0],
        #                                                                'weight': 1.0})
        nest.Connect(neuron_1, neuron_2, syn_spec={'model': 'stdp_synapse_rec'})
        nest.Simulate(1000.0)
        spks = nest.GetStatus(spk_det)[0]['events']
        ids = spks['senders']
        spks1 = spks['times'][ids == 1]
        spks2 = spks['times'][ids == 2]
        weights = nest.GetStatus(w_rec)[0]['events']['weights']
        ISI.append(spks2[0] - spks1[0])
        W.append(weights[-1] - 1)
    fig, ax = plt.subplots()
    ax.scatter(ISI, W)
    ax.set_xlabel(r'T_{post} - T_{pre}')
    ax.set_ylabel(r'\DeltaW')
    plt.show()


def calc_stdp_DA_window():
    dt = np.arange(-50, 51, 1)
    da_t = np.arange(0., 9000., 500.)

    for dd in da_t:
        ISI = []
        W = []
        for d in dt:
            nest.ResetKernel()
            nest.CopyModel('iaf_psc_alpha', 'iaf_psc_alpha_here', params={'C_m': 200.})
            neuron_1 = nest.Create('iaf_psc_alpha_here', 1)
            neuron_2 = nest.Create('iaf_psc_alpha_here', 1, params={'tau_minus': 20.})
            neuron_DA = nest.Create('parrot_neuron', 1)
            w_rec = nest.Create('weight_recorder', 1)
            vol_trans = nest.Create('volume_transmitter', 1)
            #        dc_curr_1 = nest.Create('dc_generator', 1, {'amplitude':350., 'start': 10.})
            #        dc_curr_2 = nest.Create('dc_generator', 1, {'amplitude':350., 'start': 0.})
            spk_gen1 = nest.Create('spike_generator', 1, {'spike_times': np.array([200., 9500.])})
            spk_gen2 = nest.Create('spike_generator', 1, {'spike_times': np.array([200 + d, 9900.])})
            spk_genDA = nest.Create('spike_generator', 1, {'spike_times': np.array([200. + dd])})
            nest.Connect(spk_gen1, neuron_1, syn_spec={'weight': 2000})
            nest.Connect(spk_gen2, neuron_2, syn_spec={'weight': 2000})
            nest.Connect(spk_genDA, neuron_DA)
            nest.Connect(neuron_DA, vol_trans)
            # nest.Connect(dc_curr_1, neuron_1)
            # nest.Connect(dc_curr_2, neuron_2)
            spk_det = nest.Create('spike_detector', 1, {'to_file': False})
            nest.Connect(neuron_1, spk_det)
            nest.Connect(neuron_2, spk_det)
            nest.CopyModel('stdp_dopamine_synapse', 'stdp_synapse_rec',
                           {'weight_recorder': w_rec[0], 'vt': vol_trans[0], 'weight': 1.0, 'Wmax': 2.0,
                            'tau_plus': 20., 'tau_c': 2000.})
            #    nest.CopyModel('stdp_triplet_synapse', 'stdp_synapse_rec', {'weight_recorder': w_rec[0],
            #                                                                'weight': 1.0})
            nest.Connect(neuron_1, neuron_2, syn_spec={'model': 'stdp_synapse_rec'})
            nest.Simulate(10000.0)
            spks = nest.GetStatus(spk_det)[0]['events']
            ids = spks['senders']
            spks1 = spks['times'][ids == 1]
            spks2 = spks['times'][ids == 2]
            weights = nest.GetStatus(w_rec)[0]['events']['weights']
            print(nest.GetStatus(w_rec)[0]['events'])
            ISI.append(spks2[0] - spks1[0])
            W.append(weights[-1] - 1)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(ISI, W)
        ax.set_xlabel(r'$T_{post} - T_{pre}$')
        ax.set_ylabel(r'$\Delta W$')
        ax.set_title(r'$T_{DA} - T_{pre-post}=%d$' % (dd))
        fig.savefig('diff%d.png' % (dd), format='png')


#    plt.show()

def IF_response_test():
    model = 'aeif_psc_alpha'
    params = {'a': 4.0, 'b': 80.8, 'V_th': -50.4, 'Delta_T': 2.0, 'I_e': 0.0, 'C_m': 281.0, 'g_L': 30.0,
              'V_reset': -70.6, 'tau_w': 144.0, 't_ref': 5.0, 'V_peak': -40.0, 'E_L': -70.6, # 'E_ex': 0.,
              # 'E_in': -70.
              }
    transfer = IF_curve(model, params)
    transfer.compute_transfer()
    dat = shelve.open(model + '_transfer.dat')
    dat['I_mean'] = transfer.i_range
    dat['I_std'] = transfer.std_range
    dat['rate'] = transfer.rate
    dat['CV'] = transfer.cv
    dat['CV_pop'] = transfer.cv_pop
    dat['spikes'] = transfer.spks_list
    dat.close()
    std_range = transfer.std_range
    curr_range = transfer.i_range
    rate = transfer.rate
    fig, ax = plt.subplots()
    ax.plot(curr_range, rate[:, 0])


def IF_response_fromfile_test():
    n_pop = 10
    dat = shelve.open('aeif_psc_alpha' + '_transfer.dat')
    I = dat['I_mean']
    noise = dat['I_std']
    rate = dat['rate']
    cv = dat['CV']
    cv_mean = cv.mean(axis=2)
    cv = dat['CV_pop']
    spks = dat['spikes']

    fig, ax = plt.subplots(nrows=np.ceil(noise.size / 4).astype(int), ncols=4, figsize=(12, 8), sharex=True,
                           sharey=True)
    print(ax.shape)
    for i, n in enumerate(noise):
        ax_tmp = ax[i // 4, i % 4]
        c1 = 'blue'
        ax_tmp.plot(I, rate[:, i], color=c1)
        ax_tmp.set_ylabel('Firing rate (Hz)', color=c1)
        ax_tmp.tick_params(axis='y', labelcolor=c1, color=c1)
        c2 = 'red'
        ax_tmp2 = ax_tmp.twinx()
        ax_tmp2.plot(I, cv[:, i], color=c2)
        ax_tmp2.set_ylabel('CV', color=c2)
        ax_tmp2.tick_params(axis='y', labelcolor=c2, color=c2)
        ax_tmp.set_title('std=%d' % n)

    fig.tight_layout()
    fig.savefig('noise_n%d.pdf' % n_pop, format='pdf')

    # Plotting raster
    i_s = np.where(I == 700)[0][0]
    std_s = np.where(noise == 200)[0][0]
    ind = i_s * noise.size + std_s

    spikes_s = spks[ind]
    fig, ax = plt.subplots()
    times = spikes_s['times']
    time_int = (times > 500) & (times < 1000)
    ax.plot(times[time_int], spikes_s['senders'][time_int], marker="|", linestyle='')
    # ax.scatter(times[times<4000], spikes_s['senders'][times<4000], s=1)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron ID')
    ax.set_title(r'$\mu=%d, \sigma=%d$' % (700, 200))
    fig.savefig('raster-plot.pdf', format='pdf')


class IF_curve():
    t_inter_trial = 200.  # Interval between two successive measurement trials
    t_sim = 10000.  # Duration of a measurement trial
    n_neurons = 10  # Number of neurons
    n_threads = 4  # Nubmer of threads to run the simulation

    def __init__(self, model, params=False):
        self.model = model
        self.params = params
        self.build()
        self.connect()

    def build(self):
        #######################################################################
        #  We reset NEST to delete information from previous simulations
        # and adjust the number of threads.

        nest.ResetKernel()
        nest.SetKernelStatus({'local_num_threads': self.n_threads})

        #######################################################################
        # We set the default parameters of the neuron model to those
        # defined above and create neurons and devices.

        if self.params:
            nest.SetDefaults(self.model, self.params)
        self.neuron = nest.Create(self.model, self.n_neurons)
        self.min_n_id = min(self.neuron)
        self.noise = nest.Create('noise_generator')
        self.spike_detector = nest.Create('spike_detector')

    def connect(self):
        #######################################################################
        # We connect the noisy current to the neurons and the neurons to
        # the spike detectors.

        nest.Connect(self.noise, self.neuron, 'all_to_all')
        nest.Connect(self.neuron, self.spike_detector, 'all_to_all')

    def output_cv(self):
        spk_times = nest.GetStatus(self.spike_detector, 'events')[0]['times']
        spk_ids = nest.GetStatus(self.spike_detector, 'events')[0]['senders']
        spk_ids_u = np.unique(spk_ids)
        cv = np.zeros(shape=self.n_neurons)
        if spk_ids_u.size > 0:
            spk_ids = spk_ids - self.min_n_id
            for i in range(self.n_neurons):
                spk_tmp = spk_times[spk_ids == i]
                if spk_tmp.size > 1:
                    spk_tmp.sort()
                    ISI = np.diff(spk_tmp)
                    cv[i] = ISI.std() / ISI.mean()
                else:
                    cv[i] = 0.0
        return cv

    def output_rate(self, mean, std):
        self.build()
        self.connect()

        #######################################################################
        # We adjust the parameters of the noise according to the current
        # values.

        nest.SetStatus(self.noise, [{'mean': mean, 'std': std, 'start': 0.0, 'stop': 10000., 'origin': 0.}])

        # We simulate the network and calculate the rate.

        nest.Simulate(self.t_sim)
        rate = nest.GetStatus(self.spike_detector, 'n_events')[0] * 1000.0 / (1. * self.n_neurons * self.t_sim)
        spks_t_id = nest.GetStatus(self.spike_detector, 'events')[0]
        spk_times = spks_t_id['times'].copy()
        cv = self.output_cv()
        if spk_times.size > 1:
            spk_times.sort()
            ISI = np.diff(spk_times)
            if ISI.std() < 1e-6:
                cv_pop = 0.0
            else:
                cv_pop = ISI.std() / ISI.mean()
        else:
            ISI = np.array([])
            cv_pop = 0.0
        return rate, spks_t_id, cv, cv_pop

    def compute_transfer(self, i_mean=(400.0, 900.0, 10.0), i_std=(0.0, 600.0, 50.0)):
        #######################################################################
        # We loop through all possible combinations of `(I_mean, I_sigma)`
        # and measure the output rate of the neuron.

        self.i_range = np.arange(*i_mean)
        self.std_range = np.arange(*i_std)
        self.rate = np.zeros((self.i_range.size, self.std_range.size))
        self.cv_pop = np.zeros((self.i_range.size, self.std_range.size))
        self.cv = np.zeros((self.i_range.size, self.std_range.size, self.n_neurons))
        self.spks_list = []
        nest.set_verbosity('M_WARNING')
        for n, i in enumerate(self.i_range):
            print('I  =  {0}'.format(i))
            for m, std in enumerate(self.std_range):
                self.rate[n, m], spk_times, self.cv[n, m, :], self.cv_pop[n, m] = self.output_rate(i, std)
                self.spks_list.append(spk_times)


#    print('Minimum change of the weights are: %f' %(np.diff(weights['weights']).min()))
#    nest.Connect(neuron_1, w_rec)
# action_weight()
# action_neurons_reponse()
# placecells_position_fr()
# test_single_neuron_psp()
# stdp_test()
# IF_response_test()
# IF_response_fromfile_test()
calc_stdp_DA_window()
