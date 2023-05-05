#!/usr/bin/env python
"""Runner for the model created in network.py.

The parameter files required for the model are "sim_params.json" and 
"network_params.json"
"""
from datetime import datetime
import json
import nest
import numpy as np
from network import Network
from mpi4py import MPI  # needs to be imported after NEST

comm = MPI.COMM_WORLD
with open('network_params_spikingnet.json', 'r') as fl:
    net_dict = json.load(fl)

with open('sim_params.json', 'r') as fl:
    sim_dict = json.load(fl)

net = Network(sim_dict=sim_dict, net_dict=net_dict)
net.setup()
comm.Barrier()
start = datetime.now()
nest.Simulate(sim_dict['simtime'])
end = datetime.now()
dt = end - start
run_time = dt.seconds + dt.microseconds / 1000000.
print('\n\nRUN TIME: {} seconds\n\n'.format(run_time))
