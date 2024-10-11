#!/usr/bin/env python3

"""
This program modifies parameters in the MUSIC file using data from the JSON file.
It also updates multiple configuration files, including the JSON file and the hostfile.

**** How to Use ****
1. Assign the JSON file name to the variable 'file_json'.
2. Assign the MUSIC file name to the variable 'file_music'.
3. To create a new MUSIC file with the updated parameters, assign a new name with the .music extension 
   to the variable 'new_file_music'. To modify the old MUSIC file, assign the old MUSIC file name 
   (same as the 'file_music' variable).
4. Use the functions from the `File_operation` class to update specific parameters in the MUSIC file 
   from the JSON file.
5. Run this script to update the parameters in the MUSIC file.

**** File_operation Class Description ****
List of functions in the class to update the parameters, with their descriptions:
    - update_simtime: Updates the simulation time parameter.
    - update_simdt: Updates the timestep parameter.
    - update_nest_number_of_processor: Updates the parameter for the number of processors for NEST.
    - update_input_num_neurons: Updates the number of neurons from input parameters.
    - update_actor_num_neurons: Updates the number of neurons from actor parameters.
    - update_representation_type: Updates the representation type.

List of Extra Functions:
    - print_changes: Highlights the changes in the MUSIC file.
   
**Example Usage**
Refer to the example section below for how to use the functions from the `File_operation` class.

**Notes:**
- In the JSON file, the time is in milliseconds, while in the MUSIC file, it is in seconds.
- No new MUSIC file will be created if there is no difference between the JSON and current MUSIC file.
"""

import warnings
import time
from update_classes import File_operation as f

###---  Input the file names here.  ---###

file_json_1 = './parameter_sets/current_parameter/sim_params.json'
file_json_2 = './parameter_sets/current_parameter/network_params_spikingnet.json'
env_file = './parameter_sets/current_parameter/env_params.json'
file_music = './parameter_sets/current_parameter/nest_openfield.music'
new_file_music = './parameter_sets/current_parameter/nest_openfield.music'

###------------------------------------###

# Object of the File_operation class
test1 = f(file_json_1, file_music)

###--- Example Section ---###

# Update the simulation time, simulation time step, number of processors used by NEST, 
# and number of neurons from input and actor parameters.
# test1.update_simtime_json() # Updates TO sim_params.json (just to calculate simtime from num_trials * trial_duration)

# Update from sim_params.json
test1.update_simtime()
test1.update_simdt()
test1.update_nest_number_of_processor()

# Prints the changes in the MUSIC file
number_of_updates_1 = test1.print_changes()

# Generate warning while replacing the old MUSIC file from sim_params.json
if number_of_updates_1 == 0:
    print("No changes required in the MUSIC file from sim_params.json")
else:
    print(f"There are {number_of_updates_1} change(s) in the MUSIC file from sim_params.json")
    if file_music == new_file_music:
        warnings.warn("Replacing old MUSIC file.......")
    else:
        warnings.warn("Creating new MUSIC file.......")
    # Commit changes to the MUSIC file
    test1.update_musicfile(new_file_music)

test2 = f(file_json_2, file_music)

# Update from nest_openfield_spikingnet.json
test2.update_actor_num_neurons()
test2.update_representation_type()
test2.update_input_num_neurons()

number_of_updates_2 = test2.print_changes()

# Generate warning while replacing the old MUSIC file from nest_openfield_spikingnet.json
if number_of_updates_2 == 0:
    print("No changes required in the MUSIC file from nest_openfield_spikingnet.json")
else:
    print(f"There are {number_of_updates_2} change(s) in the MUSIC file from nest_openfield_spikingnet.json")
    if file_music == new_file_music:
        warnings.warn("Replacing old MUSIC file.......")
    else:
        warnings.warn("Creating new MUSIC file.......")
    # Commit changes to the MUSIC file
    test2.update_musicfile(new_file_music)

test2.update_jsonfile()
test2.update_hostfile()
time.sleep(2)  # Wait a bit so changes can be seen

# This will always find new ports, so displaying changes isn't very informative
port_updater = f(env_file, file_music)
port_updater.update_ports()
port_updater.update_jsonfile()
port_updater.update_musicfile(new_file_music)
