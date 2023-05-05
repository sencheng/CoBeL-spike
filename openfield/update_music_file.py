"""

This Program helps to modify the parameters in the MUSIC file from the JSON file. 
Now updates multiple config files including the Json file and the hostfile.

**** How to use ****
1. Assign the JSON file name to the variable 'file_json'
2. Assign the MUSIC file name to the variable 'file_music'
3. To create a new MUSIC file with the updated parameters assign a new name with .music extension to the variable 'new_file_music'
   or to modify the old MUSIC file assign the old MUSIC file name (same as 'file_music' variable)
4. Use the functions from File_opration class to update specific paramaters in the MUSIC file from the JSON file
5. Now run this file to finally update the parameters in the MUSIC file.

**** File_operation class descrtiption ****
List of functions in the class to update the parameters with their description is below:
    update_simtime                     | updating the simulation time parameter
    update_simdt                       | updating the timestep parameter
    update_nest_number_of_processor    | updating the parameter for number of processor for nest
    update_input_num_neurons           | updating the number of neurons from input_params
    update_actor_num_neurons           | updating the number of neurons from actor_params
    update_representation_type         | updating the representation type

List of extra functions:
   print_changes                       | Highlight the changes in the MUSIC file
   
**Look at the Example section below for example of how to use the functions from File_operation class.
**Also note that in the JSON file, the time is in milisecond and in music it is in second.
**Note that no new MUSIC file will be created if there is no diffeence between the JSON and current MUSIC file.

"""
# Import the class
import warnings
import time
from update_classes import File_operation as f

###---  Input the file names here.  ---###

file_json_1 = 'sim_params.json'
file_json_2 = 'network_params_spikingnet.json'
file_music = 'nest_openfield.music'
new_file_music = 'nest_openfield.music'

###------------------------------------###

# Object of the File_operation
test1 = f(file_json_1, file_music)

###----- Example section**

# update the simulation time, simulation time step, number of processor used by nest, number of neurons from input and actor parms
test1.update_simtime_json() # Updates TO sim_params.json (just to calculate simtime from num_trials * trial_duration)
# Update from sim_params.json
test1.update_simtime()
test1.update_simdt()
test1.update_nest_number_of_processor()
obs = test1.get_obstacle_info()

###-----

# Prints the changes in Music file
number_of_updates_1 = test1.print_changes()

# Generate warning while replacing old music file from sim_params.json

if (number_of_updates_1 == 0):
    print("No change in MUSIC file require from sim_params.json")
else:
    print("There are " + str(number_of_updates_1) + " change(s) in the MUSIC file from sim_params.json")
    if (file_music == new_file_music):
        warnings.warn("Replacing old music file.......")
    else:
        warnings.warn("Creating new music file.......")
    # Finally commit changes to the MUSIC file
    test1.update_musicfile(new_file_music)

test2 = f(file_json_2, file_music)

# Update from nest_openfield_spikingnet.json
test2.update_actor_num_neurons()
test2.update_representation_type()
test2.update_input_num_neurons(num_obs=obs)

number_of_updates_2 = test2.print_changes()
# Generate warning while replacing old music file from nest_openfield_spikingnet.json


if (number_of_updates_2 == 0):
    print("No change in MUSIC file required from nest_openfield_spikingnet.json")
else:
    print("There are " + str(number_of_updates_2) + " change(s) in the MUSIC file from nest_openfield_spikingnet.json")
    if (file_music == new_file_music):
        warnings.warn("Replacing old music file.......")
    else:
        warnings.warn("Creating new music file.......")
    # Finally commit changes to the MUSIC file
    test2.update_musicfile(new_file_music)
    
test2.update_jsonfile()
test2.update_hostfile()
time.sleep(2) # wait a bit so changes can be seen