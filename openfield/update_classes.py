"""

This file helps in updating the MUSIC file or create a new MUSIC file from the JSON file.
It creates a dictionary from the JSON file and a list from the MUSIC file. Then it compares the corrosponding parameters 
and updates the list. Then the list is printed as a new MUSIC file or it is updated to the old MUSIC file.

"""

import json
import numpy as np
import traceback
import os
import difflib
from colorama import Fore, Style


# Check_content class checks each corrosponding items from the JSON (dictionary) and music (list) file
class Check_content:

    # Returns the string before equal(=) sign
    def only_string(self, string):
        only_str = ''
        for s in string:
            if (s == '='):
                break
            only_str = only_str + s
        return only_str

    # Returns the number after equal(=) sign as a string
    def only_number(self, string):
        only_num = ''
        temp = 0
        for s in string:
            if (temp == 1):
                if (s == '/n'):
                    break
                only_num = only_num + s
            if (s == '='):
                temp = 1
        return only_num

    # Returns the number within sqarebrackets([]) as a string
    def between_brackets(self, string):
        phrase = ''
        temp = 0
        for s in string:
            if (temp == 1):
                if (s == ']'):
                    break
                phrase = phrase + s
            if (s == '['):
                temp = 1
        return phrase

    def check_port(self, data_json, data_music, port):
        len = np.size(data_music)
        incr = 0
        for i in range(len):
            unstripped = self.only_string(data_music[i])
            string = self.only_string(data_music[i]).strip()
            # Check port
            if (string == 'zmq_addr'):
                old_address = data_music[i].split('=')[1]
                splt = old_address.rsplit(':', 1)
                base = splt[0]
                cport = port + incr
                new_address = base + ':' + str(cport)
                data_music[i] = unstripped + '=' + new_address + '\n'
                incr += 1
        return (data_music)
    
    # Calculates the simtime in the sim_params.json file
    def calc_simtime(self, data_json):
        return data_json['max_num_trs'] * data_json['max_tr_dur'] + 1000 
        

    # Checks and update,if requires, the simulation time parameter from JSON dictionary to MUSIC list
    def check_simtime(self, data_json, data_music):
        len = np.size(data_music)
        simulation_time = (data_json['simtime']) / 1000
        for i in range(len):
            string = self.only_string(data_music[i])
            ##Check stop time
            if (string == 'stoptime'):
                stoptime_m = self.only_number(data_music[i])
                if (float(stoptime_m) != simulation_time):
                    data_music[i] = string + '=' + str(simulation_time) + '\n'
                    break
        return (data_music)

    # Checks and update,if requires, the timestep parameter from JSON dictionary to MUSIC list
    def check_dt(self, data_json, data_music):
        len = np.size(data_music)
        simulation_timestep = (data_json['dt']) / 1000
        for i in range(len):
            string = self.only_string(data_music[i])
            ##Check stop time
            if (string == 'music_timestep'):
                stoptime_m = self.only_number(data_music[i])
                if (float(stoptime_m) != simulation_timestep):
                    data_music[i] = string + '=' + str(simulation_timestep) + '\n'
                    break
        return (data_music)

    # Checks and update,if requires, the parameter for number of processor for nest from JSON dictionary to MUSIC list
    def check_nest_numberOfprocessor(self, data_json, data_music):
        len = np.size(data_music)
        temp = 0
        simulation_np_nest = (data_json['np_nest'])
        for i in range(len):
            header = self.between_brackets(data_music[i])
            if (header == 'nest'):
                temp = 1
            if (temp == 1):
                string = self.only_string(data_music[i])
                if (string == '  np'):
                    number_of_processor_m = self.only_number(data_music[i])
                    temp = 0
                    if (int(number_of_processor_m) != simulation_np_nest):
                        data_music[i] = string + '=' + str(simulation_np_nest) + '\n'
                        break
        return (data_music)

    # Checks and update,if requires, the number of neurons from input_params of JSON dictionary to MUSIC list
    def check_input_num_neurons(self, data_json, data_music):

        music_string_dict = {
        'place' : f"discretize_p.out->nest.p_in_p",       
        'grid' : f"discretize_g.out->nest.p_in_g",
        'border' : f"discretize_b.out->nest.p_in_b",
        'obstacle': f"discretize_o.out->nest.p_in_o",
        }
        
        cell_data = []
        for pop in music_string_dict.keys():
            cell_line = music_string_dict[pop]
            num_cells = data_json[pop]['num_neurons']
            cell_data.append((pop, cell_line, num_cells))
        #cell_data = zip(['p', 'g', 'b'], [place_cells, grid_cells, border_cells], [n_place_cells, n_grid_cells, n_border_cells])
        
        for cell_type, cell_line, n_cells in cell_data:
            
            sensor_line = "sensor.out->discretize_{}.in[2]\n".format(cell_type[0])
            dopspike_line = f"dopspike.out->nest.p_in_dop_{cell_type[0]}[1]\n"
            lines_to_check = [sensor_line, dopspike_line, cell_line]
            full_line = f"{cell_line}[{n_cells}]\n"

            data_block_start = f"[discretize_{cell_type[0]}]\n"
            
            if n_cells > 0:
                for line in lines_to_check:
                    # determine the index of the line containing data for this cell type
                    line_index = [data_music.index(l) for l in data_music if l.find(line) != -1]
                    # if the line cannot be found then add it to the end             
                    if len(line_index) == 0:
                        if line == cell_line:
                            data_music.append(full_line)
                        else:
                            data_music.append(line)
                            
                    # else fill in the correct amount of cells for that line
                    elif len(line_index) > 0:
                        if line == cell_line:
                            data_music[line_index[0]] = full_line
                        else:
                            data_music[line_index[0]] = line
            
                # Also check for MUSIC application data block and add it if not found
                data_block_index = [data_music.index(l) for l in data_music if l.find(data_block_start) != -1]
                if len(data_block_index) == 0:
                    block_data_string = '[discretize_{}]\n  binary=discretize_adapter_pois\n  args=\n  np=1\n  music_timestep=0.0001\n  grid_positions_filename=grid_pos.json\n  representation_type={}'.format(cell_type[0], cell_type)
                    block_data_lines = [line + '\n' for line in block_data_string.split('\n')] # Keeping the file structure as a list of lines will help with sorting later

                    #Inserting at index 2 should prevent the data block from being inserted in the middle of another 
                    data_music[2:2] = block_data_lines
                
            # if there are 0 cells of a cell type then remove the last line containing the amount of these cells
            # and remove the field containing all the data about these cells
            elif n_cells == 0:
                # remove the other data by determining the the index of the data block
                # then remove the next 6 lines, which is typically the length of the block
                data_block_index = [data_music.index(l) for l in data_music if l.find(data_block_start) != -1]
                if len(data_block_index) > 0:
                    del data_music[data_block_index[0]: data_block_index[0] + 7]
                    
                # also checks for single lines that should be removed
                for line in lines_to_check:
                    line_index = [data_music.index(l) for l in data_music if l.find(line) != -1]
                    if len(line_index) > 0:
                        for line_to_del in line_index:
                            data_music.pop(line_to_del)
        
        return (data_music)
        
    # Checks and update,if requires, the number of neurons from actor_params of JSON dictionary to MUSIC list
    def check_actor_num_neurons(self, data_json, data_music):
        len = np.size(data_music)
        simulation_actor_params_num_neurons = data_json['action']['num_neurons']
        for i in range(len):
            location = data_music[i].find('vecsum.in[')
            if (location != -1):
                input_num_neurons_m = self.between_brackets(data_music[i][location:])
                if ((input_num_neurons_m.isnumeric()) == False):
                    raise TypeError("Number of actor neurons in MUSIC file is not proper..")
                data_music[i] = data_music[i][0:(location)] + 'vecsum.in[' + str(
                    simulation_actor_params_num_neurons) + ']\n'
                break
        return (data_music)

    def check_representation_type(self, data_json, data_music):
        len = np.size(data_music)
        temp = 0
        pops = ['place', 'grid', 'border', 'obstacle']
        for pop in pops:
            simulation_representation_type = data_json[pop]['representation_type']
            for i in range(len):
                header = self.between_brackets(data_music[i])
                if (header == 'discretize'):
                    temp = 1
                if (temp == 1):
                    location = data_music[i].find('  representation_type=')
                    if (location != -1):
                        data_music[i] = data_music[i][
                                        0:(location)] + '  representation_type=' + simulation_representation_type + '\n'
                        break
        return (data_music)
        


# File_operation opens, updates and closes the actual JSON and MUSIC file and convert it into python readable format
class File_operation(Check_content):
    data_json = None
    data_music = None
    data_music_old = None

    # Constructor that loads the files and convert it into python readable format for later use
    def __init__(self, file_json, file_music):
        try:
            f = open(file_json)
            print('JSON file loaded...')
        except FileNotFoundError:
            print(
                'JSON file not found. Please check the directory provided is right or if the file is in same folder as this python code')
        except:
            print('Other error regarding the JSON file')
        self.file_json = file_json
        self.data_json = json.load(f)
        f.close()
        try:
            g = open(file_music)
            print('MUSIC file read...')
        except FileNotFoundError:
            print(
                'MUSIC file not found. Please check the directory provided is right or if the file is in same folder as this python code')
        except:
            print('Other error regarding the JSON file')
        self.data_music = g.readlines()
        g.close()
        try:
            g = open(file_music)
        except FileNotFoundError:
            print(
                'MUSIC file not found. Please check the directory provided is right or if the file is in same folder as this python code')
        except:
            print('Other error regarding the JSON file')
        self.data_music_old = g.readlines()
        g.close()
        
    # All this does now is get the number of obstacles
    def get_obstacle_info(self):
        if self.data_json['environment']['obstacles']['flag']:
            return len(self.data_json['environment']['obstacles']['centers'])
        else:
            return 0
        
    def update_musicfile(self, new_file_music):
        try:
            self.data_music = self.sort_music_file(self.data_music)
            g = open(new_file_music, 'w+')
            for item in self.data_music:
                g.write(item)
            g.close()
            print('MUSIC file updated/created from the JSON file...')
        except:
            print('Error while creating/updating MUSIC file')
            print(traceback.format_exc())
    
    # Updates the JSON params
    def update_jsonfile(self):
        try:
            g = open(self.file_json, 'w+')
            json.dump(self.data_json, g, indent=2)
            print('JSON param file updated...')
        except:
            print('Error while creating/updating JSON file')
            print(traceback.format_exc())
    
    # Updates the hostfile based on number of Music devices found in the Music file
    def update_hostfile(self):
        device_count = 0
        for line in self.data_music:
            if line[0] == '[':
            #if '>' in line:
                device_count += 1
        try:
            if os.path.exists("hostfile"):
                os.remove("hostfile")
            with open("hostfile", 'w') as f:
                f.write('localhost slots={}\n'.format(device_count))
            print('hostfile updated...')
        except:
            print('error while updating hostfile')

    # This segment calls the functions of the inherited class Check_content to update the list created from the MUSIC file

    def update_simtime(self):
        self.data_music = Check_content().check_simtime(self.data_json, self.data_music)
        
    def update_simtime_json(self):
        self.data_json['simtime'] = Check_content().calc_simtime(self.data_json)
        self.update_jsonfile()

    def update_port(self, port):
        self.data_music = Check_content().check_port(self.data_json, self.data_music, port)

    def update_simdt(self):
        self.data_music = Check_content().check_dt(self.data_json, self.data_music)

    def update_nest_number_of_processor(self):
        self.data_music = Check_content().check_nest_numberOfprocessor(self.data_json, self.data_music)

    def update_input_num_neurons(self, num_obs=0):
        
        self.data_json['grid']['num_neurons'] = int(np.dot(self.data_json['grid']['cells_prop']['g_nrows'], self.data_json['grid']['cells_prop']['g_ncols']))
        if self.data_json['grid']['num_neurons'] == 0:
            self.data_json['dopamine_g']['num_neurons'] = 0
        else:
            self.data_json['dopamine_g']['num_neurons'] = 1
        
        self.data_json['place']['num_neurons'] = self.data_json['place']['cells_prop']['p_nrows'] * self.data_json['place']['cells_prop']['p_ncols']
        if self.data_json['place']['num_neurons'] == 0:
            self.data_json['dopamine_p']['num_neurons'] = 0
        else:
            self.data_json['dopamine_p']['num_neurons'] = 1
            
        if self.data_json['border']['cells_prop']['flag']:
            self.data_json['border']['num_neurons'] = 8
            self.data_json['dopamine_b']['num_neurons'] = 1
        else:
            self.data_json['border']['num_neurons'] = 0
            self.data_json['dopamine_b']['num_neurons'] = 0
            
        if num_obs > 0:
            self.data_json['obstacle']['num_neurons'] = 4 * num_obs
            self.data_json['dopamine_o']['num_neurons'] = 1
        else:
            self.data_json['obstacle']['num_neurons'] = 0
            self.data_json['dopamine_o']['num_neurons'] = 0
        self.data_music = Check_content().check_input_num_neurons(self.data_json, self.data_music)

    def update_actor_num_neurons(self):
        self.data_music = Check_content().check_actor_num_neurons(self.data_json, self.data_music)

    def update_representation_type(self):
        self.data_music = Check_content().check_representation_type(self.data_json, self.data_music)
        
    # Sorts a given music file into a format where changes are easily tracked - should maybe be static
    def sort_music_file(self, data_music):
        block1 = data_music[0:2] # The first two lines can remain fixed
        block2lines = [line for line in data_music if (line[0] == '[' or line[0] == ' ')] # List of data block lines
        # block 2 needs to preserve order of lines within the data block
        block2 = []
        for i, line in enumerate(block2lines):
            if line.startswith('['):
                subblock = [line]
                for subline in block2lines[i+1:]: # Picking the data entries for each block
                    if subline.startswith('['): # New block starts
                        break
                    elif subline.startswith('  '): # data entries start with 2 spaces
                        subblock.append(subline)
                block2.append(subblock)
        block2 = sorted(block2) # sort the block using the first entries
        block2 = [item for sublist in block2 for item in sublist] # Then flatten the list of subblocks back into a list of lines
        data_music_sorted = block1
        data_music_sorted.extend(block2)
        block3 = data_music[len(data_music_sorted):] # Get the single line entries at the end of the file
        block3 = sorted(block3)
        data_music_sorted.extend(block3)
        return data_music_sorted        

    def print_changes(self):
        d = difflib.Differ()
        result = list(d.compare(self.data_music_old, self.data_music))
        removed = [line[2:] for line in result if line.startswith('- ')]
        added = [line[2:] for line in result if line.startswith('+ ')]
        counter = max(len(removed),len(added)) # Since one change can be included in both lists, we use max instead of a sum
        in_block = False
        for line in removed:
            if line.startswith('  ') or line.startswith('['): # Groups datablocks
                in_block = True
                print(Fore.RED + line, end='')
            else:
                if in_block:
                    in_block = False
                    print() # Print newline to seperate block
                print(Fore.RED + line)
        for line in added:
            if line.startswith('  ') or line.startswith('['):
                in_block=True
                print(Fore.GREEN + line, end='')
            else:
                if in_block:
                    in_block = False
                    print() # Print newline to seperate block
                print(Fore.GREEN + line)
        print(Style.RESET_ALL) # reset colorama formatting
        return counter
