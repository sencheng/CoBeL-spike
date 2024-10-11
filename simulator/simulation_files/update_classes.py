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
import socket
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

    # Updates the 3 ports in the json and music file
    def check_port(self, data_json, data_music):
        # Devices that need a port
        ports_needed = ['CommandReceiver', 'ObservationSender', 'RewardSender']
        
        # Get unused ports by binding to port 0
        ports = []
        for _ in range(len(ports_needed)):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(('', 0))
            addr = s.getsockname()
            ports.append(addr[1])
            s.close()
        print(f'Found ports: {ports}')
        
        # Assign ports to devices in data_json
        for i, device in enumerate(ports_needed):
            data_json[device]['socket'] = ports[i]
        
        # Update ports in data_music
        for i in range(np.size(data_music)):
            string = self.only_string(data_music[i]).strip()
            if string == '[command]':
                port_string = data_music[i + 9].split(':')
                assert port_string[0].startswith('  zmq_addr')
                port_string[-1] = str(ports[0])
                data_music[i + 9] = ':'.join(port_string) + '\n'
            elif string == '[sensor]':
                port_string = data_music[i + 7].split(':')
                assert port_string[0].startswith('  zmq_addr')
                port_string[-1] = str(ports[1])
                data_music[i + 7] = ':'.join(port_string) + '\n'
            elif string == '[reward]':
                port_string = data_music[i + 7].split(':')
                assert port_string[0].startswith('  zmq_addr')
                port_string[-1] = str(ports[2])
                data_music[i + 7] = ':'.join(port_string) + '\n'
        
        return data_json, data_music
    


    # Calculates the simtime in the sim_params.json file
    def calc_simtime(self, data_json):
        return data_json['max_num_trs'] * data_json['max_tr_dur'] + 1000 
        

    # Checks and update,if requires, the simulation time parameter from JSON dictionary to MUSIC list
    def check_simtime(self, data_json, data_music):
        length = np.size(data_music)
        simulation_time = (data_json['simtime'] / 1000) + 2

        for i in range(length):
            string = self.only_string(data_music[i])
            # Check stop time
            if string == 'stoptime':
                stoptime_m = self.only_number(data_music[i])
                if float(stoptime_m) != simulation_time:
                    data_music[i] = string + '=' + str(simulation_time) + '\n'
                    break

        return data_music


    # Checks and update,if requires, the timestep parameter from JSON dictionary to MUSIC list
    def check_dt(self, data_json, data_music):
        length = np.size(data_music)
        simulation_timestep = data_json['dt'] / 1000

        for i in range(length):
            string = self.only_string(data_music[i])
            # Check music timestep
            if string == 'music_timestep':
                music_timestep = self.only_number(data_music[i])
                if float(music_timestep) != simulation_timestep:
                    data_music[i] = string + '=' + str(simulation_timestep) + '\n'
                    break

        return data_music


    # Checks and update,if requires, the parameter for number of processor for nest from JSON dictionary to MUSIC list
    def check_nest_numberOfprocessor(self, data_json, data_music):
        length = np.size(data_music)
        simulation_np_nest = data_json['np_nest']
        temp = False

        for i in range(length):
            header = self.between_brackets(data_music[i])
            if header == 'nest':
                temp = True
            if temp:
                string = self.only_string(data_music[i])
                if string == '  np':
                    number_of_processor_m = self.only_number(data_music[i])
                    temp = False
                    if int(number_of_processor_m) != simulation_np_nest:
                        data_music[i] = string + '=' + str(simulation_np_nest) + '\n'
                        break
        return data_music


    # Checks and update,if requires, the number of neurons from input_params of JSON dictionary to MUSIC list
    def check_input_num_neurons(self, data_json, data_music):
        music_string_dict = {
            'place': "discretize_p.out->nest.p_in_p",
            'grid': "discretize_g.out->nest.p_in_g",
            'border': "discretize_b.out->nest.p_in_b",
            'obstacle': "discretize_o.out->nest.p_in_o",
            'noise': "discretize_n.out->nest.p_in_n",
        }

        cell_data = []
        for pop in music_string_dict.keys():
            cell_line = music_string_dict[pop]
            num_cells = data_json[pop]['num_neurons']
            cell_data.append((pop, cell_line, num_cells))

        for cell_type, cell_line, n_cells in cell_data:
            sensor_line = f"sensor.out->discretize_{cell_type[0]}.in[4]\n"
            dopspike_line = f"dopspike.out->nest.p_in_dop_{cell_type[0]}[1]\n"
            lines_to_check = [sensor_line, dopspike_line, cell_line]
            full_line = f"{cell_line}[{n_cells}]\n"
            data_block_start = f"[discretize_{cell_type[0]}]\n"

            if n_cells > 0:
                for line in lines_to_check:
                    # Determine the index of the line containing data for this cell type
                    line_index = [data_music.index(l) for l in data_music if l.find(line) != -1]
                    
                    # If the line cannot be found, add it to the end
                    if not line_index:
                        if line == cell_line:
                            data_music.append(full_line)
                        else:
                            data_music.append(line)
                    
                    # Else, fill in the correct number of cells for that line
                    else:
                        if line == cell_line:
                            data_music[line_index[0]] = full_line
                        else:
                            data_music[line_index[0]] = line

                # Check for MUSIC application data block and add it if not found
                data_block_index = [data_music.index(l) for l in data_music if l.find(data_block_start) != -1]
                if not data_block_index:
                    block_data_string = (
                        f"[discretize_{cell_type[0]}]\n"
                        "  binary=discretize_adapter_pois\n"
                        "  args=\n"
                        "  np=1\n"
                        "  music_timestep=0.0001\n"
                        "  grid_positions_filename=grid_pos.json\n"
                        f"  representation_type={cell_type}"
                    )
                    block_data_lines = [line + '\n' for line in block_data_string.split('\n')]
                    
                    # Inserting at index 2 to avoid disrupting existing blocks
                    data_music[2:2] = block_data_lines

            # Remove lines and data blocks for cell types with 0 neurons
            elif n_cells == 0:
                # Remove data block
                data_block_index = [data_music.index(l) for l in data_music if l.find(data_block_start) != -1]
                if data_block_index:
                    del data_music[data_block_index[0]: data_block_index[0] + 7]

                # Remove associated lines
                for line in lines_to_check:
                    line_index = [data_music.index(l) for l in data_music if l.find(line) != -1]
                    if line_index:
                        for line_to_del in line_index:
                            data_music.pop(line_to_del)

        return data_music



    # Checks and update,if requires, the number of neurons from actor_params of JSON dictionary to MUSIC list
    def check_actor_num_neurons(self, data_json, data_music):
        length = np.size(data_music)
        simulation_actor_params_num_neurons = data_json['action']['num_neurons']
        
        for i in range(length):
            location = data_music[i].find('vecsum.in[')
            if location != -1:
                input_num_neurons_m = self.between_brackets(data_music[i][location:])
                if not input_num_neurons_m.isnumeric():
                    raise TypeError("Number of actor neurons in MUSIC file is not proper.")
                
                data_music[i] = (data_music[i][:location] + 
                                f'vecsum.in[{simulation_actor_params_num_neurons}]\n')
                break
                
        return data_music


    def check_representation_type(self, data_json, data_music):
        length = np.size(data_music)
        pops = ['place', 'grid', 'border', 'obstacle']
        
        for pop in pops:
            simulation_representation_type = data_json[pop]['representation_type']
            temp = False
            
            for i in range(length):
                header = self.between_brackets(data_music[i])
                if header == 'discretize':
                    temp = True
                
                if temp:
                    location = data_music[i].find('  representation_type=')
                    if location != -1:
                        data_music[i] = (
                            data_music[i][:location] + 
                            f'  representation_type={simulation_representation_type}\n'
                        )
                        break

        return data_music

        


# File_operation opens, updates and closes the actual JSON and MUSIC file and convert it into python readable format
class File_operation(Check_content):
    data_json = None
    data_music = None
    data_music_old = None

    # Constructor that loads the files and convert it into python readable format for later use
    def __init__(self, file_json, file_music, env_file='parameter_sets/current_parameter/env_params.json'):
        try:
            with open(file_json) as f:
                print('JSON file loaded...')
                self.data_json = json.load(f)
        except FileNotFoundError:
            print('JSON file not found. Please check the directory or ensure the file is in the same folder as this Python code.')
        except Exception as e:
            print(f'An error occurred while loading the JSON file: {e}')

        self.file_json = file_json

        try:
            with open(file_music) as g:
                print('MUSIC file read...')
                self.data_music = g.readlines()
        except FileNotFoundError:
            print('MUSIC file not found. Please check the directory or ensure the file is in the same folder as this Python code.')
        except Exception as e:
            print(f'An error occurred while loading the MUSIC file: {e}')

        try:
            with open(file_music) as g:
                self.data_music_old = g.readlines()
        except FileNotFoundError:
            print('MUSIC file not found. Please check the directory or ensure the file is in the same folder as this Python code.')
        except Exception as e:
            print(f'An error occurred while loading the MUSIC file: {e}')

        try:
            with open(env_file, 'r') as fl:
                env_dict = json.load(fl)
                self.sim_env = env_dict['sim_env']
                self.num_obs = len(env_dict['environment']['obstacles']['centers']) if env_dict['environment']['obstacles']['flag'] else 0
        except FileNotFoundError:
            print(f'Environment file not found: {env_file}')
        except Exception as e:
            print(f'An error occurred while loading the environment file: {e}')
    

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
                device_count += 1
        
        try:
            if os.path.exists("hostfile"):
                os.remove("hostfile")
            with open("hostfile", 'w') as f:
                f.write('localhost slots={}\n'.format(device_count))
            print('hostfile updated...')
        except:
            print('Error while updating hostfile')


    # This segment calls the functions of the inherited class Check_content to update the list created from the MUSIC file
    def update_simtime(self):
        self.data_music = Check_content().check_simtime(self.data_json, self.data_music)
        
    def update_simtime_json(self):
        self.data_json['simtime'] = Check_content().stoptime(self.data_json)
        self.update_jsonfile()

    def update_ports(self):
        self.data_json, self.data_music = Check_content().check_port(self.data_json, self.data_music)

    def update_simdt(self):
        self.data_music = Check_content().check_dt(self.data_json, self.data_music)

    def update_nest_number_of_processor(self):
        self.data_music = Check_content().check_nest_numberOfprocessor(self.data_json, self.data_music)

    def update_input_num_neurons(self):
        self.data_json['grid']['num_neurons'] = int(
            np.dot(self.data_json['grid']['cells_prop']['g_nrows'], 
                self.data_json['grid']['cells_prop']['g_ncols'])
        )
        if self.data_json['grid']['num_neurons'] == 0:
            self.data_json['dopamine_g']['num_neurons'] = 0
        else:
            self.data_json['dopamine_g']['num_neurons'] = 1

        self.data_json['place']['num_neurons'] = (
            self.data_json['place']['cells_prop']['p_nrows'] *
            self.data_json['place']['cells_prop']['p_ncols']
        )
        if self.data_json['place']['num_neurons'] == 0:
            self.data_json['dopamine_p']['num_neurons'] = 0
        else:
            self.data_json['dopamine_p']['num_neurons'] = 1

        if self.data_json['border']['cells_prop']['flag']:
            if self.sim_env == 'openfield':
                self.data_json['border']['num_neurons'] = 8
            elif self.sim_env == 'tmaze':
                self.data_json['border']['num_neurons'] = 14
            self.data_json['dopamine_b']['num_neurons'] = 1
        else:
            self.data_json['border']['num_neurons'] = 0
            self.data_json['dopamine_b']['num_neurons'] = 0

        if self.num_obs > 0:
            self.data_json['obstacle']['num_neurons'] = 4 * self.num_obs
            self.data_json['dopamine_o']['num_neurons'] = 1
        else:
            self.data_json['obstacle']['num_neurons'] = 0
            self.data_json['dopamine_o']['num_neurons'] = 0

        if self.data_json['noise']['num_neurons'] == 0:
            self.data_json['dopamine_n']['num_neurons'] = 0
        else:
            self.data_json['dopamine_n']['num_neurons'] = 1

        self.data_music = Check_content().check_input_num_neurons(self.data_json, self.data_music)


    def update_actor_num_neurons(self):
        self.data_music = Check_content().check_actor_num_neurons(self.data_json, self.data_music)


    def update_representation_type(self):
        self.data_music = Check_content().check_representation_type(self.data_json, self.data_music)
        
        
    # Sorts a given music file into a format where changes are easily tracked - should maybe be static
    def sort_music_file(self, data_music):
        block1 = data_music[0:2]  # The first two lines can remain fixed

        # List of data block lines
        block2lines = [line for line in data_music if line[0] == '[' or line[0] == ' ']

        # block2 needs to preserve the order of lines within each data block
        block2 = []
        for i, line in enumerate(block2lines):
            if line.startswith('['):
                subblock = [line]
                for subline in block2lines[i+1:]:  # Picking the data entries for each block
                    if subline.startswith('['):  # New block starts
                        break
                    elif subline.startswith('  '):  # data entries start with 2 spaces
                        subblock.append(subline)
                block2.append(subblock)

        # Sort the block using the first entries
        block2 = sorted(block2)
        
        # Flatten the list of subblocks back into a list of lines
        block2 = [item for sublist in block2 for item in sublist]

        # Combine the sorted blocks with the initial fixed block
        data_music_sorted = block1
        data_music_sorted.extend(block2)

        # Get the single line entries at the end of the file and sort them
        block3 = data_music[len(data_music_sorted):]
        block3 = sorted(block3)
        data_music_sorted.extend(block3)

        return data_music_sorted
        
    
    def print_changes(self):
        d = difflib.Differ()
        result = list(d.compare(self.data_music_old, self.data_music))
        removed = [line[2:] for line in result if line.startswith('- ')]
        added = [line[2:] for line in result if line.startswith('+ ')]
        
        # Since one change can be included in both lists, we use max instead of sum
        counter = max(len(removed), len(added))  
        
        in_block = False
        for line in removed:
            if line.startswith('  ') or line.startswith('['):  # Group data blocks
                in_block = True
                print(Fore.RED + line, end='')
            else:
                if in_block:
                    in_block = False
                    print()  # Print newline to separate block
                print(Fore.RED + line)
        
        for line in added:
            if line.startswith('  ') or line.startswith('['):
                in_block = True
                print(Fore.GREEN + line, end='')
            else:
                if in_block:
                    in_block = False
                    print()  # Print newline to separate block
                print(Fore.GREEN + line)
        
        print(Style.RESET_ALL)  # Reset colorama formatting
        return counter