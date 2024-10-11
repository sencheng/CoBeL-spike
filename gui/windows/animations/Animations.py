from PyQt5.QtWidgets import QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import socket
import threading
import pickle
import os
import time
import numpy as np
import pandas as pd
import math


# IMPORTANT: only working with openfield
class TrajectoryAnimation(QWidget):
    def __init__(self, x_min, y_min, x_max, y_max):
        super().__init__()

        # Create the figure
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Trajectory")
        
        # Create the Widget where the figure lays
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        # Initialize variables for the animation
        self.x_data = []
        self.y_data = []
        self.trajectory, = self.ax.plot([], [], lw=2, color="blue")
        self.start_position, = self.ax.plot([], [], color="green", marker='o')
        self.current_position, = self.ax.plot([], [], color="red", marker='o')
        self.animation = None
        self.frame = 0
        
        # Plot setup
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_aspect('equal', adjustable='box')
        
        # Container and variables for data received from simulation
        self.x_container = []
        self.y_container = []
        self.current_trial = 1
        self.prev_trial = 1
        self.trial_change = True
        self.current_time = 0.0
        
        # Create socket object
        self.host = socket.gethostname()
        self.port = 41111
        
        self.ports_opened = False
        self.animation_running = False
    
    
    def start_listening_to_port(self):
        # Start listening to port for data
        self.port_thread = threading.Thread(target=self.listen_to_port)
        self.port_thread.daemon = True
        self.port_thread.start()
        
    
    def stop_listening(self):
        self.listening = False
        self.close_server_socket()
        self.close_connections()
        self.port_thread.join()
    
    
    def connect(self):
        print("[INFO] Opening port")

        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # Enable SO_REUSEADDR option
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)  # Start listening for connections
            
            # Accept a connection
            self.c, _ = self.server_socket.accept()
            self.c.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            print("[INFO] Connection accepted")
            self.ports_opened = True  # Set flag to True when port is successfully opened
        except socket.error as e:
            print(f"[ERROR] Failed to open port: {e}")
            self.ports_opened = False
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
            self.ports_opened = False
    
    
    def close_server_socket(self):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        client_socket.settimeout(5)
        try:
            client_socket.connect((self.host, self.port))
        except socket.error as e:
            print(f"[CONNECTION NOT POSSIBLE] {e}")
            self.connected = False
        
        try:
            client_socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
    
    
    def close_connections(self):
        print("[INFO] Connection closed")
        if self.ports_opened:
            self.c.close() # Close connection
            self.server_socket.close() # Close server socket
            self.ports_opened = False
    
    
    def listen_to_port(self):
        self.listening = True
        self.connect()
        
        while self.listening: 
            try:
                # If the socket is ready for reading, receive data
                data = self.c.recv(4096)
                
                # Deserialize data
                self.prev_trial = self.current_trial
                self.current_trial, self.current_time, x_pos, y_pos = list(pickle.loads(data))
                self.x_container.append(x_pos)
                self.y_container.append(y_pos)
                
                if self.prev_trial != self.current_trial:
                    self.x_data = []
                    self.y_data = []
                    self.trial_change = True
                    
            except Exception as e:
                # Handle other exceptions
                print(f"[ERROR] {e}")
                time.sleep(1.0)
                self.close_connections()
                self.connect()
        
        print("[INFO] Listening to port stopped")
        
        
    def start_animation(self):
        self.animation = FuncAnimation(self.figure, self.update_plot, interval=100)
        self.animation_running = True


    def stop_animation(self):
        self.animation.event_source.stop()
        #self.animation = None
    
    
    def toggle_animation(self):
        if self.animation_running:
            self.animation.event_source.stop()
            self.animation_running = False
        else:
            self.animation.event_source.start()
            self.animation_running = True
    
    
    def update_plot(self, _):
        self.x_data.extend(self.x_container.copy())
        self.x_container = []

        self.y_data.extend(self.y_container.copy())
        self.y_container = []
        
        if self.current_trial <= self.max_num_trial:
            self.trajectory.set_data(self.x_data, self.y_data)
            if self.x_data != []:
                self.start_position.set_data(self.x_data[0], self.y_data[0])
                self.current_position.set_data(self.x_data[-1], self.y_data[-1])
        else:
            self.trajectory.set_data([], [])
            self.start_position.set_data([], [])
            self.current_position.set_data([], [])
            
        if self.trial_change:
            self.add_goal_zone(self.ax, self.current_trial)
            self.trial_change = False
        
        return self.trajectory, self.start_position, self.current_position


    def set_trial_data(self, trial_data):
        self.trial_data = trial_data
        self.max_num_trial = self.trial_data["trial_num"].to_numpy()[-1]
        self.goal_zones = []

    
    def add_goal_zone(self, ax, tr, c='purple'):   
        while self.goal_zones != []:
            goal_zone = self.goal_zones.pop()
            goal_zone.remove()
        
        tr_reward_dict,all_reward_dict = self.tr_reward(tr,self.trial_data)
        
        if tr_reward_dict == 0:
            return 
        
        if (tr_reward_dict['goal_shape'] == 'round'):
            goal_zone = plt.Circle((tr_reward_dict['goal_x'], 
                                    tr_reward_dict['goal_y']),
                                    tr_reward_dict['goal_size1'], 
                                    color=c, alpha=1, fill=False,
                                    linewidth=4,
                                    label='current goal zone')
            self.goal_zones.append(ax.add_patch(goal_zone))
            
        elif (tr_reward_dict['goal_shape'] == 'rect'):
            goal_x = tr_reward_dict['goal_x']
            goal_y = tr_reward_dict['goal_y']
            delta_x = tr_reward_dict['goal_size1']
            delta_y = tr_reward_dict['goal_size2']
            
            ll = (goal_x - delta_x, goal_y - delta_y) # lower left
            lr = (goal_x + delta_x, goal_y - delta_y) # lower right
            ur = (goal_x + delta_x, goal_y + delta_y) # upper right
            ul = (goal_x - delta_x, goal_y + delta_y) # upper left
                
            goal_zone = plt.Polygon([ll, lr, ur, ul], closed=True, color=c, alpha=0.1, label='Goal zone')
            self.goal_zones.append(ax.add_patch(goal_zone))
            
        for i in range(len(all_reward_dict['goal_x'])):
            if all_reward_dict['goal_shape'][i] == 'round':
                goal_zone = plt.Circle((all_reward_dict['goal_x'][i], 
                                        all_reward_dict['goal_y'][i]),
                                        all_reward_dict['goal_size1'][i], 
                                        color=c, alpha=1, fill=False, 
                                        linewidth = 2,
                                        linestyle='--',
                                        label='previous goal zone' if i == 0 else None
                                        )
            elif all_reward_dict['goal_shape'][i] == 'rect':
                goal_x = all_reward_dict['goal_x'][i]
                goal_y = all_reward_dict['goal_y'][i]
                delta_x = all_reward_dict['goal_size1'][i]
                delta_y = all_reward_dict['goal_size2'][i]
                
                ll = (goal_x - delta_x, goal_y - delta_y) # lower left
                lr = (goal_x + delta_x, goal_y - delta_y) # lower right
                ur = (goal_x + delta_x, goal_y + delta_y) # upper right
                ul = (goal_x - delta_x, goal_y + delta_y) # upper left
                                
                goal_zone = plt.Polygon([ll, lr, ur, ul], closed=True, fill=False, color=c, alpha=0.1, label=None)
            self.goal_zones.append(ax.add_patch(goal_zone))

    
    def tr_reward(self, tr,trials_params):
        if tr > self.max_num_trial:
            return 0, 0
        
        trial_dummy = trials_params.loc[trials_params['trial_num']==tr]
        
        start_x = trial_dummy['start_x'].values[0]
        start_y = trial_dummy['start_y'].values[0]
        
        goal_x = trial_dummy['goal_x'].values[0]
        goal_y = trial_dummy['goal_y'].values[0]    

        goal_shape = trial_dummy['goal_shape'].values[0]   
        goal_size1 = trial_dummy['goal_size1'].values[0]
        goal_size2 = trial_dummy['goal_size2'].values[0]
        
        tr_reward_dict = {
            'start_x' : start_x,
            'start_y' : start_y,
            'goal_x': goal_x,
            'goal_y': goal_y,
            'goal_shape': goal_shape,
            'goal_size1': goal_size1,
            'goal_size2': goal_size2,
        }

        trials_params = trials_params.loc[trials_params['trial_num']<=tr]    
        trials_params = trials_params.drop(['trial_num'],axis=1).drop_duplicates()
        
        start_x = trials_params['start_x'].to_numpy()
        start_y = trials_params['start_y'].to_numpy()

        
        goal_x = trials_params['goal_x'].to_numpy()
        goal_y = trials_params['goal_y'].to_numpy()
        goal_shape = trials_params['goal_shape'].to_numpy()
        goal_size1 = trials_params['goal_size1'].to_numpy()
        goal_size2 = trials_params['goal_size2'].to_numpy()
        
        all_reward_dict = {'start_x' : start_x,
                        'start_y' : start_y,
                        'goal_x': goal_x,
                        'goal_y': goal_y,
                        'goal_shape': goal_shape,
                        'goal_size1': goal_size1,
                        'goal_size2': goal_size2,
                        }    


        return tr_reward_dict,all_reward_dict
    

    def get_current_time(self):
        if hasattr(self, "current_time"):
            return self.current_time, self.current_trial
        else:
            return 0, self.current_trial


class FiringRateAnimation(QWidget):
    def __init__(self, neuron_type: str, start_seed: int, end_seed: int, max_sim_time: float, current_sim_time_getter, max_trial_num: int, update_rate, x_axis_interval, bin_size) -> None:
        super().__init__()
        
        # default values
        self.plot_update_interval_in_ms = update_rate
        self.bin_size_in_ms = bin_size
        self.x_axis_interval_in_ms = x_axis_interval
        self.neuron_type = neuron_type
        
                
        # Create the figure
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_visible(False)
        self.ax.set_ylabel('neuron IDs')
        self.ax.set_xlabel('sim time in ms')
        
        # Create the Widget where the figure lays
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        
        self.max_sim_time = max_sim_time
        self.start_seed = start_seed
        self.end_seed = end_seed
        self.current_seed = start_seed
        self.current_sim_time_getter = current_sim_time_getter
        self.max_trial_num = max_trial_num
        self.plot_title = f"{neuron_type} cells"
        
        
        self.current_trial_number = 1
        self.wait_for_file_time = 0.5 # time that process sleeps until it tries again if a file does not exist
        self.current_data_folder = None
        
        self.listening = False
        self.animating = False
        self.initialized = False
        self.new_seed = False
        
        
        
    
    def set_data_folder(self, path):
        # new simulation was launched
        self.current_data_folder = path
        self.update_file_paths()
    
    
    def set_network_dict(self, network_dict):
        try:
            max_fire_rate = network_dict[self.neuron_type]["cells_prop"]["max_fr"]
        except KeyError:
            max_fire_rate = 300
            print(f"[WARNING] {self.neuron_type} cell does not have a max_fr entry in the network dict. Defaulting to {max_fire_rate}")
        if isinstance(max_fire_rate, list):
            max_fire_rate = max_fire_rate[0]
        self.max_fire_rate = max_fire_rate
        
    
    def update_file_paths(self):
        # the action cells population file has a different name than the others 
        if self.neuron_type == "action":
            pop_file_suffix = "ID_dir.dat"
        else:
            pop_file_suffix = "ID_pos.dat"
        self.pop_filepath = os.path.join(self.current_data_folder, f"agent{self.current_seed}", f"{self.neuron_type}{pop_file_suffix}")
        self.data_filepath = os.path.join(self.current_data_folder, f"agent{self.current_seed}", f"{self.neuron_type}-0.gdf")
    
    
    def start(self, data_folder_path, seed=None, trial=None):
        self.set_data_folder(data_folder_path)
        if trial:
            self.current_trial_number = trial
            
        if seed:
            self.current_seed = seed
            self.update_file_paths()
            
        # create Animation object or continue animation
        self.animating = True
        self.ax.set_visible(True)
        if hasattr(self, "animation"):
            self.animation.resume()
        else:
            self.animation = FuncAnimation(self.figure, self.update_plot, interval=self.plot_update_interval_in_ms)
            
            
        
        
        # create new thread that reads spikes from file in background
        self.listening = True
        self.file_observe_thread = threading.Thread(target=self.observe_file)
        self.file_observe_thread.daemon = True
        self.file_observe_thread.start()
    
    
    def stop(self):
        # notice animation thread, it checks this variable and deactivates the animation
        self.animating = False
        
        # notice listening thread to terminate by setting variable and wait for it to terminate
        if self.listening:
            self.listening = False
            self.file_observe_thread.join()
            if hasattr(self, "file"):
                self.file.close()
        
        
    def initialize_data_structure(self):
        self.initialized = False
        while True:
            if not self.listening:
                return
            try:
                pop_data = pd.read_csv(self.pop_filepath, sep='\t', index_col=False, header=0)
                print(f"[INFO] read population file {self.pop_filepath}")
                break
            except FileNotFoundError:
                print(f"[WARNING] file \"{self.pop_filepath}\" does not exist (yet). Trying again in {self.wait_for_file_time} seconds.")
                time.sleep(self.wait_for_file_time)
                continue
        
        self.neuron_id_offset = pop_data.id.min()
        self.neuron_ids = np.array(pop_data.id) - self.neuron_id_offset
        
        
        time_vector = np.arange(start=0., stop=self.max_sim_time, step=self.bin_size_in_ms)
        self.firing_rate_bins = np.zeros(shape=(pop_data.id.max() - pop_data.id.min() + 1, time_vector.size - 1), dtype=np.uint16)
        self.hist_edges = (time_vector[:-1] + time_vector[1:]) / 2
        self.initialized = True
        
    
                
    def observe_file(self):
        while True:
            # initialize bins
            self.initialize_data_structure()
            
            # open file
            while True:
                if not self.listening:
                    return
                try:
                    self.file = open(self.data_filepath, "r")
                    print(f"[INFO] observing file {self.data_filepath}")
                    
                    break
                except(FileNotFoundError):
                    print(f"[WARNING] file \"{self.data_filepath}\" does not exist (yet). Trying again in {self.wait_for_file_time} seconds.")
                    time.sleep(self.wait_for_file_time)
                    continue
            
            # read file
            self.observations = 0
            while True:
                
                # check if the animation was stopped
                if not self.listening:
                    return
                
                # check if a new simulation instance started
                if self.new_seed:
                    self.file.close()
                    self.new_seed = False
                    break
                
                # read next line in file
                line = self.file.readline() # example line: '266\t179.200\t'
                
                # split string into list
                entries = line.split("\t") # example entry: ['266', '179.200', '']
                
                if len(entries) >= 2:   # is not always two because there is an empty string element at index 2 sometimes
                    # save data
                    try:
                        neuron_id = int(entries[0])
                        neuron_id = neuron_id - self.neuron_id_offset
                        
                        t_spike = float(entries[1])
                        bin = int(math.floor(t_spike / self.bin_size_in_ms))
                        
                        self.firing_rate_bins[neuron_id, bin] += 1
                        self.observations += 1
                    except ValueError:
                        pass
                    except IndexError:
                        # sometimes the simulation exceeds the maximum simulation time. Those data points should be dropped
                        pass
                else:
                    # reached EOF: wait until nest writes new data
                    time.sleep(self.plot_update_interval_in_ms / 1000 / 2)
                    continue
        
        
        
    def update_plot(self, frame):
        
        # clear ax
        if hasattr(self, "colorbar"):
            try:
                self.colorbar.remove()
            except ValueError:
                pass

        if hasattr(self, "colorplot"):
            try:
                self.colorplot.remove()
            except ValueError:
                pass
        
        
        # check if Initialized
        if not self.initialized:
            return self.ax
        
        if not self.animating:
            self.animation.pause()
            self.ax.set_visible(False)
            return self.ax
        
        
        current_sim_time, new_trial_number = self.current_sim_time_getter()
        
        
        if new_trial_number > self.max_trial_num:
            self.current_trial_number = new_trial_number
            self.ax.set_title(f"simulation is in buffer state")
            
            # simulation is in buffer state
            return self.ax
            
            
        
        if new_trial_number < self.current_trial_number:
            # the trial number was reseted, therefore a new simulation was launched 
            # therefore the seed needs to be incremented
            if self.current_seed + 1 > self.end_seed:
                # simulation is finished, wait for main gui to start new simulation
                pass
            else:
                self.current_seed += 1
                self.update_file_paths()
                self.new_seed = True
                self.current_trial_number = new_trial_number
            
            return self.ax
        
        self.current_trial_number = new_trial_number
        
        
        # convert to ms
        current_sim_time_in_ms = current_sim_time * 1000
        
        # calculate indices of bins to be plotted [a; b]
        b = current_sim_time_in_ms
        a = max(0, current_sim_time_in_ms - self.x_axis_interval_in_ms)
        b_idx = int(math.ceil(b / self.bin_size_in_ms))
        a_idx = int(math.ceil(a / self.bin_size_in_ms))
        
        
        if a_idx == b_idx:
            return self.ax
        
        # print(f"hist: {self.hist_edges[a_idx:b_idx].shape}")
        # print(f"neuron_ids: {self.neuron_ids.shape}")
        # print(f"fr: {self.firing_rate_bins[:, a_idx:b_idx].shape}")
        
        self.colorplot = self.ax.pcolor(self.hist_edges[a_idx:b_idx], self.neuron_ids, self.firing_rate_bins[:, a_idx:b_idx] * 1000 / self.bin_size_in_ms, edgecolors='none', vmin=0, vmax=self.max_fire_rate)
        # max_fr = np.max(self.firing_rate_bins[:, a_idx:b_idx] * 1000 / self.bin_size_in_ms)
        
        self.colorbar = self.figure.colorbar(self.colorplot)
        self.ax.set_xlim([a_idx * self.bin_size_in_ms, b_idx * self.bin_size_in_ms])
        self.ax.set_ylim(self.neuron_ids.min() - 0.5, self.neuron_ids.max() + 0.5)
        self.ax.set_title(self.plot_title)
                
        return self.ax

