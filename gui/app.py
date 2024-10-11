from PyQt5.QtWidgets import QWidget, QHBoxLayout, QMainWindow, QAction, QStackedLayout, QApplication, QMessageBox
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QThread, QObject, pyqtSignal as Signal
import subprocess
import threading
import time
import signal
import os
import sys
import json
import re
import numpy as np



import windows.models.openfield_parameter_model as OpenfieldParametersModel
from windows.analysis import analysis_view
from windows.network import network_view
from windows.environment import environment_view
from windows.simulation import sim_view, sim_controller
from windows.parameter_window import ParameterSection, StartSection
from windows.animation_window import AnimationWindow, BreakSection
from windows.animation_settings_window import GUISettingsPopUpWindow
from windows.animations.Animations import TrajectoryAnimation, FiringRateAnimation
from stylesheet import stylesheet

class Worker(QObject):
    
    new_network_dict_signal = Signal()

    def update_animations(self):
        self.new_network_dict_signal.emit()

class MainWindow(QMainWindow):
    work_requested = Signal()
    def __init__(self):
        super().__init__()
        
        # qthread is needed as a bridge between normal threads and the Qtimers of the animation
        self.worker = Worker()
        self.worker_thread = QThread()

        self.worker.new_network_dict_signal.connect(self.update_animations)
        
        self.work_requested.connect(self.worker.update_animations)

        # move worker to the worker thread
        self.worker.moveToThread(self.worker_thread)

        # start the thread
        self.worker_thread.start()
        
        self.data_path = "../data/"
        self.folder_name = "run_"
        self.gui_settings_json_path = "../gui/gui_animation_settings.json"
        self.original_gui_settings_json_path = "../gui/original_gui_animation_settings.json"
        self.show_animation = True
        self.update_sim_information_interval = 1.0 / 3
        
        
        self.menuBarMenus = []
        
        self.setWindowTitle('CoBeL-spike')
        
        # Parameter view construction
        self.ParameterSection = ParameterSection()
        self.startSection = StartSection()
        self.startSection.addEventToButton(self.setAnimationWindow)
        self.parameterView = QHBoxLayout()
        self.parameterView.addWidget(self.ParameterSection)
        self.parameterView.addWidget(self.startSection)
        self.parameterWidget = QWidget()
        self.parameterWidget.setLayout(self.parameterView)

        #Create Openfield Params Model
        self.openfield_params_model = OpenfieldParametersModel.OpenfieldParams()
        
        # Create simulation params in the parameter view
        self.simulation_params_view = sim_view.SimulationParameterWidget()
        self.simulation_controller = sim_controller.SimulationController(self.simulation_params_view)
        self.ParameterSection.addContentToSimulation(self.simulation_params_view)
        
        # Create network params in the parameter view
        self.network_params_view = network_view.NetworkParameterWidget(self.openfield_params_model)
        self.ParameterSection.addContentToNetwork(self.network_params_view)
        
        # Create environment params in paramter view
        self.environment_params_view = environment_view.EnvironmentParameterWidget(self.openfield_params_model, self.simulation_controller)
        self.ParameterSection.addContentToEnvironment(self.environment_params_view)
        
        # Create analysis params in parameter view
        self.analyis_params_view = analysis_view.AnalysisParameterWidget()
        self.ParameterSection.addcontentToAnalysis(self.analyis_params_view)
        
        
        # Animation view construction
        self.animationWindow = AnimationWindow()
        self.animationWindow.addEventToButton(self.toggleAnimation)
        self.breakSection = BreakSection()
        self.breakSection.addEventToButton(self.setParameterSection)
        self.animationView = QHBoxLayout()
        self.animationView.addWidget(self.animationWindow)
        self.animationView.addWidget(self.breakSection)
        self.animationWidget = QWidget()
        self.animationWidget.setLayout(self.animationView)
        
        # Putting both views together
        self.main_layout = QStackedLayout()
        self.main_layout.addWidget(self.parameterWidget)
        self.main_layout.addWidget(self.animationWidget)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)  
        self.central_widget.setLayout(self.main_layout)
         
        self.setParameterMenuBar()
        
        self.resize(1400, 800)
        self.show()
        
    
    
    def closeEvent(self, event):
        # check if simulation is running
        if hasattr(self, "simulation_process"):
            if self.simulation_process.poll() is None:
                # experiment is still running
                reply = QMessageBox.question(
                    self,
                    "Confirmation",
                    "Closing the window will stop the running simulation. Are you sure you want to continue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    os.killpg(os.getpgid(self.simulation_process.pid), signal.SIGKILL)
                    event.accept()  # Accept the close event
                else:
                    event.ignore()  # Ignore the close event
        
        else:
            event.accept()
            
    
    
    def update_animations(self):
        self.stop_all_fr_animations()
        self.update_active_fr_animations(self.current_network_dict)
        self.start_fr_animations(self.current_network_dict)
    
    
    def update_animations(self):
        self.stop_all_fr_animations()
        self.update_active_fr_animations(self.current_network_dict)
        self.start_fr_animations(self.current_network_dict)
    
    
    def setParameterSection(self):
        self.main_layout.setCurrentIndex(0)
        self.removeMenuBar()
        self.setParameterMenuBar()
        
        self.trajectory_animation.stop_animation()
        self.trajectory_animation.stop_listening()
        del self.trajectory_animation
        
        self.stop_all_fr_animations(delete=True)
                
        # reset animation pause button
        self.show_animation = True
        self.animationWindow.nameButton("Deactivate animation")
        
        self.stopListening()
        if self.simulation_thread_run:       
            self.simulation_thread_run = False
            os.killpg(os.getpgid(self.simulation_process.pid), signal.SIGKILL)
            
        
    def setAnimationWindow(self):
        # Check the analysis file for type safety
        if self.analyis_params_view.checkFile():
            return

        trial_params, self.sim_params = self.simulation_controller.generate_sim_param_files(False)
        self.network_dicts = self.openfield_params_model.get_network_dicts().copy()
        env_params = self.openfield_params_model.get_env_params()

        # Save environment params and analysis config since they won't change across simulations
        saveFile(env_params, "parameter_sets/current_parameter/env_params.json")
        saveFile(self.analyis_params_view.file, "parameter_sets/current_parameter/analysis_config.json")
        trial_params.to_csv('parameter_sets/current_parameter/trials_params.dat', sep="\t", index=False)

        # check the content in the text boxes
        self.startSeed = self.startSection.readStart()
        self.endSeed = self.startSection.readEnd()

        # Check the seeds for type safety and convert them
        if not self.startSection.checkSeed(self.startSeed, "The start seed must be an integer"):
            return
        if not self.startSection.checkSeed(self.endSeed, "The end seed must be an integer"):
            return
        
        self.startSeed, self.endSeed = int(self.startSeed), int(self.endSeed)
        
        if not self.startSection.checkSeedRatio(self.startSeed, self.endSeed, "The start seed must be smaller than the end seed"):
            return

        # Clear the parameter window
        self.startSection.clearStart()
        self.startSection.clearEnd()

        # Prepare animation window
        self.breakSection.nameButton('Break')
        self.breakSection.clearInfo()

        # Change to the animation window
        self.main_layout.setCurrentIndex(1)

        # Set the menu bar for animation
        self.removeMenuBar()
        self.setAnimationMenuBar()

        self.folder_num = self.generateDataFolderNum()

        x_min = env_params["environment"]["openfield"]["xmin_position"]
        y_min = env_params["environment"]["openfield"]["ymin_position"]
        x_max = env_params["environment"]["openfield"]["xmax_position"]
        y_max = env_params["environment"]["openfield"]["ymax_position"]
        
        # Set the trajectory animation
        self.trajectory_animation = TrajectoryAnimation(x_min, y_min, x_max, y_max)
        self.trajectory_animation.set_trial_data(trial_params)
        self.animationWindow.addContentToAnimation(self.trajectory_animation, 0, 0)
        self.trajectory_animation.start_listening_to_port()
        self.trajectory_animation.start_animation()
        
        # set callback for current sim time and seed
        self.current_sim_time_getter = self.trajectory_animation.get_current_time
        
        # initialize the firing rate animations
        self.initialize_firing_rate_animations(trial_params)
        
        self.startSimulationThread()
        self.startUpdateSimInfoThread()
        
            
    def simulationThread(self):
        while self.network_dicts != [] and self.simulation_thread_run:
            
            # Get the current folder path and increment for future runs
            folder_name_i = self.folder_name + str(self.folder_num)
            folder_path = os.path.join(self.data_path, folder_name_i, "agent1")
            self.folder_num += 1
            
            self.run_directory = os.path.join(self.data_path, folder_name_i)
            
            # Update the path to the folder path
            self.sim_params["data_path"] = folder_path

            # Get the first network_dic
            network_dict = self.network_dicts.pop(0)
            self.current_network_dict = network_dict
            
            self.work_requested.emit()
            
            # Save sim_params and network_params in openfield folder
            saveFile(self.sim_params, "parameter_sets/current_parameter/sim_params.json")
            saveFile(network_dict, "parameter_sets/current_parameter/network_params_spikingnet.json")
            
            # Start the simulation
            self.simulation_process = subprocess.Popen(
                f"./run_simulation.sh {self.startSeed} {self.endSeed} --skip-trial-gen",
                shell=True,
                text=True,
                preexec_fn=os.setsid
            )
            self.startListeningToSimulation()
            
            # Wait with further execution until the process is finished
            while True:
                status = self.simulation_process.poll()
                if status is not None:
                    break
        
        self.simulation_thread_run = False
    
    
    def startSimulationThread(self):
        self.simulation_thread_run = True
        self.simulation_thread = threading.Thread(target=self.simulationThread)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
    
        
    def setParameterMenuBar(self):
        # Create File menu
        file_menu = self.menuBar().addMenu('&File')
        parameter_selection_menu = self.menuBar().addMenu('&Parameter selection')
        animation_menu = self.menuBar().addMenu("&Animation")

        self.addAction(file_menu, "&Quit", "Ctrl+Q", "Exit application", self.close)
        self.addAction(file_menu, "&Start simulation", "Ctrl+shift+Enter", "Start simulation", self.startSection.start_button.click)
        self.addAction(parameter_selection_menu, "&Switch to next parameter section", "CTRL+shift+right", "Next parameter selection", self.ParameterSection.nextTab)
        self.addAction(parameter_selection_menu, "&Switch to previous parameter section", "CTRL+shift+left", "Previous parameter selection", self.ParameterSection.prevTab)
        
        self.addAction(animation_menu, "&Settings", "Ctrl+A", "Open animation settings", self.show_new_window)
        
        self.menuBarMenus.append(file_menu)
        self.menuBarMenus.append(parameter_selection_menu)
        self.menuBarMenus.append(animation_menu)
    
    
    def show_new_window(self):
        self.w = GUISettingsPopUpWindow(self.gui_settings_json_path, self.original_gui_settings_json_path)
        self.w.show()
    
    def setAnimationMenuBar(self):
        file_menu = self.menuBar().addMenu('&File')
        
        self.addAction(file_menu, "&Quit", "Ctrl+Q", "Exit application", self.close)
        self.addAction(file_menu, "&Break simulation", "Ctrl+shift+b", "Break simulation", self.breakSection.button.click)
        
        self.menuBarMenus.append(file_menu)
    
    
    def removeMenuBar(self):
        while self.menuBarMenus:
            menu = self.menuBarMenus.pop()
            
            for action in menu.actions():
                menu.removeAction(action)
            
            self.menuBar().removeAction(menu.menuAction())
       
        
    def addAction(self, menu, name, shortcut, statusTip, trigger):
        action = QAction(name, self)
        action.setShortcut(shortcut)
        action.setStatusTip(statusTip)
        action.triggered.connect(trigger)

        menu.addAction(action)
    
    
    def listenSimulation(self):
        self.listening = True
        
        while self.listening:
            if self.simulation_process.poll() is None:
                time.sleep(0.5)
            else:
                self.breakSection.nameButton('Return')
                self.breakSection.simulation_finished()
                self.stop_update_sim_info_thread()
                self.trajectory_animation.stop_listening()
                self.stop_all_fr_animations()
                break
    
    
    def startListeningToSimulation(self):
        # Start listening to port for data
        self.listening_thread = threading.Thread(target=self.listenSimulation)
        self.listening_thread.daemon = True
        self.listening_thread.start()
    
    
    def stopListening(self):
        self.listening = False
        self.listening_thread.join()

    
    def startUpdateSimInfoThread(self):
        self.update_sim_information = True
        self.update_sim_thread = threading.Thread(target=self.update_sim_information_thread_callback)
        self.update_sim_thread.daemon = True
        self.update_sim_thread.start()
        
    
    def stop_update_sim_info_thread(self):
        self.update_sim_information = False
        self.update_sim_thread.join()
    
    
    def update_sim_information_thread_callback(self):
        self.trial_num = 0
        self.breakSection.setInfo(seed=self.startSeed)
        
        while self.update_sim_information:
            sim_time, trial_num = self.current_sim_time_getter()
            if trial_num != self.trial_num:
                self.breakSection.setInfo(seed=self.retreiveCurrentSeed(), trial=trial_num, data_dir=os.path.basename(self.run_directory), current_sim_time=sim_time)
            else:
                self.breakSection.setInfo(current_sim_time=sim_time)
            self.trial_num = trial_num
            time.sleep(self.update_sim_information_interval)
            
    
    def retreiveCurrentSeed(self):
        if not hasattr(self, "run_directory"):
            return 0
        
        directory_names = [x[0] for x in os.walk(self.run_directory)]
        seeds = []
        for dir_name in directory_names:
            # check if folder is of correct structure
            if re.search(self.run_directory + "/agent\d+$", dir_name) is not None:
                seed = re.search("\d+$", dir_name).group()
                seeds.append(int(seed))
        
        if len(seeds) == 0:
            return 0
        return max(seeds)        
    
    
    def generateDataFolderNum(self):
        directory_names = [x[0] for x in os.walk(self.data_path)]
        
        run_ids = []
        for dir_name in directory_names:
            # check if folder is of correct structure
            if re.search("../data/run_\d+$", dir_name) is not None:
                # extract run id
                run_id = re.search("\d+", dir_name).group()
                run_ids.append(int(run_id))

        if len(run_ids) > 0:
            return max(run_ids) + 1
        else:
            return 0
            
    
    def toggleAnimation(self):        
        if self.show_animation:
            # Deactivate animation
            self.show_animation = False
            self.stop_all_fr_animations()
            self.animationWindow.nameButton("Activate animation")
        else:
            # Activate animation
            self.show_animation = True
            self.start_fr_animations()
            self.animationWindow.nameButton("Deactivate animation")
            
        self.trajectory_animation.toggle_animation()
    
    
    def initialize_firing_rate_animations(self, trial_params):
        # initializes all firing rate animation objects and places them according to self.fire_rate_animations_arrangement
        
        # read settings file
        if os.path.exists(self.gui_settings_json_path):
            with open(self.gui_settings_json_path) as json_file:
                GUI_settings_json = json.load(json_file)
        elif os.path.exists(self.original_gui_settings_json_path):
            with open(self.original_gui_settings_json_path) as json_file:
                GUI_settings_json = json.load(json_file)
        
        self.fire_rate_animations_arrangement = GUI_settings_json["animations"]
        
        
        self.fr_animation_objects = np.zeros_like(self.fire_rate_animations_arrangement).tolist()
        self.active_fr_animations_mask = np.zeros_like(self.fire_rate_animations_arrangement).tolist()
        for i_column, column in enumerate(self.fire_rate_animations_arrangement):
            for i_row, neuron_type in enumerate(column):
                if not neuron_type in self.network_dicts[0].keys():
                    continue
                self.fr_animation_objects[i_column][i_row] = FiringRateAnimation(neuron_type, self.startSeed, self.endSeed, self.sim_params["simtime"], self.trajectory_animation.get_current_time, trial_params["trial_num"].to_numpy()[-1], GUI_settings_json["update_rate"], GUI_settings_json["sim_time_interval"], GUI_settings_json["bin_size"])
                self.animationWindow.addContentToAnimation(self.fr_animation_objects[i_column][i_row], i_column, i_row)
    
    
    def update_active_fr_animations(self, network_dict):
        # checks the network dict which neurons are present in the simulation and updates the mask accordingly
        for i_column, column in enumerate(self.fire_rate_animations_arrangement):
            for i_row, neuron_type in enumerate(column):
                if not neuron_type in network_dict.keys():
                    continue
                self.active_fr_animations_mask[i_column][i_row] = network_dict[neuron_type]["num_neurons"] > 0
    
    
    def stop_all_fr_animations(self, delete=False):
        # stops all animations. only set the delete argument, when returning to the main gui view. do not set it when pausing the animations!
        for i_column, column in enumerate(self.fr_animation_objects):
                for i_row, anim in enumerate(column):
                    if not isinstance(anim, FiringRateAnimation):
                        continue
                    
                    anim.stop()

        if delete:
            del self.fr_animation_objects
    
    
    def start_fr_animations(self, new_network_dict=None):
        # starts all activations that should be active according to the mask.
        current_seed = self.retreiveCurrentSeed()
        current_trial = self.current_sim_time_getter()[1]
        for i_column, column in enumerate(self.fr_animation_objects):
                for i_row, anim in enumerate(column):
                    if not isinstance(anim, FiringRateAnimation):
                        continue
                    if self.active_fr_animations_mask[i_column][i_row]:
                        if new_network_dict:
                            anim.set_network_dict(new_network_dict)
                            current_trial = 1
                            current_seed = self.startSeed
                        anim.start(data_folder_path=self.run_directory,seed=current_seed, trial=current_trial)
    
    
    

def saveFile(file, path):
    with open(path, "w") as f: 
        json.dump(file, f, indent=2)


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(stylesheet)
    app.setWindowIcon(QIcon('../gui/resources/icons/app_icon.png'))
    window = MainWindow()
    sys.exit(app.exec_())



if __name__ == '__main__':
    main()

    