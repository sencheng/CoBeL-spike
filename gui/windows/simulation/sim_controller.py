from windows.simulation.sim_view import SimulationParameterWidget
import utils.CobelSpikeOriginalParams as OriginalParameters
import pandas as pd
import json
from PyQt5.QtWidgets import QMessageBox
import copy


class SimulationController:
    def __init__(self, view):
        self.view: SimulationParameterWidget = view
        self.original_sim_parameters = OriginalParameters.getAllSimParameters()
        self.configured_sim_parameters = self.original_sim_parameters
        self.original_env_parameters = OriginalParameters.getAllEnvParameters()
        self.original_trial_params = self.original_sim_parameters.get('trial_params', [])
        self.last_valid_values = {}
        self.is_updating = {} 
        self.reset_ui_model()
        self.view.set_controller(self)
        self.add_trial()

    def reset_ui_model(self):
        openfield_size = self.original_env_parameters.get('environment', {}).get('openfield', {})
        self.ui_model = {'trial_params': [],
                         'openfield_size': openfield_size}
        
    def reset_sim_parameters(self):
        OriginalParameters.deleteSimParameters()
        self.original_sim_parameters = OriginalParameters.getAllSimParameters()
        self.original_env_parameters = OriginalParameters.getAllEnvParameters()
        self.reset_ui_model()
        self.add_trial()
        
    def add_trial(self):
        if not self.ui_model['trial_params'] or len(self.ui_model['trial_params']) == 0:
            new_trial = self.create_default_trial_params() 
            self.ui_model['trial_params'].extend(new_trial)
        else:
            last_trial = self.ui_model['trial_params'][-1]
            new_trial = copy.deepcopy(last_trial)
            new_trial["acc_closed"] = False
            for trial in self.ui_model['trial_params']:
                trial["acc_closed"] = True
            self.ui_model['trial_params'].append(new_trial)


        self.view.update_simulation_parameters()
        self.view.update_plot()

    def create_default_trial_params(self):
        trial_params_list = []
        num_trials = len(self.original_sim_parameters['trial_params']['max_tr_dur'])

        for i in range(num_trials):
            trial_params = {
                "Max trial duration": int(self.original_sim_parameters['trial_params']['max_tr_dur'][i]),
                "Number of trials": int(self.original_sim_parameters['max_num_trs'] // num_trials),
                "Start Position": self.create_default_start_params(i),
                "Goal Positions": self.create_default_goal_params(i),
                "acc_closed": i != 0,  # Only the first trial is open by default
            }
            trial_params_list.append(trial_params)

        return trial_params_list
    
    def create_default_trial_params(self):
        trial_params_list = []
        num_configs = len(self.original_sim_parameters['trial_params']['max_tr_dur'])

        for i in range(num_configs):
            trial_params = {
                "Max trial duration": int(self.original_sim_parameters['trial_params']['max_tr_dur'][i]),
                "Number of trials": int(self.original_sim_parameters['max_num_trs'][i]),  # Now using the list
                "Start Position": self.create_default_start_params(i),
                "Goal Positions": self.create_default_goal_params(i),
                "acc_closed": i != 0,  # Only the first trial is open by default
            }
            trial_params_list.append(trial_params)

        return trial_params_list
    
    def create_default_start_params(self, index):
        return {
            "Start x": self.original_sim_parameters['trial_params']['start_x'][index],
            "Start y": self.original_sim_parameters['trial_params']['start_y'][index],
            "acc_closed": False,
        }

    def create_default_goal_params(self, index):
        return {
            "Goal Shape": self.original_sim_parameters['trial_params']['goal_shape'][index],
            "Goal Size 1": self.original_sim_parameters['trial_params']['goal_size1'][index],
            "Goal Size 2": self.original_sim_parameters['trial_params']['goal_size2'][index],
            "Goal x": self.original_sim_parameters['trial_params']['goal_x'][index],
            "Goal y": self.original_sim_parameters['trial_params']['goal_y'][index],
            "acc_closed": False,
        }
    
    def delete_trial(self, trial_id):
        # Logic to remove a trial from the model
        if trial_id < len(self.ui_model.get('trial_params', [])):
            del self.ui_model['trial_params'][trial_id]
        self.view.update_simulation_parameters() 
        self.view.update_plot() 

    def load_initial_data(self):
        self.view.update_simulation_parameters() 

    def get_ui_model(self):
        return self.ui_model
    
    def on_accordion_clicked(self, acc_button):
        button_text = acc_button.text()
        # if button_text contains "Trial Config" extract the trial index
        if "Trial Config" in button_text:
            button_index = int(button_text.split()[-1])
            trial_index = button_index - 1
            self.ui_model['trial_params'][trial_index]["acc_closed"] = not self.ui_model['trial_params'][trial_index]["acc_closed"]
            # if that trial is open, close all other trials
            if self.ui_model['trial_params'][trial_index]["acc_closed"] == False:
                for i, trial in enumerate(self.ui_model['trial_params']):
                    if i != trial_index:
                        trial["acc_closed"] = True
        elif "Goal Position" in button_text:
            # find the parent widget of the button
            button_widget = self.view.sender()
            button_parent_widget = button_widget.parentWidget()
            trail_config_widget = button_parent_widget.parentWidget()
            trail_config_layout = trail_config_widget.layout()
            name_of_trial_config_layout = trail_config_layout.objectName()
            trial_index = int(name_of_trial_config_layout.split('_')[-1])
            self.ui_model['trial_params'][trial_index]['Goal Positions']["acc_closed"] = not self.ui_model['trial_params'][trial_index]['Goal Positions']["acc_closed"]
        elif "Start Position" in button_text:
            button_widget = self.view.sender()
            button_parent_widget = button_widget.parentWidget()
            trail_config_widget = button_parent_widget.parentWidget()
            trail_config_layout = trail_config_widget.layout()
            name_of_trial_config_layout = trail_config_layout.objectName()
            trial_index = int(name_of_trial_config_layout.split('_')[-1])
            self.ui_model['trial_params'][trial_index]['Start Position']["acc_closed"] = not self.ui_model['trial_params'][trial_index]['Start Position']["acc_closed"]
        self.view.update_simulation_parameters()
        self.view.update_plot()

    def on_delete_button_clicked(self):
        button_widget = self.view.sender()
        name_of_the_button = button_widget.objectName()
        button_index = int(name_of_the_button.split('_')[-1])

        if 'deleteTrialButton' in name_of_the_button:
            self.delete_trial(button_index)

    def on_add_trial_clicked(self):
        self.add_trial()
    
    def check_type_safety(self, value, expected_type):
        try:
            if expected_type is bool:
                if value.lower() == "true":
                    return True, True
                elif value.lower() == "false":
                    return True, False
                else:
                    return False, None
            elif expected_type is int:
                if value.isdigit():
                    return True, int(value)
                else:
                    return False, None
            elif expected_type is float:
                converted = float(value)
                return True, converted
            else:
                return True, expected_type(value)
        except ValueError:
            return False, None

    def update_value(self, trial_index, param_name, value, expected_type, update_function, line_edit):
        key = (trial_index, param_name)
        
        # Check if we're already updating this field
        if self.is_updating.get(key, False):
            return
        
        self.is_updating[key] = True
        
        is_safe, converted_value = self.check_type_safety(value, expected_type)

        if is_safe:
            self.last_valid_values[key] = converted_value
            update_function(trial_index, param_name, converted_value)
            if isinstance(converted_value, bool):
                line_edit.setText(str(converted_value))
        else:
            if key in self.last_valid_values:
                line_edit.setText(str(self.last_valid_values[key]))
            else:
                line_edit.setText("")
            QMessageBox.warning(None, "Warning", f"Invalid input for {param_name}.  Expected type: {expected_type.__name__}")
        
        self.is_updating[key] = False

    def update_start_position(self, trial_index, param_name, value):
        if trial_index < len(self.ui_model['trial_params']):
            self.ui_model['trial_params'][trial_index]['Start Position'][param_name] = value
            self.view.update_plot()

    def update_goal_position(self, trial_index, param_name, value):
        if trial_index < len(self.ui_model['trial_params']):
            self.ui_model['trial_params'][trial_index]['Goal Positions'][param_name] = value
            self.view.update_plot()

    def update_trial_param(self, trial_index, param_name, value):
        if trial_index < len(self.ui_model['trial_params']):
            self.ui_model['trial_params'][trial_index][param_name] = value
            self.view.update_cumulated_trial_params(ui_model=self.ui_model)
    
    def update_env_params(self, changed_property, new_property_value):
        self.ui_model['openfield_size'][changed_property] = new_property_value
        self.view.update_plot()

    def generate_sim_param_files(self, save_files):

        # Initialize the new trial_params dictionary
        new_trial_params = {
            "goal_shape": [],
            "goal_size1": [],
            "goal_size2": [],
            "goal_x": [],
            "goal_y": [],
            "start_x": [],
            "start_y": [],
            "max_tr_dur": [],
        }

        # Loop through each trial in the ui_model and add its parameters to new_trial_params
        for trial in self.ui_model['trial_params']:
            new_trial_params["goal_shape"].append(trial["Goal Positions"]["Goal Shape"])
            new_trial_params["goal_size1"].append(trial["Goal Positions"]["Goal Size 1"])
            new_trial_params["goal_size2"].append(trial["Goal Positions"]["Goal Size 2"])
            new_trial_params["goal_x"].append(trial["Goal Positions"]["Goal x"])
            new_trial_params["goal_y"].append(trial["Goal Positions"]["Goal y"])
            new_trial_params["start_x"].append(trial["Start Position"]["Start x"])
            new_trial_params["start_y"].append(trial["Start Position"]["Start y"])
            new_trial_params["max_tr_dur"].append(trial["Max trial duration"])

        # Set the trial_params in the original_parameters to the new values
        self.configured_sim_parameters['trial_params'] = new_trial_params

        # Sum the number of trials across all trials and set it in the original_parameters
        total_number_of_trials = sum(trial["Number of trials"] for trial in self.ui_model['trial_params'])
        self.configured_sim_parameters['max_num_trs'] = int(total_number_of_trials)

        # extract the number of trials per config
        nr_of_trials = [trial["Number of trials"] for trial in self.ui_model['trial_params']]

        # Call function to generate .dat file
        dat_trials = self.generate_files(self.configured_sim_parameters, nr_of_trials, save_files)
        return dat_trials
    
    
    def generate_files(self, sim_parameters, nr_of_trials, save_files):
        total_trial_num = sim_parameters['max_num_trs']
        trial_param_dict = sim_parameters['trial_params']

        tmp = {'trial_num': range(1, total_trial_num + 1)}

        for param_key, param in trial_param_dict.items():
            if param_key not in tmp:
                tmp[param_key] = []
            for i in range(len(param)):
                tmp[param_key].extend([param[i]] * int(nr_of_trials[i]))
     
        total_simtime = sum(tmp['max_tr_dur'], 1000)
        sim_parameters['simtime'] = int(total_simtime)

        trials = pd.DataFrame(tmp)
        
        if save_files:
            # Write sim params to json file in openfield
            with open('./parameter_sets/current_parameter/sim_params.json', 'w+') as fl:
                json.dump(sim_parameters, fl, indent=2)

            trials.to_csv('parameter_sets/current_parameter/trials_params.dat', sep="\t", index=False)
            
        return trials, sim_parameters
    
