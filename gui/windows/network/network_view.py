from PyQt5.QtCore import Qt
import copy
from PyQt5.QtWidgets import (
    QLabel,
    QLineEdit,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QPushButton,
    QScrollArea,
    QMessageBox
)

from windows.network.network_plot import PlotCanvas

class NetworkParameterWidget(QWidget):
    def __init__(self, openfield_params_model):
        super().__init__()

        self.openfield_params_model = openfield_params_model
        self.init_ui()
    
    def init_ui(self):

        self.layout = QHBoxLayout(self)
        self.setLayout(self.layout)

        self.add_network_parameters()
        self.add_visualization()

        # Set stretch factors to ensure equal space distribution
        self.layout.setStretch(0, 1)
        self.layout.setStretch(1, 1)

        # network_tab_container = QHBoxLayout()
        # network_tab_container.addLayout(network_container)
        # network_tab_container.addWidget(img)


        # self.setLayout(network_tab_container)

    def add_network_parameters(self):
        
        network_parameter_layout = QVBoxLayout()
        reset_button = QPushButton("Reset Network Parameters")
        reset_button.clicked.connect(lambda: self.reset_network_parameters())
        self.multi_run_label = QLabel()
        self.multi_run_label.setWordWrap(True)
        self.multi_run_label.setText("Use \",\" or \":\" to trigger multiple runs with different parameters.\n \":\" can only be used with positive, natural numbers.")
        self.multi_run_label.setStyleSheet("background-color: lightgrey")
        self.multi_run_label.setMargin(10)


        setting_widget = QWidget()
        multi_run_layout = QHBoxLayout(setting_widget)
        multi_run_layout.addWidget(self.multi_run_label)


        #multi_run_layout.addWidget(reset_button, alignment=Qt.AlignRight | Qt.AlignBottom)

        multi_run_label_scroll_area = QScrollArea()
        multi_run_label_scroll_area.setWidgetResizable(True)
        multi_run_label_scroll_area.setMaximumHeight(200)
        multi_run_label_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        

        multi_run_label_scroll_area.setWidget(setting_widget)

        settings_layout = QHBoxLayout()
        settings_layout.addWidget(multi_run_label_scroll_area)
        settings_layout.addWidget(reset_button, alignment=Qt.AlignRight | Qt.AlignBottom)


        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        #self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.update_network_parameters()



        
        network_parameter_layout.addWidget(self.scroll_area) 
        network_parameter_layout.addLayout(settings_layout)   
        #network_parameter_layout.addWidget(multi_run_label_scroll_area) 
        #network_parameter_layout.addWidget(self.multi_run_label)
        #network_parameter_layout.addWidget(reset_button, alignment=Qt.AlignRight)
        self.layout.addLayout(network_parameter_layout)

    def add_visualization(self):

        self.plot_widget = PlotCanvas(openfield_params_model=self.openfield_params_model, parent=self, width=10,height=10,dpi=60)
        self.layout.addWidget(self.plot_widget)
        self.update_plot()

    def update_plot(self):
        self.plot_widget.plot()

    def update_network_parameters(self):

        original_network_params = self.openfield_params_model.get_network_params()

        self.network_params_widget = QWidget()
        self.network_params_layout = QVBoxLayout(self.network_params_widget)
        self.network_params_layout.setAlignment(Qt.AlignTop)

        self.place_num_neurons_le = None
        self.grid_num_neurons_le = None
        self.create_full_layout_from_dict(original_network_params, self.network_params_layout, "")
        self.scroll_area.setWidget(self.network_params_widget)

    def reset_network_parameters(self):
        self.openfield_params_model.reset_network_params()
        self.update_network_parameters()
        self.update_plot()
        self.multi_run_label.setText("Use \",\" or \":\" to trigger multiple runs with different parameters.\n \":\" can only be used with positive, natural numbers.")
        #self.multi_run_label.setVisible(False)

    def update_param_dict(self, changed_property, new_property_value, property_parent_dict, lineEdit):

        network_param_types = self.openfield_params_model.getAllNetworkParameterTypes()
        prop = self.openfield_params_model.get_network_params()

        for d in property_parent_dict.split(","):
            if d == "":
                continue
            prop = prop[d]
            network_param_types = network_param_types[d]
        
        if "," not in new_property_value and ":" not in new_property_value:
            safe, value = self.check_type_safety(new_property_value, network_param_types[changed_property])
            if safe:
                if type(value) is bool and value == True: lineEdit.setText("True")
                if type(value) is bool and value == False: lineEdit.setText("False")
                prop[changed_property] = value
                if (changed_property == "p_nrows" or changed_property == "p_ncols") and "place" in property_parent_dict: 
                    full_network_params = self.openfield_params_model.get_network_params()
                    full_network_params['place']['num_neurons'] = full_network_params['place']['cells_prop']['p_nrows'] * full_network_params['place']['cells_prop']['p_ncols']
                elif (changed_property == "g_nrows" or changed_property == "g_ncols") and "grid" in property_parent_dict: 
                    full_network_params = self.openfield_params_model.get_network_params()
                    full_network_params['grid']['num_neurons'] = full_network_params['grid']['cells_prop']['g_nrows'][0] * full_network_params['grid']['cells_prop']['g_ncols'][0]
            else:
                lineEdit.setText(str(prop[changed_property]))
                QMessageBox.warning(self, "Warning", "Wrong Parameter Type!\nThe Value musst be of type " + str(network_param_types[changed_property]))
        else:          
            prop[changed_property] = new_property_value


        if self.place_num_neurons_le != None:

            p_nrows = self.openfield_params_model.get_network_params()['place']['cells_prop']['p_nrows']
            if isinstance(p_nrows, str): 
                if ":" in p_nrows: p_nrows = p_nrows.split(":")[0]
                elif "," in p_nrows: p_nrows = p_nrows.split(",")[0]
            p_nrows = int(p_nrows)

            p_ncols = self.openfield_params_model.get_network_params()['place']['cells_prop']['p_ncols']
            if isinstance(p_ncols, str): 
                if ":" in p_ncols: p_ncols = p_ncols.split(":")[0]
                elif "," in p_ncols: p_ncols = p_ncols.split(",")[0]
            p_ncols = int(p_ncols)

            self.place_num_neurons_le.setText(str(p_nrows * p_ncols))

        if self.grid_num_neurons_le != None:

            g_nrows = self.openfield_params_model.get_network_params()['grid']['cells_prop']['g_nrows']
            if isinstance(g_nrows, str): 
                if ":" in g_nrows: g_nrows = g_nrows.replace("[","").replace("]","").split(":")[0]
                elif "," in g_nrows: g_nrows = g_nrows.replace("[","").replace("]","").split(",")[0]
            else: g_nrows = g_nrows[0]
            g_nrows = int(g_nrows)

            g_ncols = self.openfield_params_model.get_network_params()['grid']['cells_prop']['g_ncols']
            if isinstance(g_ncols, str): 
                if ":" in g_ncols: g_ncols = g_ncols.replace("[","").replace("]","").split(":")[0]
                elif "," in g_ncols: g_ncols = g_ncols.replace("[","").replace("]","").split(",")[0]
            else: g_ncols = g_ncols[0]
            g_ncols = int(g_ncols)

            self.grid_num_neurons_le.setText(str(g_nrows * g_ncols))

        self.update_plot()
        self.create_network_dicts()

    def check_type_safety(self, new_value, value_type):
        try:
            if value_type == bool:
                if new_value.lower() == "true":
                    new_value = True
                elif new_value.lower() == "false":
                    new_value = False
                else:
                    return False, ""
            else:
                if value_type != list:
                    new_value = value_type(new_value)
                else:
                    new_value = [int(new_value.replace("[","").replace("]",""))]

            return True, new_value
        except:
            return False, ""

    def create_full_layout_from_dict(self, dictionary : dict, parent_layout, parent_dict):

        for key in dictionary.keys():             

            if type(dictionary[key]) is dict:
                open = False
                #if key == "place":
                #    open = True
                acc_layout = self.create_accordion_widget(key, QWidget(), open)
                self.create_full_layout_from_dict(dictionary[key], acc_layout.acc_body, parent_dict + "," + key)

                if type(parent_layout) is QWidget:
                    if(parent_layout.layout() is not None):
                        current_layout = parent_layout.layout()
                        current_layout.addLayout(acc_layout)
                    else:
                        current_layout = acc_layout

                    parent_layout.setLayout(current_layout)
                else:
                    parent_layout.addLayout(acc_layout)
            else:
                param_layout = QVBoxLayout()

                param_grid = QGridLayout()
                param_grid.setColumnStretch(0,1)
                param_grid.setAlignment(Qt.AlignRight)
                param_grid.addWidget(QLabel(key + ":  "), 0,0)
                line_edit = QLineEdit(str(dictionary[key]))
                line_edit.editingFinished.connect(lambda le = line_edit, prop = key, dictionary = parent_dict: self.update_param_dict(prop, le.text(), dictionary, le))
                if key == "num_neurons" and parent_dict == ",place":
                    line_edit.setEnabled(False)
                    self.place_num_neurons_le = line_edit
                elif key == "num_neurons" and parent_dict == ",grid":
                    line_edit.setEnabled(False)
                    self.grid_num_neurons_le = line_edit

                param_grid.addWidget(line_edit, 0,1, alignment=Qt.AlignRight)
                param_layout.addLayout(param_grid)
                
                if type(parent_layout) is QWidget:

                    if(parent_layout.layout() is not None):
                        current_layout = parent_layout.layout()
                        current_layout.addLayout(param_layout)
                    else:
                        current_layout = param_layout

                    parent_layout.setLayout(current_layout)
                else:
                    param_grid.setContentsMargins(0,0,10,0)
                    parent_layout.addLayout(param_layout)

    def on_accordion_clicked(self,acc_body, acc_button):
        acc_body.setHidden(not acc_body.isHidden())
        button_text = acc_button.text()[2:]

        if acc_body.isHidden():
            acc_button.setText("▶ " + button_text)
        else:
            acc_button.setText("▼ " + button_text)

    #creates a button (accordion header) and links it with given QWidget (accordion body), that gets hidden/shows when accordion header is clicked
    def create_accordion_widget(self, acc_title, acc_body_widget, acc_open = False):
        vContainer = QVBoxLayout()
        vContainer.setAlignment(Qt.AlignLeft)

        vContainer.acc_body = acc_body_widget

        if acc_open:
            vContainer.acc_button = QPushButton("▼ " + acc_title, objectName='accordionHeaderButton')
        else:
            vContainer.acc_button = QPushButton("▶ " + acc_title, objectName='accordionHeaderButton')
        
        vContainer.acc_body.setContentsMargins(50,0,0,0)
        vContainer.acc_body.setHidden(acc_open)

        vContainer.acc_button.clicked.connect(lambda: self.on_accordion_clicked(vContainer.acc_body, vContainer.acc_button))
        self.on_accordion_clicked(vContainer.acc_body, vContainer.acc_button)
        vContainer.addWidget(vContainer.acc_button)
        vContainer.addWidget(vContainer.acc_body)

        return vContainer
    
    def create_network_dicts(self):
        multiple_entries = []
        longest_entry_length = 1
        current_network_params = self.openfield_params_model.get_network_params()
        network_params_list = []
        multi_run_text = "Multiple Runs selected:"
        
        for key, value, parent_path in self.recursive_items(current_network_params, ""):
            if "," in str(value) or ":" in str(value):
                multiple_entries.append((key, value, parent_path))
                if "," in str(value):
                    length = len(str(value).split(","))
                elif ":" in str(value) :
                    length = int(str(value).split(":")[1]) - int(str(value).split(":")[0]) + 1
                if length > longest_entry_length:
                   longest_entry_length = length

        if len(multiple_entries) == 0:
            #self.multi_run_label.setVisible(False)
            self.multi_run_label.setText("Use \",\" or \":\" to trigger multiple runs with different parameters.\n \":\" can only be used with positive, natural numbers.")
            self.openfield_params_model.set_network_dicts([current_network_params])
            return
        
        for i in range(0, longest_entry_length):
            
            split_network_dict = copy.deepcopy(current_network_params)
            multi_run_text += "\n\nRun " + str(i+1) + ":\n"
            
            for entry in multiple_entries:
                if "," in str(entry[1]):
                    key_entries = str(entry[1]).split(",")
                    key_entries = [str(s).strip() for s in key_entries]
                    if len(key_entries) >= i+1:
                        if not self.change_value(split_network_dict, entry[0], key_entries[i], entry[2]):
                            return
                        multi_run_text += str(entry[0]) + ": " + str(key_entries[i] + " ; ")
                    else:
                        if not self.change_value(split_network_dict, entry[0], key_entries[-1], entry[2]):
                            return
                        multi_run_text += str(entry[0]) + ": " + str(key_entries[-1] + " ; ")
                elif ":" in str(entry[1]):
                    key_entries = [*range(int(entry[1].split(":")[0]), int(entry[1].split(":")[1]) +1 )]
                    key_entries = [str(s).strip() for s in key_entries]
                    if len(key_entries) >= i+1:
                        if not self.change_value(split_network_dict, entry[0], key_entries[i], entry[2]):
                            return
                        multi_run_text += str(entry[0]) + ": " + str(key_entries[i]) + " ; "
                    else:
                        if not self.change_value(split_network_dict, entry[0], key_entries[-1], entry[2]):
                            return
                        multi_run_text += str(entry[0]) + ": " + str(key_entries[-1]) + " ; "

            network_params_list.append(split_network_dict)

        self.multi_run_label.setText(multi_run_text)
        self.multi_run_label.setVisible(True)

        self.openfield_params_model.set_network_dicts(network_params_list)


    def change_value(self, curr_dict, key, new_value, parent_path):
        
        network_param_types = self.openfield_params_model.getAllNetworkParameterTypes()
        grid_dict = curr_dict['grid']
        place_dict = curr_dict['place']

        for d in parent_path.split(";"):
            if d == "":
                continue
            curr_dict = curr_dict[d]
            network_param_types = network_param_types[d]

        safe, value = self.check_type_safety(new_value, network_param_types[key])

        # print(safe,value)
        if safe:
            curr_dict[key] = value
            if key == "p_ncols" or key == "p_nrows":
                place_dict['num_neurons'] = place_dict['cells_prop']['p_nrows'] * place_dict['cells_prop']['p_ncols']
            if key == "g_ncols" or key == "g_nrows":
                grid_dict['num_neurons'] = grid_dict['cells_prop']['g_nrows'][0] * grid_dict['cells_prop']['g_ncols'][0]
            return True
        else:
            QMessageBox.warning(self, "Warning", "At least one Parameter Type is not compatible!\nAll values musst be of type " + str(network_param_types[key]) + "\nPlease change \"" +new_value+"\" to be of that type.")
            return False



    def recursive_items(self, dictionary, parent_dict_path):
        for key, value in dictionary.items():
            if type(value) is dict:
                yield from self.recursive_items(value, parent_dict_path + ";" + key)
            else:
                yield (key, value, parent_dict_path)
        

def create_network_parameter_widget():
    """ Factory function to create and return a ConfigWidget instance. """
    return NetworkParameterWidget()
