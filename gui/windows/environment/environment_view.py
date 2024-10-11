from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QPalette, QColor
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QSlider,
    QSpinBox,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QPushButton,
    QStyle,
    QScrollArea,
    QMessageBox
)

from windows.environment.environment_plot import PlotCanvas
import utils.CobelSpikeOriginalParams as OriginalParameters

class EnvironmentParameterWidget(QWidget):
    def __init__(self, openfield_params_model, sim_params_controller):
        super().__init__()

        self.openfield_params_model = openfield_params_model
        self.sim_params_controller = sim_params_controller
        self.init_ui()
    
    def init_ui(self):

        self.layout = QHBoxLayout(self)
        self.setLayout(self.layout)

        self.add_environment_parameters()
        self.add_visualization()

        # Set stretch factors to ensure equal space distribution
        self.layout.setStretch(0, 1)
        self.layout.setStretch(1, 1)

    def add_environment_parameters(self):
        
        env_parameter_layout = QVBoxLayout()
        reset_button = QPushButton("Reset Environment Parameters")
        reset_button.clicked.connect(lambda: self.reset_env_parameters())
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        #self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.update_environment_parameters()

        env_parameter_layout.addWidget(self.scroll_area)     
        env_parameter_layout.addWidget(reset_button, alignment=Qt.AlignRight)
        self.layout.addLayout(env_parameter_layout)

    def add_visualization(self):

        #print(self.params_model.get_env_params())
        self.plot_widget = PlotCanvas(openfield_params_model=self.openfield_params_model, parent=self, width=10,height=10,dpi=60)
        self.layout.addWidget(self.plot_widget)
        self.update_plot()

    def update_plot(self):
        self.plot_widget.plot()

    def update_environment_parameters(self):

        original_env_params = OriginalParameters.getAllEnvParameters()
        # self.ui_model = original_env_params
        
        self.environment_params_widget = QWidget()
        self.environment_params_layout = QVBoxLayout(self.environment_params_widget)
        self.environment_params_layout.setAlignment(Qt.AlignTop)

        self.create_full_layout_from_dict(original_env_params, self.environment_params_layout, "")
        self.scroll_area.setWidget(self.environment_params_widget)

    def reset_env_parameters(self):
        self.openfield_params_model.reset_env_params()
        self.update_environment_parameters()
        self.update_plot()

    def update_param_dict(self, changed_property, new_property_value, property_parent_dict, lineEdit):
        if property_parent_dict == ",environment,openfield": 
            self.sim_params_controller.update_env_params(changed_property, new_property_value)

        env_param_types = self.openfield_params_model.getAllEnvParameterTypes()
        prop = self.openfield_params_model.get_env_params()

        for d in property_parent_dict.split(","):
            if d == "":
                continue
            prop = prop[d]
            env_param_types = env_param_types[d]

        safe, value = self.check_type_safety(new_property_value, env_param_types[changed_property])
        if safe:
            if type(value) is bool and value == True: lineEdit.setText("True")
            if type(value) is bool and value == False: lineEdit.setText("False")
            prop[changed_property] = value
        else:
            lineEdit.setText(str(prop[changed_property]))
            QMessageBox.warning(self, "Warning", "Wrong Parameter Type!\nThe Value musst be of type " + str(env_param_types[changed_property]))

        self.update_plot()

    def check_type_safety(self, new_value, type):
        try:
            if type is bool:
                if new_value.lower() == "true":
                    new_value = True
                elif new_value.lower() == "false":
                    new_value = False
                else:
                    return False, ""
            else:
                new_value = type(new_value)

            return True, new_value
        except:
            return False, ""


    def create_full_layout_from_dict(self, dictionary : dict, parent_layout, parent_dict):

        for key in dictionary.keys():             

            if type(dictionary[key]) is dict:
                open = False
                if key == "All":
                    open = True
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
        

def create_environment_parameter_widget():
    """ Factory function to create and return a ConfigWidget instance. """
    return EnvironmentParameterWidget()
