from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, 
    QVBoxLayout, 
    QLineEdit, 
    QPushButton, 
    QGridLayout, 
    QLabel,
    QSpacerItem,
    QSizePolicy,
    QHBoxLayout,
    QScrollArea,
    QComboBox,
)

from windows.simulation.sim_plot import PlotCanvas

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from windows.simulation.sim_controller import SimulationController

class SimulationParameterWidget(QWidget):
    def __init__(self):
        super().__init__()

    def set_controller(self, controller):
        self.controller: SimulationController = controller
        self.init_ui()  

    def init_ui(self):
        # Basic layout
        self.layout = QHBoxLayout(self)
        self.setLayout(self.layout)

        # Add the parameter and visualization sections
        self.add_simulation_parameters_section()
        self.add_visualization_section()

        # Set stretch factors to ensure equal space distribution
        self.layout.setStretch(0, 1)
        self.layout.setStretch(1, 1)
    
    def add_visualization_section(self):
        # Initialize the plot widget once
        self.plot_widget = PlotCanvas(parent=self, width=10, height=10, dpi=60)
        self.layout.addWidget(self.plot_widget)

        # Initial plot update
        self.update_plot()

    def add_simulation_parameters_section(self):
        # Scroll area for parameters
        self.sim_params_layout = QVBoxLayout()

        # Create a new sim_params_widget
        self.trial_params_widget = None
        self.cumulated_trial_params_layout = None

        # Add the scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        # self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.sim_params_layout.addWidget(self.scroll_area)

        # Initial trial configuration 
        self.update_simulation_parameters()

        # Add a horizontal layout for the buttons
        buttons_layout = QHBoxLayout()

        # Add the add trial button
        add_button = QPushButton("+")
        add_button.clicked.connect(self.controller.on_add_trial_clicked)
        buttons_layout.addWidget(add_button, alignment=Qt.AlignLeft)

        # Add a spacer item to push the buttons to the sides
        spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        buttons_layout.addItem(spacer)

        # Add the reset button
        reset_button = QPushButton("Reset Simulation Parameters")
        reset_button.clicked.connect(lambda: self.controller.reset_sim_parameters())
        buttons_layout.addWidget(reset_button, alignment=Qt.AlignRight)

        # Add the buttons layout to the sim_params_layout
        self.sim_params_layout.addLayout(buttons_layout)

        # Add the sim_params_layout to the main layout
        self.layout.addLayout(self.sim_params_layout)

    def update_simulation_parameters(self):
        # Remove the old sim_params_widget and its layout if it exists
        if self.trial_params_widget:
            self.trial_params_widget.deleteLater()  # Ensure the old widget is marked for deletion

        # Create a new sim_params_widget
        self.trial_params_widget = QWidget()
        self.trial_params_layout = QVBoxLayout(self.trial_params_widget)
        self.trial_params_layout.setAlignment(Qt.AlignTop)

        # Fetch the ui_model from the controller
        ui_model = self.controller.get_ui_model() 

         # Add the cumulated trial parameters
        self.update_cumulated_trial_params(ui_model)

        # Populate the new sim_params_layout
        for index, trial_params in enumerate(ui_model["trial_params"]):
            self.trial_params_layout.addLayout(self.create_trial_config_layout(trial_params, index))

        # Set the new widget in the scroll area again
        self.scroll_area.setWidget(self.trial_params_widget)

    def update_cumulated_trial_params(self, ui_model):
        # Remove the old cumulated_trial_params_layout if it exists
        if self.cumulated_trial_params_layout:
            while self.cumulated_trial_params_layout.count():
                item = self.cumulated_trial_params_layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
            self.trial_params_layout.removeItem(self.cumulated_trial_params_layout)
            self.cumulated_trial_params_layout.deleteLater()

        self.cumulated_trial_params_layout = self.get_cumulated_trial_params(ui_model)
        self.cumulated_trial_params_layout.setAlignment(Qt.AlignTop)
        self.sim_params_layout.insertLayout(0, self.cumulated_trial_params_layout)

    def get_cumulated_trial_params(self, ui_model):
        max_trial_duration = sum(trial.get("Number of trials", 0) * trial.get("Max trial duration", 0) for trial in ui_model["trial_params"])
        number_of_trials = sum(trial.get("Number of trials", 0) for trial in ui_model["trial_params"])

        cumulated_layout = QVBoxLayout()
        cumulated_layout.setAlignment(Qt.AlignTop)

        # Create a container widget for the text
        container_widget = QWidget()
        container_layout = QHBoxLayout()
        container_layout.setAlignment(Qt.AlignCenter)

        text_widget = QLabel(
            f"Total Nr. of Trials: {int(number_of_trials)} | Total Trial Duration: {int(max_trial_duration)}", objectName='cumulatedTrialParams' )
        container_layout.addWidget(text_widget)

        container_widget.setLayout(container_layout)
        cumulated_layout.addWidget(container_widget)

        return cumulated_layout

    def update_plot(self):
        ui_model = self.controller.get_ui_model()
        if self.plot_widget:
            self.plot_widget.plot(ui_model)


    def create_trial_config_layout(self, trial_params, index):
        # create a vertical layout for the trial parameters
        trial_config_layout = QVBoxLayout()

        # add general trial parameters
        nr_and_time_grid = QGridLayout()
        nr_and_time_grid.setColumnStretch(0, 1)
        nr_and_time_grid.setContentsMargins(0,0,11,0)
        for i, key in enumerate(trial_params.keys()):
            if key == 'Start Position' or key == 'Goal Positions' or key == 'acc_closed':
                continue
            nr_and_time_grid.addWidget(QLabel(f"{key}: "), i, 0)
            line_edit = QLineEdit(str(trial_params[key]))
            
            # Use int for "Max trial duration" and "Number of trials", float for others
            if key in ["Max trial duration", "Number of trials"]:
                expected_type = int
            else:
                expected_type = float
            
            self.setup_line_edit_connection(line_edit, index, key, expected_type, self.controller.update_trial_param)
            nr_and_time_grid.addWidget(line_edit, i, 1, alignment=Qt.AlignTop)
        trial_config_layout.addLayout(nr_and_time_grid)


        # add start position accordion
        start_pos_accordion = self.get_start_pos(trial_params["Start Position"], index)
        trial_config_layout = self.add_layout_to_layout(start_pos_accordion, trial_config_layout)

        # add goal position accordion
        goal_pos_accordion = self.get_goal_pos(trial_params["Goal Positions"], trial_index=index)
        trial_config_layout = self.add_layout_to_layout(goal_pos_accordion, trial_config_layout)

        # add deletebutton if nr_of_widgets + 1 > 1
        add_delete_button = False
        if index + 1 > 1:
            add_delete_button = True

        trial_config_layout.setObjectName(f"trialConfigLayout_{str(index)}")
        acc_closed = trial_params.get('acc_closed', False)
        trial_config_accordion = self.create_accordion_layout(
            f"Trial Config {index + 1}", 
            trial_config_layout, 
            acc_closed, 
            add_delete_button,
            delete_button_name=f'deleteTrialButton_{index}'
            )
        return trial_config_accordion
    
    def get_start_pos(self, start_params, trial_index):
        start_pos_layout = QGridLayout()
        start_pos_layout.setColumnStretch(0, 1)
        start_pos_layout.setContentsMargins(0,0,0,0)
        acc_closed = start_params.get('acc_closed', False)
        for i, key in enumerate(start_params.keys()):
            if key == "acc_closed":
                continue
            start_pos_layout.addWidget(QLabel(f"{key}: "), i, 0)
            line_edit = QLineEdit(str(start_params[key]))
            self.setup_line_edit_connection(line_edit, trial_index, key, float, self.controller.update_start_position)
            start_pos_layout.addWidget(line_edit, i, 1, alignment=Qt.AlignTop)
        start_pos_accordion = self.create_accordion_layout(
            "Start Position",
            start_pos_layout,
            acc_closed=acc_closed
        )
        return start_pos_accordion


    def get_goal_pos(self, goal_position, trial_index):
        goal_pos_layout = QGridLayout()
        goal_pos_layout.setColumnStretch(0, 1)
        goal_pos_layout.setContentsMargins(0,0,0,0)
        acc_closed = goal_position.get('acc_closed', False)
        for j, key in enumerate(goal_position.keys()):
            if key == "acc_closed":
                continue
            if key == "Goal Shape":
                goal_pos_layout.addWidget(QLabel(f"{key}: "), j, 0)
                combo_box = QComboBox()
                combo_box.addItems(["round", "rect"])
                combo_box.setCurrentText(goal_position[key])
                combo_box.currentIndexChanged.connect(
                    lambda k=key, cb=combo_box: self.controller.update_goal_position(trial_index, "Goal Shape", cb.currentText())
                )
                goal_pos_layout.addWidget(combo_box, j, 1, alignment=Qt.AlignTop)
            else:
                goal_pos_layout.addWidget(QLabel(f"{key}: "), j, 0)
                line_edit = QLineEdit(str(goal_position[key]))
                self.setup_line_edit_connection(line_edit, trial_index, key, float, self.controller.update_goal_position)
                goal_pos_layout.addWidget(line_edit, j, 1, alignment=Qt.AlignTop)
        
        goal_pos_accordion = self.create_accordion_layout(
            "Goal Position",
            goal_pos_layout,
            acc_closed=acc_closed
        )
        return goal_pos_accordion

    
    def create_accordion_layout(self, acc_title, acc_body_layout, acc_closed, add_delete_button=False, delete_button_name=''):
        # Turn acc_body_layout into a widget
        acc_body_widget = QWidget()
        acc_body_widget.setLayout(acc_body_layout)
        acc_body_widget.setContentsMargins(50, 0, 0, 0)
        acc_body_widget.setHidden(acc_closed)

        # Open/close button
        if acc_closed:
            acc_button = QPushButton("▶ " + acc_title, objectName='accordionHeaderButton')
        else:
            acc_button = QPushButton("▼ " + acc_title, objectName='accordionHeaderButton')

        acc_button.clicked.connect(lambda: self.controller.on_accordion_clicked(acc_button))

        # Button layout: Horizontal layout to contain both the accordion and delete buttons
        button_layout = QHBoxLayout()
        button_layout.addWidget(acc_button)
        button_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        # Delete button
        if add_delete_button:
            delete_button = QPushButton("-")
            delete_button.clicked.connect(self.controller.on_delete_button_clicked)
            delete_button.setObjectName(delete_button_name)
            delete_button.setFixedWidth(40) 
            delete_button.setFixedHeight(40)
            button_layout.addWidget(delete_button)

        # Accordion layout: Vertical layout to contain the button layout and the accordion body widget
        accordion_layout = QVBoxLayout()
        accordion_layout.setAlignment(Qt.AlignLeft)
        accordion_layout.addLayout(button_layout)
        accordion_layout.addWidget(acc_body_widget)

        return accordion_layout
        
    def get_detele_trail_button_layout(self):
        # create button itself
        delete_button = QPushButton("-")
        delete_button.clicked.connect(self.on_delete_trial_clicked)
        delete_button.setObjectName('deleteTrialButton')

        # create layout for the button
        delete_trial_button_layout = QVBoxLayout()
        delete_trial_button_layout.setAlignment(Qt.AlignTop)
        delete_trial_button_layout.addWidget(delete_button)
        delete_trial_button_layout.setObjectName('deleteTrialButtonLayout')
        return delete_trial_button_layout
    
    def setup_line_edit_connection(self, line_edit, trial_index, param_name, expected_type, update_function):
        line_edit.editingFinished.connect(
            lambda le=line_edit, ti=trial_index, pn=param_name: 
            self.controller.update_value(ti, pn, le.text(), expected_type, update_function, le)
        )
    
    def add_layout_to_layout(self, layout_1, layout_2):
        """Converts layout_1 to a widget and adds it to layout_2."""
        layout_1_widget = QWidget()
        layout_1_widget.setLayout(layout_1)
        layout_2.addWidget(layout_1_widget, alignment=Qt.AlignTop)
        return layout_2