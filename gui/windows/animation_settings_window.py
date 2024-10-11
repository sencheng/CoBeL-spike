from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QGridLayout, QLineEdit, QSpacerItem, QMessageBox
from PyQt5.QtCore import Qt

import sys, os, json


class GUISettingsPopUpWindow(QWidget):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """
    def __init__(self, json_path, orig_json_path):
        super().__init__()
        main_layout = QVBoxLayout()
        self.json_path = json_path
        self.original_json_path = orig_json_path
        self.animation_arrangement_key = "animations"
        self.update_rate_key = "update_rate"
        self.sim_time_interval_key = "sim_time_interval"
        
        self.setWindowTitle("animation settings")
        
        title_label = QLabel("Set which neuron types you would like to be included at which position. Your input has to match the neuron type's key in the network dict. If you want to leave some animations empty, just input anything that is not a key", self)
        title_label.setWordWrap(True)
        main_layout.addWidget(title_label)
        
        
        grid_layout_positioning = QGridLayout()
        

        if os.path.exists(self.json_path):
            with open(self.json_path) as json_file:
                self.GUI_settings_json = json.load(json_file)
        elif os.path.exists(self.original_json_path):
            with open(self.original_json_path) as json_file:
                self.GUI_settings_json = json.load(json_file)
        
        self.lineEdits = []
        
        for row in range(2):
            for column in range(3):
                lineEdit = QLineEdit(self.GUI_settings_json[self.animation_arrangement_key][row][column], self)
                grid_layout_positioning.addWidget(lineEdit, row, column)
                self.lineEdits.append(lineEdit)
        self.lineEdits[0].setEnabled(False)
        self.lineEdits[0].setText("agent trajectory")
        
        main_layout.addLayout(grid_layout_positioning)
        
        main_layout.addSpacing(20)
        
        grid_layout_settings = QGridLayout()
        
        self.updateRateLabel = QLabel("firing rate plots update rate in ms:", self)
        self.updateRateLineEdit = QLineEdit(str(self.GUI_settings_json[self.update_rate_key]))
        grid_layout_settings.addWidget(self.updateRateLabel, 0, 0)        
        grid_layout_settings.addWidget(self.updateRateLineEdit, 0, 1) 
        
        self.xIntervalLabel = QLabel("firing rate plots x-axis length in ms:", self)
        self.xIntervalLineEdit = QLineEdit(str(self.GUI_settings_json[self.sim_time_interval_key]))
        grid_layout_settings.addWidget(self.xIntervalLabel, 1, 0)        
        grid_layout_settings.addWidget(self.xIntervalLineEdit, 1, 1) 
        
        main_layout.addLayout(grid_layout_settings)
        
        main_layout.addSpacing(30)
        
        grid_layout_buttons = QGridLayout()
        
        saveButton = QPushButton("save", self)
        saveButton.clicked.connect(self.save)
        grid_layout_buttons.addWidget(saveButton, 0, 1)
        
        resetButton = QPushButton("reset", self)
        resetButton.clicked.connect(self.reset)
        grid_layout_buttons.addWidget(resetButton, 0, 0)
        
        main_layout.addLayout(grid_layout_buttons)
        
        
        
                
        # Set the layout to the window
        self.setLayout(main_layout)

        # Set window size
        self.resize(400, 300)

    
    
    def fillLineEdits(self):
        self.updateRateLineEdit.setText(str(self.GUI_settings_json[self.update_rate_key]))
        self.xIntervalLineEdit.setText(str(self.GUI_settings_json[self.sim_time_interval_key]))
        
        for i in range(6):
            if i == 0:
                continue
            self.lineEdits[i].setText(self.GUI_settings_json[self.animation_arrangement_key][i // 3][i % 3])
    
    
    def save(self):
        # check type safety
        try:
            update_rate = int(self.updateRateLineEdit.text())
            x_interval = int(self.xIntervalLineEdit.text())
            self.GUI_settings_json[self.update_rate_key] = update_rate
            self.GUI_settings_json[self.sim_time_interval_key] = x_interval
        except ValueError:
            QMessageBox.warning(self, "Warning", "please enter an int")
            return
        
        n = 0
        for row in range(2):
            for column in range(3):
                self.GUI_settings_json[self.animation_arrangement_key][row][column] = self.lineEdits[n].text()
                n += 1
                
        with open(self.json_path, "w") as json_file:
            json.dump(self.GUI_settings_json, json_file, indent=2)
        
        self.close()
        
        
    def reset(self):
        if os.path.exists(self.original_json_path):
            with open(self.original_json_path) as json_file:
                self.GUI_settings_json = json.load(json_file)
        
        self.fillLineEdits()

