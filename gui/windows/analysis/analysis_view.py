from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QCheckBox, QLineEdit, QSpacerItem, QSizePolicy, QMessageBox
import os
import json



class CheckBoxWidget(QWidget):
    def __init__(self, name:str, state:bool, file:dict, keys:list):
        super(CheckBoxWidget, self).__init__()
        
        self.name = name
        self.state = state
        self.file = file
        self.keys = keys
        
        self.correct_type = True
        
        self.label = QLabel(name)
        self.checkbox = QCheckBox()
        self.checkbox.setChecked(state)
        self.checkbox.stateChanged.connect(self.update)
        
        self.layout = QHBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addStretch(1)  # Add stretchable space
        self.layout.addWidget(self.checkbox)
        self.setLayout(self.layout)
    
    
    def update(self):
        self.file[self.keys[0]][self.keys[1]] = self.checkbox.isChecked()
    
    
    def set_value(self, state):
        self.state = state
        self.checkbox.setChecked(self.state)


class TextBoxWidget(QWidget):
    def __init__(self, name:str, content, file:dict, keys:list, type_f:type = str):
        super(TextBoxWidget, self).__init__()
        
        self.name = name
        self.content = content
        self.file = file
        self.keys = keys
        self.type_f = type_f
        
        self.correct_type = True
        
        self.label = QLabel(name)
        self.line_edit = QLineEdit()
        self.line_edit.setText(str(content))
        self.line_edit.textChanged.connect(self.update)
        
        self.layout = QHBoxLayout()
        self.layout.addWidget(self.label)
        #self.layout.addStretch(1)
        self.layout.addWidget(self.line_edit)
        self.setLayout(self.layout)
    
    
    def update(self):
        if self.type_f is list:
            try:
                self.correct_type = True
                self.file[self.keys[0]][self.keys[1]] = json.loads(self.line_edit.text().replace("'", '"'))
            except:
                self.correct_type = False
        elif self.type_f is str:
            try:
                self.correct_type = True
                self.file[self.keys[0]][self.keys[1]] = str(self.line_edit.text())
            except:
                self.correct_type = False
        elif self.type_f is int:
            try:
                self.correct_type = True
                self.file[self.keys[0]][self.keys[1]] = int(self.line_edit.text())
            except:
                self.correct_type = False
        elif self.type_f is float:
            try:
                self.correct_type = True
                self.file[self.keys[0]][self.keys[1]] = float(self.line_edit.text())
            except:
                self.correct_type = False
        else:
            try:
                self.file[self.keys[0]][self.keys[1]] = int(self.line_edit.text())
            except:
                pass
            
            try:
                self.file[self.keys[0]][self.keys[1]] = float(self.line_edit.text())
            except:
                pass
            
            try:
                self.correct_type = True
                self.file[self.keys[0]][self.keys[1]] = str(self.line_edit.text())
            except:
                self.correct_type = False

    
    def set_value(self, text):
        self.content = text
        self.line_edit.setText(str(self.content))    


class CollapsibleWidget(QWidget):
    def __init__(self, title, content_layout):
        super(CollapsibleWidget, self).__init__()
        
        self.title = title
        self.checked = False
        
        self.content_area = QWidget()
        self.content_area.setVisible(self.checked)
        self.content_area.setLayout(content_layout)
        
        self.toggle_button = QPushButton("▶ " + self.title, objectName='accordionHeaderButton')
        self.toggle_button.clicked.connect(lambda: self.collapse() if self.checked else self.expand())

        layout = QVBoxLayout()
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.content_area)
        self.setLayout(layout)


    def collapse(self):
        self.checked = False
        self.toggle_button.setText("▶ " + self.title)
        self.content_area.setVisible(self.checked)
    
    
    def expand(self):
        self.checked = True
        self.toggle_button.setText("▼ " + self.title)
        self.content_area.setVisible(self.checked)


class AnalysisParameterWidget(QWidget):
    def __init__(self):
        super().__init__()
        
        # Set paths to the analysis configs
        self.path_analysis_file_orig = "parameter_sets/original_params/original_analysis_config.json"
        self.path_analysis_file = "parameter_sets/current_parameter/analysis_config.json"
        
        self.param_widgets = []
        self.collapsible_widgets = []
        
        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)
        
        self.setAnalysisFile()
        self.readFile()
        self.createParamsViewFromFile()
        self.createButtons()
    
    
    def createButtons(self):
        # Create a QHBoxLayout for the buttons
        self.button_layout = QHBoxLayout()

        # Add a spacer to push buttons to the right
        spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.button_layout.addItem(spacer)

        # Create buttons
        self.expand_button = QPushButton("Expand all")
        self.collapse_button = QPushButton("Collapse all")
        self.reset_button = QPushButton("Reset")
        
        self.expand_button.clicked.connect(self.expandAll)
        self.collapse_button.clicked.connect(self.collapseAll)
        self.reset_button.clicked.connect(self.resetAnalysisConfigs)

        # Add buttons to the button_layout
        self.button_layout.addWidget(self.expand_button)
        self.button_layout.addWidget(self.collapse_button)
        self.button_layout.addWidget(self.reset_button)
        
        self.layout.addLayout(self.button_layout)
    
    
    def expandAll(self):
        for widget in self.collapsible_widgets:
            widget.expand()
    
    
    def collapseAll(self):
        for widget in self.collapsible_widgets:
            widget.collapse()
        
    
    def setAnalysisFile(self):
        if os.path.exists(self.path_analysis_file):
            self.file_path = self.path_analysis_file
        else:
            self.file_path = self.path_analysis_file_orig


    def readFile(self):
        with open(self.file_path, 'r') as f:
            self.file = json.load(f)
    
    
    def checkFile(self):
        false_widgets = []
        
        for widget in self.param_widgets:
            if not widget.correct_type:
                false_widgets.append(widget.keys)
                
        if false_widgets:
            warning = "Following analysis configurations is/are wrong:\n"
            for widget in false_widgets:
                warning += widget[0] + "->" + widget[1] + "\n"
                
            QMessageBox.warning(self, "Warning", warning)
            return True
        
        return False
    
    
    def resetAnalysisConfigs(self):
        if os.path.exists(self.path_analysis_file):
            os.remove(self.path_analysis_file)
            self.setAnalysisFile()
            self.readFile()
        
        for widget in self.param_widgets:
            value = self.file
            for key in widget.keys:
                value = value[key]
            
            widget.set_value(value)
    
    
    def createParamsViewFromFile(self):
        for key, value in self.file.items():
            layout = QVBoxLayout()
            
            for param_name, param in value.items():
                if type(param) in [str, float, int, list, dict] or param is None:
                    widget = TextBoxWidget(param_name, param, self.file, [key, param_name], type(param))
                    self.param_widgets.append(widget)
                    layout.addWidget(widget)
                elif type(param) == bool:
                    widget = CheckBoxWidget(param_name, param, self.file, [key, param_name])
                    self.param_widgets.append(widget)
                    layout.addWidget(widget)
            
            widget = CollapsibleWidget(title=key, content_layout=layout)
            self.collapsible_widgets.append(widget)
            self.layout.addWidget(widget)
  