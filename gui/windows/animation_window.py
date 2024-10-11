from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QHBoxLayout, QPushButton, QLabel, QSizePolicy
from PyQt5.QtCore import Qt

class AnimationWindow(QWidget):
    def __init__(self):
        super().__init__()
        
        self.main_layout = QVBoxLayout(self)

        self.layout = QGridLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.layout_widget = QWidget()
        self.layout_widget.setLayout(self.layout)
        self.main_layout.addWidget(self.layout_widget)
        
        self.button = QPushButton('Deactivate animation')
        self.button.setFixedSize(225, 75)
        
        self.button_layout = QHBoxLayout()
        self.button_layout.setContentsMargins(0, 0, 0, 0)
        self.button_layout.addStretch()
        self.button_layout.addWidget(self.button)
        
        self.button_widget = QWidget()
        self.button_widget.setLayout(self.button_layout)
        
        self.main_layout.addStretch()
        self.main_layout.addWidget(self.button_widget)

        self.setLayout(self.main_layout)
        
        
    def addContentToAnimation(self, widget: QWidget, row: int, column: int, rowSpan: int = None, columnSpan: int = None):
        if rowSpan is None and columnSpan is None:
            self.layout.addWidget(widget, row, column)
        else:
            self.layout.addWidget(widget, row, column, rowSpan, columnSpan)

    
    def addEventToButton(self, event):
        self.button.clicked.connect(event)
    
        
    def nameButton(self, name):
        self.button.setText(name)



class BreakSection(QWidget):
    def __init__(self):
        super().__init__()
        
        self.layout = QVBoxLayout(self)
        
        # Add QLabel for text field
        self.info = QLabel("")
        self.info.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.info.setFixedSize(150, 250)
        self.info.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(self.info)
        
        self.button = QPushButton()
        self.button.setFixedSize(150, 75)
        self.layout.addWidget(self.button)
        
        self.layout.setAlignment(self.button, Qt.AlignBottom)
        self.layout.setAlignment(self.info, Qt.AlignCenter)
        
        self.setLayout(self.layout)
        
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        
        self.seed = ""
        self.trial = ""
        self.data_dir = ""
        self.sim_time = 0
        

    def addEventToButton(self, event):
        self.button.clicked.connect(event)
    
        
    def nameButton(self, name):
        self.button.setText(name)
    
    
    def simulation_finished(self):
        self.info.setText("The simulation is\n finished. Return\n to the main\n screen with the\n button below.")
        self.sim_time = 0
        self.trial = 0
        self.data_dir = ""
    
    
    def clearInfo(self):
        self.info.setText("")
    
        
    def setInfo(self, seed=None, trial=None, data_dir=None, current_sim_time=None):
        if seed:
            self.seed = seed
        if trial:
            self.trial=trial
        if data_dir:
            self.data_dir = data_dir
        if current_sim_time:
            self.sim_time = current_sim_time
        self.info.setText(f"sim time: {format(self.sim_time, '.2f')} s\n" + 
                          f"seed: {self.seed}\n" + 
                          f"trial: {self.trial}\n" +
                          f"data dir: {self.data_dir}\n")
