from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLineEdit, QScrollArea, QTabWidget, QSizePolicy, QMessageBox
from PyQt5.QtCore import Qt
     

        
class StartSection(QWidget):
    def __init__(self):
        super().__init__()
        
        self.layout = QVBoxLayout(self)
        
        self.layout.setAlignment(Qt.AlignBottom)
        
        self.start_button = QPushButton('Start')
        self.start_button.setFixedSize(150, 75)

        self.start_seed = QLineEdit()
        self.start_seed.setFixedWidth(self.start_button.width())
        self.start_seed.setPlaceholderText("Start seed")
        
        self.end_seed = QLineEdit()
        self.end_seed.setFixedWidth(self.start_button.width())
        self.end_seed.setPlaceholderText("End seed")
        
        self.layout.addWidget(self.start_seed)
        self.layout.addWidget(self.end_seed)
        self.layout.addWidget(self.start_button)
        
        self.setLayout(self.layout)
        
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
    
    
    def readStart(self):
        start = self.start_seed.text()
        return start

    
    def clearStart(self):
        self.start_seed.clear()
    
    
    def readEnd(self):
        end = self.end_seed.text()
        return end
    
    
    def clearEnd(self):
        self.end_seed.clear()
    
    
    def checkSeed(self, seed, warning):
        if not seed.isdigit():
            QMessageBox.warning(self, "Warning", warning)
            return False
        return True
    
    
    def checkSeedRatio(self, start, end, warning):
        if end - start < 0:
            QMessageBox.warning(self, "Warning", warning)
            return False
        return True
    

    def addEventToButton(self, event):
        self.start_button.clicked.connect(event)


class ParameterSection(QWidget):
    def __init__(self):
        super().__init__()
        
        self.layout = QVBoxLayout(self)
        
        self.tab_widget = QTabWidget()
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        self.layout.addWidget(self.tab_widget)
        
        self.simulation_parameter_area = QScrollArea()
        self.simulation_parameter_area.setWidgetResizable(True)
        self.tab_widget.addTab(self.simulation_parameter_area, "Simulation")
        
        self.environment_parameter_area = QScrollArea()
        self.environment_parameter_area.setWidgetResizable(True)
        self.tab_widget.addTab(self.environment_parameter_area, "Environment")
        
        self.network_parameter_area = QScrollArea()
        self.network_parameter_area.setWidgetResizable(True)
        self.tab_widget.addTab(self.network_parameter_area, "Network")
        
        self.analysis_parameter_area = QScrollArea()
        self.analysis_parameter_area.setWidgetResizable(True)
        self.tab_widget.addTab(self.analysis_parameter_area, "Analysis")
    
    
    def nextTab(self):
        currentIndex = self.tab_widget.currentIndex()
        newIndex = (currentIndex + 1) % self.tab_widget.count()
        self.tab_widget.setCurrentIndex(newIndex)
    
    
    def prevTab(self):
        currentIndex = self.tab_widget.currentIndex()
        newIndex = (currentIndex - 1) % self.tab_widget.count()
        self.tab_widget.setCurrentIndex(newIndex)
    
    def on_tab_changed(self, i):
        if i == 1:
            self.env_widget.update_plot()
        elif i == 2:
            self.content_widget.update_plot()
    
    
    def addContentToSimulation(self, widget:QWidget):
        self.simulation_parameter_area.setWidget(widget)
    
    
    def addContentToEnvironment(self, widget:QWidget):
        self.env_widget = widget
        self.environment_parameter_area.setWidget(widget)
    
    
    def addContentToNetwork(self, widget:QWidget):
        self.content_widget = widget
        self.network_parameter_area.setWidget(widget)
    
    
    def addcontentToAnalysis(self, widget:QWidget):
        self.analysis_parameter_area.setWidget(widget)
    