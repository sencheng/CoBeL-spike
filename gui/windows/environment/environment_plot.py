from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class PlotCanvas(FigureCanvas):
    def __init__(self, openfield_params_model, parent=None, width=5, height=5, dpi=100):
        
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.params_model = openfield_params_model

    def plot(self):
        # Clear the previous plot
        self.axes.clear()
        #print(ui_model)

        env_params = self.params_model.get_env_params()
        network_params = self.params_model.get_network_params()

        openfield_env_ymax = float(env_params['environment']['openfield']['ymax_position'])
        openfield_env_xmax = float(env_params['environment']['openfield']['xmax_position'])
        openfield_env_ymin = float(env_params['environment']['openfield']['ymin_position'])
        openfield_env_xmin = float(env_params['environment']['openfield']['xmin_position'])

        # set background color
        self.fig.patch.set_facecolor('#ececec')

        # Set plot boundaries
        self.axes.set_xlim(openfield_env_xmin, openfield_env_xmax)
        self.axes.set_ylim(openfield_env_ymin, openfield_env_ymax)

        # Enforce equal aspect ratio
        self.axes.set_aspect('equal', adjustable='box')

        self.axes.plot(0,0,'o', color = 'g', markersize=10, label='Starting Position')

        # Add labels and legend
        self.axes.set_xlabel("X")
        self.axes.set_ylabel("Y")
        self.axes.set_title("Environment")
        self.axes.legend()

        # Draw the canvas
        self.draw()
