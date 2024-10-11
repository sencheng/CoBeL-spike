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

        p_nrows = network_params['place']['cells_prop']['p_nrows']
        if isinstance(p_nrows, str): 
            if ":" in p_nrows: p_nrows = p_nrows.split(":")[0]
            elif "," in p_nrows: p_nrows = p_nrows.split(",")[0]
        p_nrows = int(p_nrows)

        p_ncols = network_params['place']['cells_prop']['p_ncols']
        if isinstance(p_ncols, str): 
            if ":" in p_ncols: p_ncols = p_ncols.split(":")[0]
            elif "," in p_ncols: p_ncols = p_ncols.split(",")[0]
        p_ncols = int(p_ncols)

        for i in range(0,p_nrows):
            for j in range(0,p_ncols):
                self.axes.plot(openfield_env_xmin + (abs(openfield_env_xmin) + abs(openfield_env_xmax))/p_nrows*i + (abs(openfield_env_xmin) + abs(openfield_env_xmax))/p_nrows/2, openfield_env_ymin + (abs(openfield_env_ymin) + abs(openfield_env_ymax))/p_ncols*j + (abs(openfield_env_ymin) + abs(openfield_env_ymax))/p_ncols/2,'o', color = 'r', label=str(p_nrows*p_ncols) + " Place Cells" if i == 0 and j==0 else "")

        g_nrows = network_params['grid']['cells_prop']['g_nrows']
        if isinstance(g_nrows, str): 
            if ":" in g_nrows: g_nrows = g_nrows.replace("[","").replace("]","").split(":")[0]
            elif "," in g_nrows: g_nrows = g_nrows.replace("[","").replace("]","").split(",")[0]
        else: g_nrows = g_nrows[0]
        g_nrows = int(g_nrows)

        g_ncols = network_params['grid']['cells_prop']['g_ncols']
        if isinstance(g_ncols, str): 
            if ":" in g_ncols: g_ncols = g_ncols.replace("[","").replace("]","").split(":")[0]
            elif "," in g_ncols: g_ncols = g_ncols.replace("[","").replace("]","").split(",")[0]
        else: g_ncols = g_ncols[0]
        g_ncols = int(g_ncols)

        for i in range(0, g_nrows):
            for j in range(0, g_ncols):
                self.axes.plot(openfield_env_xmin/2 + i*0.05 + j*0.05, openfield_env_ymin/2 + j*0.05, '*', color = 'y', markersize = 15, label=str(g_nrows*g_ncols) + " Grid Cells" if i == 0 and j==0 else "")

        self.axes.plot(0,0,'o', color = 'g', markersize=10, label='Starting Position')

        # Add labels and legend
        self.axes.set_xlabel("X")
        self.axes.set_ylabel("Y")
        self.axes.set_title("Location of Cells")
        self.axes.legend()

        # Draw the canvas
        self.draw()
