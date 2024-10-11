from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

    def plot(self, ui_model):
        # Clear the previous plot
        self.axes.clear()

        # Set plot boundaries based on openfield_size 
        ymax_position = float(ui_model["openfield_size"]["ymax_position"])
        ymin_position = float(ui_model["openfield_size"]["ymin_position"])
        xmax_position = float(ui_model["openfield_size"]["xmax_position"])
        xmin_position = float(ui_model["openfield_size"]["xmin_position"])
        self.axes.set_xlim(xmin_position, xmax_position)
        self.axes.set_ylim(ymin_position, ymax_position)

        # set background color
        self.fig.patch.set_facecolor('#ececec')

        
        # Enforce equal aspect ratio
        self.axes.set_aspect('equal', adjustable='box')

        # Extract the specific trial configuration where acc_closed == False
        trial_config = next((trial for trial in ui_model["trial_params"] if not trial["acc_closed"]), None)
        
        if trial_config:
            # Plot start position
            start_pos = trial_config["Start Position"]
            all_closed = True
            if not start_pos["acc_closed"]:
                self.axes.plot(start_pos["Start x"], start_pos["Start y"], 'bo', label='Start')
                all_closed = False

            # Track if 'Goal' label has been added to avoid duplicates
            goal_label_added = False

            # Plot goal positions
            goal = trial_config["Goal Positions"]
            if not goal["acc_closed"]:
                if goal["Goal Shape"] == "round":
                    ellipse = patches.Ellipse((goal["Goal x"], goal["Goal y"]), 
                                        width=goal["Goal Size 1"], 
                                        height=goal["Goal Size 2"], 
                                        color='r', fill=False, label='Goal' if not goal_label_added else "")
                    self.axes.add_patch(ellipse)
                elif goal["Goal Shape"] == "rect":
                    rect = plt.Rectangle((goal["Goal x"] - goal["Goal Size 1"]/2, goal["Goal y"] - goal["Goal Size 2"]/2), 
                                        goal["Goal Size 1"], goal["Goal Size 2"], 
                                        color='r', fill=False, label='Goal' if not goal_label_added else "")
                    self.axes.add_patch(rect)
                goal_label_added = True
                all_closed = False

            # Add labels and legend
            self.axes.set_xlabel("X")
            self.axes.set_ylabel("Y")
            self.axes.set_title("Start and Goal Positions")
            if not all_closed:
                self.axes.legend()

            # Draw the canvas
            self.draw()
        else:
            self.axes.set_title("No trial configurations selected")
            self.draw()
