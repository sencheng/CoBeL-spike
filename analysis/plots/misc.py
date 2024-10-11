import matplotlib.pyplot as plt



def set_params_plots():
    plt.rcParams.update({
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'axes.linewidth': 2,
        'lines.linewidth': 5,
        'lines.markersize': 12.5,
        'legend.fontsize': 18,
        'figure.subplot.wspace': 0.3
    })
    

def return_trajectory_plot_params():
    linewidth = 4 
    markersize = 8
    trial_titlesize = 18
    titlesize = 22
    padding_title = 35
    legendsize = 10
    label_fontsize = 40
    return linewidth, markersize, trial_titlesize, titlesize, padding_title, legendsize, label_fontsize


def get_padding_title():
    padding_title = 30
    return padding_title


def get_formats():
    formats = ["pdf", "jpg", "png", "svg"]
    return formats


def get_figure_size_multiplier():
    multiplier = 5
    return multiplier
