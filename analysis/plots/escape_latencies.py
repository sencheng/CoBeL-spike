from misc import set_params_plots, get_formats, get_figure_size_multiplier

import matplotlib.pyplot as plt
import pandas as pd



def load_filtered_data(data_path, a_plus, a_minus, fr_noise, dat=False):
    if dat:
        df = pd.read_csv(data_path)
    else:
        df = pd.read_csv(data_path, sep="\t")

    df = df.loc[df["A_plus"]==a_plus]
    df = df.loc[df['A_minus']==a_minus]
    df = df.loc[df['fr_noise']==fr_noise]
    return df


def create_title():
    a_minus = df['A_minus'].unique()[0]
    omega_nc = df['fr_noise'].unique()[0]

    first_line = f"Symmetric STDP" if a_minus < 0 else f"$\\bf{{Asymmetric}}$" + " " + f"$\\bf{{STDP}}$"
    second_line = f"$\\mathbf{{\\Omega_{{NC}}={omega_nc}}}$" if omega_nc > 0 else f"$\Omega_{{NC}}=${omega_nc}"

    return first_line + "\n" + second_line


def add_legend(ax, colors=['red', 'green', 'blue'], markers = ["*","o","*"], labels=['Trial 10', 'Trial 20', 'Trial 30']):
    for color, marker, label in zip(colors, markers, labels):
        ax.plot([], [], marker=marker, color=color, label=label)
    ax.legend(loc='best', frameon=True)


def plot_sigma(ax, df, column, colors=['red', 'green', 'blue'], markers = ["*","o","*"], trials=[10, 20, 30], p_sigma_value=0.2, xticks=[0, 20, 40, 60, 80, 100], title_addition=False):
    data = df[df['p_sigma'] == p_sigma_value]

    for color, marker, trial in zip(colors, markers, trials):
        trial_data = data[data['trial'] == trial].reset_index()
        trial_data = trial_data.groupby(['p_ncols']).mean()
        trial_data = trial_data.reset_index()
        
        y = trial_data[column['data_column']].tolist()
        x = trial_data['p_ncols'].tolist()
        ax.plot(x, y, marker=marker, color=color)
    
    if title_addition:
        ax.set_title(title_addition + "\n\n" + f"$\sigma_{{PC}} = {p_sigma_value}~m$")
    else:
        ax.set_title(f"$\sigma_{{PC}} = {p_sigma_value}~m$")

    ax.set_ylabel(column['title'])
    ax.set_xlabel(f"$N_{{PC}}$")

    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{xtick:.0f}²' for xtick in xticks])
    ax.set_yticks(column['y_ticks'])

    
def plot_N(ax, df, column, colors=['red', 'green', 'blue'], markers = ["*","o","*"], trials=[10, 20, 30], p_ncols_value=21):
    data = df[df['p_ncols'] == p_ncols_value]
    
    for color, marker, trial in zip(colors, markers, trials):
        trial_data = data[data['trial'] == trial].reset_index()
        trial_data = trial_data.groupby(['p_sigma']).mean()
        trial_data = trial_data.reset_index()

        y = trial_data[column['data_column']].tolist()
        x = trial_data['p_sigma'].tolist()
        ax.plot(x, y, marker=marker, color=color)
    
    ax.set_title(f"$N_{{PC}} = {p_ncols_value}^2$")
    ax.set_ylabel(column['title'])
    ax.set_xlabel(f'$\sigma_{{PC}} [m]$')

    ax.set_yticks(column['y_ticks'])
    ax.invert_xaxis()


def plot_N_sigma(ax, df, column, colors=['red', 'green', 'blue'], markers = ["*","o","*"], trials=[10, 20, 30], xticks=[0, 20, 40, 60, 80, 100]):
    epsilon = 0.5  # Tolerance for comparing the product to 4.2
    data = df[(df['p_ncols'] * df['p_sigma']).abs().sub(4.2).abs() < epsilon]

    for color, marker, trial in zip(colors, markers, trials):
        trial_data = data[data['trial'] == trial].reset_index()
        trial_data = trial_data.groupby('p_ncols').mean().reset_index()
        
        y = trial_data[column['data_column']].tolist()
        x = trial_data['p_ncols'].tolist()
        ax.plot(x, y, marker=marker, color=color)

    ax.set_title(r"$N_{PC} \times \sigma_{PC}^2$"+r"$ \approx 17.64$")
    ax.set_ylabel(column['title'])
    ax.set_xlabel("$N_{PC}, \sigma_{PC}$")
    ax.set_yticks(column['y_ticks'])
    ax.set_xticks(xticks)
    labels_inf = [f'{xtick:.0f}²'+f"\n{' ' if xtick==0 else round(4.2/xtick,2)}" for xtick in xticks]
    labels = [s.replace("0²\ninf", "0") for s in labels_inf]
    ax.set_xticklabels(labels)


df_specs = [
    {'a_plus': 0.1, 'a_minus': -0.1, 'fr_noise': 0},
    {'a_plus': 0.1, 'a_minus': 0.1, 'fr_noise': 0},
    {'a_plus': 0.1, 'a_minus': -0.1, 'fr_noise': 2000}
]

rows = [
    {'title': r'$\tau$[s]', 'data_column': 'tr_duration', 'y_ticks': [0, 1, 2, 3, 4, 5]},
    {'title': 'DTW', 'data_column': 'DTW', 'y_ticks': [0, 2000, 4000, 6000]},
    {'title':'minimum proximity', 'data_column': 'proximity_min', 'y_ticks': [0, 0.1, 0.2, 0.3, 0.4, 0.5]}
]

data_path = "summary_beh_dataframe_final.csv"
data_path_old_data = "complete_beh_data_formatted.csv"
merging_columns = ['trial', 'p_ncols', 'p_sigma']

set_params_plots()
for row in rows:
    fig, axes = plt.subplots(3, len(df_specs), figsize=(len(df_specs)*get_figure_size_multiplier(), 3*get_figure_size_multiplier()))
    add_legend(axes[0, 0])

    for i, df_spec in enumerate(df_specs):
        df = load_filtered_data(data_path, df_spec['a_plus'], df_spec['a_minus'], df_spec['fr_noise'])

        if row['data_column'] == 'tr_duration':
            df_beh_data = load_filtered_data(data_path_old_data, df_spec['a_plus'], df_spec['a_minus'], df_spec['fr_noise'])
            df = pd.concat([df, df_beh_data], ignore_index=True)

        title = create_title()

        df = df.groupby(merging_columns+[row['data_column']]).mean()
        df = df.reset_index()

        plot_sigma(axes[0, i], df, row, title_addition=title)
        plot_N(axes[1, i], df, row)
        plot_N_sigma(axes[2, i], df, row)

    fig.tight_layout()
    for frm in get_formats():
        fig.savefig(f"../../data/escape_latencies_{row['data_column']}.{frm}")
