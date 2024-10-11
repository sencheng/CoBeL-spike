from misc import set_params_plots, get_padding_title, get_formats, get_figure_size_multiplier

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import string



def load_filtered_data(data_path, df_spec):
    df = pd.read_csv(data_path, sep="\t")

    for key, value in df_spec.items():
        df = df.loc[df[key]==value]

    return df


def createTitle(df, base):
    a_minus = df['A_minus'].unique()[0]
    sigma_pc = df['p_sigma'].unique()[0]
    N_pc = df['p_ncols'].unique()[0]
    omega_ng = df['fr_noise'].unique()[0]

    first_line = f"Symmetric STDP" if a_minus < 0 else f"$\\bf{{Asymmetric}}$" + " " + f"$\\bf{{STDP}}$"
    second_line = f"$\\mathbf{{\sigma_{{PC}} = {sigma_pc} m, N_{{PC}} = {N_pc}^2}}$" if sigma_pc != base['p_sigma'] else f"$\sigma_{{PC}} = {sigma_pc} m, N_{{PC}} = {N_pc}^2$"
    third_line = f"$\\mathbf{{\\Omega_{{NG}}={omega_ng}}}$" if omega_ng != base['fr_noise'] else f"$\Omega_{{NG}}=${omega_ng}"
    
    return first_line + '\n' + second_line + '\n' + third_line


def split_df(df):
    df_no_switch = df[df['with_trial_switch'] == 3].reset_index()
    df_no_switch = df_no_switch.drop(columns=['goal_size1', 'goal_x', 'goal_y', 'trial_switch', 'with_trial_switch', 'index'])

    df = df[df['with_trial_switch'] == 11]
    df_goalA = df[df['goal_x'] > 0].reset_index()
    df_goalA = df_goalA.drop(columns=['goal_size1', 'goal_x', 'goal_y', 'trial_switch', 'with_trial_switch', 'index'])
    df_goalB = df[df['goal_x'] < 0].reset_index()
    df_goalB = df_goalB.drop(columns=['goal_size1', 'goal_x', 'goal_y', 'trial_switch', 'with_trial_switch', 'index'])
    df_goalB = df_goalB.drop(list(range(0, 10)) + list(range(20, 30)),axis=0)

    return df_goalA, df_goalB, df_no_switch


def plot_without_switch(ax, df, column):
    trials = df['trial'].tolist()
    data = df[column].tolist()
    ax.plot(trials, data, dashes=[6, 2], color='black', linewidth=mpl.rcParams['lines.linewidth']*0.4)


def plot_B(ax, df, column):
    trials = df['trial'].tolist()
    data = df[column].tolist()
    ax.plot(trials, data, color='green')


def plot_A(ax, df, column):
    trials = df['trial'].tolist()
    data = df[column].tolist()

    ax.plot(trials[0:10], data[0:10], color='red')
    ax.plot(trials[10:20], data[10:20], color='green', linestyle=':')
    ax.plot(trials[20:30], data[20:30], color='blue')


def add_legend(ax):
    ax.plot([], [], color='red', label="A")
    ax.plot([], [], color='green', label="B")
    ax.plot([], [], color='blue', label="A")
    ax.plot([], [], color='green', linestyle=':', label="A")
    ax.plot([], [], dashes=[6, 2], color='black', linewidth=mpl.rcParams['lines.linewidth']*0.4, label="A")
    ax.legend(loc='upper right', fancybox=True, shadow=True, ncol=1)

df_specs = [
    {'A_plus': 0.1, 'A_minus': -0.1, 'p_sigma': 0.2, 'p_ncols': 21, 'fr_noise': 0},
    {'A_plus': 0.1, 'A_minus': 0.1, 'p_sigma': 0.2, 'p_ncols': 21, 'fr_noise': 0},
    {'A_plus': 0.1, 'A_minus': -0.1, 'p_sigma': 0.04, 'p_ncols': 101, 'fr_noise': 0},
    {'A_plus': 0.1, 'A_minus': -0.1, 'p_sigma': 0.2, 'p_ncols': 21, 'fr_noise': 2000},
]

ylims = {
    'tr_duration': [0, 5.9],
    'DTW': [0, 7000],
    'proximity': [0.0, 0.5]
}

yticks = {
    'tr_duration': [0, 1, 2, 3, 4, 5],
    'DTW': [0, 2000, 4000, 6000],
    'proximity': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
}


data_path = "summary_beh_dataframe_final_0.3.csv"
merging_columns = ['trial', 'p_ncols', 'p_sigma', 'A_plus', 'p_max_fr', 'goal_size1', 'goal_x', 'goal_y', 'trial_switch', 'with_trial_switch']
rows = [r'$\tau$[s]', 'minimum distance [m]','DTW']
label_letters = ['A', 'B', 'C', 'D']

set_params_plots()
fig, axes = plt.subplots(3, len(df_specs), figsize=(len(df_specs)*get_figure_size_multiplier(), 3*get_figure_size_multiplier()))
add_legend(axes[-1, -1])


for i, (letter, _) in enumerate(zip(string.ascii_uppercase, df_specs)):
    axes[0, i].text(-0.07, 1.7, letter, horizontalalignment='left', verticalalignment='top', fontsize=30, transform=axes[0, i].transAxes, weight='bold')

for i, row in enumerate(rows):
    axes[i, 0].set_ylabel(row)

for i, df_spec in enumerate(df_specs):
    df = load_filtered_data(data_path, df_spec)

    if df.empty:
        print(f"There is no data in {data_path} for {df_spec}")
        continue
    
    title = createTitle(df, df_specs[0])

    df = df.groupby(merging_columns).mean()
    df = df.reset_index()
    df = df.drop(columns=['seed', 'p_sigma', 'p_ncols', 'p_ncols', 'fr_noise', 'p_max_fr', 'A_plus', 'A_minus', 'traj_length', 'fr_noise'])

    df_goalA, df_goalB, df_no_switch = split_df(df)

   
    axes[0, i].set_ylim(ylims['tr_duration'])
    axes[1, i].set_ylim(ylims['proximity'])
    axes[2, i].set_ylim(ylims['DTW'])

    axes[0, i].set_yticks(yticks['tr_duration'])
    axes[1, i].set_yticks(yticks['proximity'])
    axes[2, i].set_yticks(yticks['DTW'])

    axes[0, i].set_title(title, pad=get_padding_title())
    axes[-1, i].set_xlabel("Trials")

    plot_without_switch(axes[0, i], df_no_switch, 'tr_duration')
    plot_without_switch(axes[1, i], df_no_switch, 'proximity_min')
    plot_without_switch(axes[2, i], df_no_switch, 'DTW')

    plot_B(axes[0, i], df_goalB, 'tr_duration')
    plot_B(axes[1, i], df_goalB, 'proximity_min')
    plot_B(axes[2, i], df_goalB, 'DTW')

    plot_A(axes[0, i], df_goalA, 'tr_duration')
    plot_A(axes[1, i], df_goalA, 'proximity_min')
    plot_A(axes[2, i], df_goalA, 'DTW')

    for j, row in enumerate(rows):
        axes[j, i].set_xlim(1, 30)
        axes[j, i].set_xticks([10, 20, 30])


fig.tight_layout()
for frm in get_formats():
    fig.savefig(f"../../data/learning_curve.{frm}")

