import sys
import json
import os
import numpy as np
import pandas as pd

# Updating the parameter file


trial = 7

with open('sim_params.json', 'r') as fl:
    net_dict = json.load(fl)

openfield = np.array(
    [[net_dict['enviourment']['openfield']['xmin_position'], net_dict['enviourment']['openfield']['ymin_position']],
     [net_dict['enviourment']['openfield']['xmax_position'], net_dict['enviourment']['openfield']['ymax_position']]])

x_start = net_dict['start']['x_position']
y_start = net_dict['start']['y_position']

init_list = []
x_array = np.linspace(openfield[0][0], openfield[1][0], num=6, endpoint=True)
y_array = np.linspace(openfield[0][1], openfield[1][1], num=6, endpoint=True)
reward = net_dict['goal']['hide_goal']
net_dict['goal']['hide_goal'] = True

for i in x_array:
    for j in y_array:
        dummy = [i, j]
        init_list.append(dummy)

net_dict['simtime'] = 1000.

data_path = net_dict['data_path']
filename = 'agents_location.dat'
file_path = os.path.join(data_path, filename)


def find_move_index(df):
    df_diff = df.diff(periods=1, axis=0)
    print(df)
    print(df_diff)
    for i in range(2, len(df)):
        if i < len(df) - 1:
            if df_diff.iloc[i]['x'] + df_diff.iloc[i]['y'] != 0:
                print(df.iloc[i - 1])
                return df.iloc[i - 1]['time'], df.iloc[i - 1]['x'], df.iloc[i - 1]['y'], df_diff.iloc[i]['x'], \
                       df_diff.iloc[i]['y']
        else:
            return df.iloc[i - 1]['time'], df.iloc[i - 1]['x'], df.iloc[i - 1]['y'], 0, 0


headers = ['time', 'x_position', 'y_position', 'x_step', 'y_step']


def save_log(data_path, filename, summary):
    file_path = os.path.join(data_path, filename)

    log_file = open(file_path, 'a')  # creates the text file
    log_file.write('{},{},{},{},{} \r\n'.format(summary[0], summary[1], summary[2], summary[3], summary[4]))
    log_file.close  # closes file


for i in range(len(init_list)):
    net_dict['start']['x_position'] = init_list[i][0]
    net_dict['start']['y_position'] = init_list[i][1]
    print(net_dict['start']['y_position'])

    with open('sim_params.json', 'w') as fl:
        json.dump(net_dict, fl)

    os.system('./run_sim_init_dir.sh ' + str(trial) + ' ' + str(trial))

    df = pd.read_csv(file_path, sep='\t')
    summary = find_move_index(df)
    save_log(data_path, 'initial_direction.dat', summary)
    print(summary)

net_dict['start']['x_position'] = x_start
net_dict['start']['y_position'] = y_start
net_dict['goal']['hide_goal'] = reward

# os.system('python plot_initial_direction.py')
