import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

with open('sim_params.json', 'r') as fl:
    sim_dict = json.load(fl)

dir_ = sim_dict['data_path'].replace("data-tr", "fig-")
fig_path = dir_ + '/initial_directions.png'

openfield = np.array([[sim_dict['enviourment']['openfield']['xmin_position'],
                       sim_dict['enviourment']['openfield']['ymin_position']],
                      [sim_dict['enviourment']['openfield']['xmax_position'],
                       sim_dict['enviourment']['openfield']['ymax_position']]])

x = sim_dict['start']['x_position']
y = sim_dict['start']['y_position']

data_path = sim_dict['data_path']
filename = 'initial_direction.dat'
file_path = os.path.join(data_path, filename)

headers = ['time', 'x_position', 'y_position', 'x_step', 'y_step']
df = pd.read_csv(file_path, sep=',', names=headers)

fig, ax = plt.subplots()

x_pos = df['x_position']
y_pos = df['y_position']
x_direct = df['x_step']
y_direct = df['y_step']

ax.quiver(x_pos, y_pos, x_direct, y_direct)
fig.savefig(fig_path, format='png')
plt.show()
