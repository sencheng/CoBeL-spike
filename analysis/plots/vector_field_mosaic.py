import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis import Weight as W



sim_path = "../data/test/agent25/sim_params.json"
fig_path = "../data/test/fig-25"
title = True
formats = ["pdf"]
cell_type = "place"
subplot_ids = [
    [1, 2, 3, 4],
    [11, 12, 13, 14]
]

shape = (5*len(subplot_ids[0]), 5*len(subplot_ids))
elements = np.unique(np.array(subplot_ids).flatten())
fig, axes = plt.subplot_mosaic(
    subplot_ids,
    empty_sentinel="BLANK",
    figsize=shape, 
    layout="tight"
) 

with open(sim_path, 'r') as fl:
        sim_config = json.load(fl)

w_obj = W(data_path=sim_config['data_path'], cell_type=cell_type, fig_path=fig_path, times=np.array([0.0]), quiet=True)
w_obj.read_files(cell_type=cell_type, quiet=True)
w_obj.read_sim_data(data_path=os.path.dirname(sim_path))
w_obj.set_cell_type(cell_type)


start_times = w_obj.tr_start_times['time'].tolist()
end_times = w_obj.tr_end_times['time'].tolist()
for i, (start, end) in enumerate(zip(start_times, end_times)):
    if i not in elements:
        continue
    
    temp_vec_field, stack_vec_field, n = w_obj.calc_vector_field_stack_at_time(start)
    
    axes[i].quiver(
        temp_vec_field[0, :],
        temp_vec_field[1, :],
        stack_vec_field[0, :] / n, 
        stack_vec_field[1, :] / n,
        scale_units='xy', 
        angles='xy', 
        scale=1,
        color = "b"
    )
    
    if title:
        w_obj.format_plot(axes[i], f"Trial: {i}")


for frm in formats:
    fig.savefig(f"../../data/vector_mosaic.{frm}")
