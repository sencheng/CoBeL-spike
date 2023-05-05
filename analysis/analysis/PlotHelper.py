#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 16:33:16 2023

Helper functions for making analysis plots.

@author: Ray Black
"""
import os
import json
import matplotlib.pyplot as plt


'''
Adds a visual representation of the reward zone to a given ax
c: designates the chosen color
ax: axis to add reward zone to
sim_dict: sim parameter dict including reward zone parameters
'''
def add_goal_zone(ax, sim_dict, c='r'): # TODO: rename to goal_zone
        
        goal_zone = plt.Circle((sim_dict['goal']['x_position'], sim_dict['goal']['y_position']),
                                 sim_dict['goal']['reward_recep_field'], color=c, alpha=0.1, fill=True,
                                 label='reward zone')
        ax.add_patch(goal_zone)
        
def add_obstacles(ax, sim_dict, c='k'):
    obs_dict = sim_dict["environment"]["obstacles"]
    if obs_dict["flag"]:
        for center, vert, horiz in zip(obs_dict["centers"], obs_dict["vert_lengths"], obs_dict["horiz_lengths"]):
            delta_y = vert / 2. # Get the length and width 
            delta_x = horiz / 2.  # as distances from the center point
            
            ll = (center[0] - delta_x, center[1] - delta_y) # lower left
            lr = (center[0] + delta_x, center[1] - delta_y) # lower right
            ur = (center[0] + delta_x, center[1] + delta_y) # upper right
            ul = (center[0] - delta_x, center[1] + delta_y) # upper left
            
            obs_actual = plt.Polygon([ll, lr, ur, ul], closed=True, color='k')
            ax.add_patch(obs_actual)