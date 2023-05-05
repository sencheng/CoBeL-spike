#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 11:51:04 2019

@author: mohagmnr
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


class virtual_env():

    def __init__(self, net_obj):
        self.data_path = net_obj.data_path
        self.fig_path = net_obj.fig_path
        self.pop_dict = net_obj.pop_dict
        self.pops = net_obj.pop_dict.keys()
        self.bin_width = 50.0
        self.ms_scale = 1000.0

    def compute_action_value(self):
