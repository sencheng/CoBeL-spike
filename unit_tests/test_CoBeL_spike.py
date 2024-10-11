#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 15:42:06 2021

@author: amit

This code performs unit testing. To test the implementation a smiple test is
performed where it checks the distribution of the duration in a single run from
the behavior_data,txt file and compare it with a reference duration from
behavior_data_GT.txt file. It reads the duration from both the files and distributes
them ramdonly into two sets of data (of same sizes as of the durations from the
two files). Then means are calculated from both the data sets. This is done
iteratively and then it is checked if those means are near the mean of the original
data in the initial distribution.

"""
import os
import unittest
import json
import numpy as np
import pandas as pd


class TestCoBeLSpike(unittest.TestCase):
    """This function includes all the tests for unit test"""

    def _compare_data(self, data_paths):
        """
        #Function: _compare_data(self, data_paths) compares the distributions of the
                  durations after a single run with the reference duration.
        Variable descriptions:
            data_paths  :  has the path to the computed and refernce behaviour data files
            num_iter    :  Number of iterations for calculation of means
        """
        num_iter = 10000
        ref_path = data_paths[0]
        new_data_path = data_paths[1]
        ref_data = pd.read_csv(ref_path)
        new_data = pd.read_csv(new_data_path)
        ref_dur = ref_data.duration.to_numpy()
        new_dur = new_data.duration.to_numpy()
        diff = ref_dur.mean() - new_dur.mean()
        pooled_dur = np.concatenate((ref_dur, new_dur))
        diff_shuffle = np.zeros(num_iter)
        for i in range(num_iter):
            np.random.shuffle(pooled_dur)
            diff_shuffle[i] = \
                pooled_dur[:ref_dur.size].mean() - \
                pooled_dur[ref_dur.size:].mean()
        p_val = np.min((np.sum(diff_shuffle > diff) / num_iter, np.sum(diff_shuffle < diff) / num_iter))
        if p_val < 0.05:
            print("fail")
            return False
        print("success")
        return True

    def test_duration(self):
        """
        #Function: test_duration(self) runs CoBeL-spike via the shell file in openfield
                   to create the data file and checks the distribution of the duration
                   from the behaviour data with the reference.
        Variable descriptions:
            param1         :  first random number generators (rngs) as input arguments
                              to run the simulation (equal to 1 for this test)
            param2         :  last random number generators (rngs) as input arguments
                              to run the simulation (equal to 1 for this test)
            flag           :  Track the state of the test
                              (true = successful test, false = failed test)
            REF_DATA_PATH  :  path to reference behaviour data file
            file_json      :  name of the JSON file in openfiled folder where the path to the
                              computed behaviour data text file is saved.
        """
        param1 = 1
        param2 = 1
        flag = False
        os.chdir('openfield')
        call_result = os.system('./run_sim_openfield_recursive.sh {} {}'.format(param1, param2))
        if call_result == 0:
            # exceptions can be handled
            pass
        ref_data_path = "../unit_tests/reference_datafiles/behavior_data_GT.txt"
        # IT MUST READ FROM SIM_PARAMS.JSON FOR THE FOLLOWING DATA.
        file_json = 'parameter_sets/current_parameter/sim_params.json'
        file = open(file_json)
        data_json = json.load(file)
        file.close()
        # computed_file = "data/nestsim/20-12-14-W60-placetest-rep7/data-tr1/behvavior_data.txt"
        computed_file = data_json['data_path'] + '/behvavior_data.txt'
        flag = self._compare_data([ref_data_path, computed_file])
        self.assertEqual(flag, True)


if __name__ == '__main__':
    unittest.main()
