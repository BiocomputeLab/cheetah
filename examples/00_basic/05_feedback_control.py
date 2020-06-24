
# Everything we need is in the cheetah package
import cheetah as ch

import os
# Turn off TensorFlow warnings (should be fixed in future)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

###############################################################################
# CREATE A CUSTOM CONTROL ALGORITHM

def analyse (t_step):
    print('Example analyse function run (does nothing), time_step:', t_step) 
    return None

def control_action (t_step, data):
    print('Example control_action function run (does nothing), time_step:', t_step) 
    return None

# https://www.winemantech.com/blog/leveraging-the-power-of-python-in-labview
# https://www.winemantech.com/products/testscript-python-labview-connector

control_alg = ch.ControlAlgorithm(analyse_fn=analyse,
	                              control_action_fn=control_action,
	                              sleep_time=1.0)

control_alg.start(max_t_step=10)

###############################################################################
