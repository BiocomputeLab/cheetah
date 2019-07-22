
# Everything we need is in the cheetah package
import cheetah as ch

import os
# Turn off TensorFlow warnings (should be fixed in future)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

###############################################################################
# CREATE A CUSTOM ANALYSIS ALGORITHM

def analyse (t_step):
    print('Example analyse function run (does nothing), time_step:', t_step) 
    return None

control_alg = ch.ControlAlgorithm(analyse_fn=analyse,
	                              sleep_time=1.0)

control_alg.start(max_t_step=10)

###############################################################################
