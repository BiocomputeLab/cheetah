
# Everything we need is in the cheetah package
import cheetah as ch

import os
# Turn off TensorFlow warnings (should be fixed in future)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

###############################################################################
# CREATE A CUSTOM CONTROL ALGORITHM

# Create a new segmenter from the previously trained model
seg_predict = ch.Segmenter(model_filename='./output/test_model.json',
	                       weights_filename='./output/test_model_weights.h5')

# EXAMPLE NOT COMPLETE - PROCESS A LIST OF IMAGES, ANALYSE AND OUTPUT RESULTS

###############################################################################
