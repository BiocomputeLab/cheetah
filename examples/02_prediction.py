
# Everything we need is in the cheetah package
import cheetah as ch

import os
# Turn off TensorFlow warnings (should be fixed in future)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

###############################################################################
# LOADING A TRAINED MODEL, TESTING AND PREDICTING USING NEW DATA

# Create a new segmenter from the previously trained model
seg_predict = ch.load_segmenter('test_model.json', 'test_model_weights.h5')

# Load a single image and corrisponding label set (ground truth)
single_image = ch.load_image('./data/test_image.png', normalization='max')
single_label_set = ch.load_label_set('./data/test_image_labels.png', 2)

# Test quality of the trained segmenter on new image with known ground truth
prediction_metric = seg_predict.test(single_image, single_label_set)
print('Prediction metric:', prediction_metric)

# Predict the labels of new image with no ground truth
predicted_labels = seg_predict.predict(single_image)

###############################################################################
