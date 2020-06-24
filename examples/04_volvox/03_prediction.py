
# Everything we need is in the cheetah package
import cheetah as ch

import os
# Turn off TensorFlow warnings (should be fixed in future)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

###############################################################################
# LOADING A TRAINED MODEL, TESTING AND PREDICTING USING NEW DATA

# Create a new segmenter from the previously trained model
seg_predict = ch.Segmenter(model_filename='./output/volvox_model.json',
	                       weights_filename='./output/volvox_model_weights.h5')

# Load a single image and corrisponding label set (ground truth)
single_image = ch.load_image('./data/test_image.png', normalization='max')
single_label_set = ch.load_label_set('./data/test_image_labels.png', 4)

# Test quality of the trained segmenter on new image with known ground truth
prediction_metric = seg_predict.test(single_image, single_label_set)
print('Prediction metrics:', prediction_metric)

# Predict the labels of new image with no ground truth
predicted_labels = seg_predict.predict(single_image)

# Plot the predicted segmentation mask against graound truth
ch.plot_segmask(predicted_labels[0], y_true=single_label_set[0], class_to_plot=4, xtick_int=100,
                 ytick_int=100, show_plt=False, save_imag=True,
                 imag_name='./output/test_image_predicted_mask', save_as='pdf')

# Plot the pixel probability scores for each class (cell and background)
ch.plot_predprob(predicted_labels[0], class_names=['cell alive', 'cell dead', 'cell border', 'background'], n_classes=4, xtick_int=100,
                  ytick_int=100, show_plt=False, save_imag=True,
                  imag_name='./output/test_image_predicted_prob', save_as='pdf')

###############################################################################
