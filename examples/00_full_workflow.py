
# Everything we need is in the cheetah package
import cheetah as ch

import os
# Turn off TensorFlow warnings (should be fixed in future)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

###############################################################################
# CREATION OF A SEGMENTER AND TRAINING THE MODEL

# Load data sets for training and validation
train_X = ch.load_images('./data/train/images', normalization='max', 
	                     file_type=('.png', '.tif'))
train_Y = ch.load_label_sets('./data/train/labels', 2,
	                         file_type=('.png', '.tif'))
val_X = ch.load_images('./data/validate/images', normalization='max',
	                   file_type=('.png', '.tif'))
val_Y = ch.load_label_sets('./data/validate/labels', 2,
	                       file_type=('.png', '.tif'))

# Create a blank segmenter
seg_train = ch.Segmenter(img_height=256, img_width=256, input_chn=1, n_classes=2)

# Train the segmenter using training data and validate using separate images
model_fit = seg_train.train(train_X, train_Y, val_X, val_Y,
                            weighted_loss=True, class_weights=None, 
                            batch_size=8, n_epochs=50)

# Plot averaged overall accuracy and loss of the trained segmentation model
ch.plot_acc_loss(model_fit.history['acc'], 
                 model_fit.history['val_acc'],
                 model_fit.history['loss'],
                 model_fit.history['val_loss'],
                 'test_segmentor')

# Save the segmentation model and weights to files
seg_train.save('test_model.json', 'test_model_weights.h5')

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
