
# Everything we need is in the cheetah package
import cheetah as ch

import os
# Turn off TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

###############################################################################
# CREATION OF A SEGMENTER AND TRAINING THE MODEL

# Load data sets for training and validation
train_X = ch.load_images('./data/patches_train/images', normalization='max', 
                         file_type=('.png', '.tif'))
train_Y = ch.load_label_sets('./data/patches_train/labels', 3,
                             file_type=('.png', '.tif'))
val_X = ch.load_images('./data/patches_validate/images', normalization='max',
                       file_type=('.png', '.tif'))
val_Y = ch.load_label_sets('./data/patches_validate/labels', 3,
                           file_type=('.png', '.tif'))

# Create a blank segmenter
seg_train = ch.Segmenter(img_height=512, img_width=512, input_chn=1, n_classes=3)

# Train the segmenter using training data and validate using separate images
model_fit = seg_train.train(train_X, train_Y, val_X, val_Y,
                            weighted_loss=True, class_weights=None, 
                            batch_size=1, n_epochs=20)

# Plot averaged overall accuracy and loss of the trained segmentation model
ch.plot_acc_loss(model_fit.history['acc'], 
                 model_fit.history['val_acc'],
                 model_fit.history['loss'],
                 model_fit.history['val_loss'],
                 './output/test_segmentor')

# Save the segmentation model and weights to files
seg_train.save('./output/volvox_model.json', './output/volvox_model_weights.h5')

###############################################################################
