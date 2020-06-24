
# Everything we need is in the cheetah package
import cheetah as ch

import os
# Turn off TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

###############################################################################
# GENERATE PATCHES FROM THE TRAINING DATA AND AUGMENT

input_images_path_train = './data/train/images'
input_labels_path_train = './data/train/labels'
output_images_path_train = './data/patches_train/images'
output_labels_path_train = './data/patches_train/labels'

input_images_path_validate = './data/validate/images'
input_labels_path_validate = './data/validate/labels'
output_images_path_validate = './data/patches_validate/images'
output_labels_path_validate = './data/patches_validate/labels'

augmentor = ch.DataAugmentor()

augmentor.generate_patches(input_images_path_train, input_labels_path_train,
                           output_images_path_train, output_labels_path_train, 
                           50, patch_shape=(256, 256),
                           prob_augment=0.2,
                           rotate_prob=0.2, hflip_prob=0.2, 
                           vflip_prob=0.2, scale_prob=0.2, scale_factor=1.2, 
                           shear_prob=0.2, shear_factor=0.3,
                           adjust_hist_prob=0.0, num_points=3)

augmentor.generate_patches(input_images_path_validate, input_labels_path_validate,
                           output_images_path_validate, output_labels_path_validate, 
                           20, patch_shape=(256, 256),
                           prob_augment=0.5,
                           rotate_prob=0.5, hflip_prob=0.5, 
                           vflip_prob=0.5, scale_prob=0.5, scale_factor=1.2, 
                           shear_prob=0.5, shear_factor=0.3,
                           adjust_hist_prob=0.0, num_points=3)

###############################################################################
