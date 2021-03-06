
# Everything we need is in the cheetah package
import cheetah as ch

import os
# Turn off TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

###############################################################################
# GENERATE PATCHES FROM THE TRAINING DATA AND AUGMENT

input_images_path_train = './data/train/images_converted'
input_labels_path_train = './data/train/labels_converted'
output_images_path_train = './data/patches_train/images'
output_labels_path_train = './data/patches_train/labels'

input_images_path_validate = './data/validate/images_converted'
input_labels_path_validate = './data/validate/labels_converted'
output_images_path_validate = './data/patches_validate/images'
output_labels_path_validate = './data/patches_validate/labels'

augmentor = ch.DataAugmentor()

augmentor.generate_patches(input_images_path_train, input_labels_path_train,
                           output_images_path_train, output_labels_path_train, 
                           50, patch_shape=(512, 512),
                           prob_augment=0.2,
                           rotate_prob=0.2, hflip_prob=0.2, 
                           vflip_prob=0.2, scale_prob=0.2, scale_factor=1.5, 
                           shear_prob=0.2, shear_factor=0.3,
                           adjust_hist_prob=0.0, num_points=1)

augmentor.generate_patches(input_images_path_validate, input_labels_path_validate,
                           output_images_path_validate, output_labels_path_validate, 
                           25, patch_shape=(512, 512),
                           prob_augment=0.2,
                           rotate_prob=0.2, hflip_prob=0.2, 
                           vflip_prob=0.2, scale_prob=0.2, scale_factor=1.5, 
                           shear_prob=0.2, shear_factor=0.3,
                           adjust_hist_prob=0.0, num_points=1)

###############################################################################
