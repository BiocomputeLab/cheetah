
import numpy as np
import random
from os import listdir, makedirs
from os.path import isfile, isdir, join
from skimage import io
from skimage import transform as trf
import glob as gl

import warnings
warnings.filterwarnings("ignore")


class DataAugmentor():
    '''Class for data augmentation through patch selection and manipulation'''
    
    def generate_patches(self, input_images_path, input_labels_path,
                         output_images_path, output_labels_path, 
                         n_patches, patch_shape=(256, 256),
                         prob_augment=0.0,
                         rotate_prob=0.2, hflip_prob=0.2, 
                         vflip_prob=0.2, scale_prob=0.2, scale_factor=1.2, 
                         shear_prob=0.2, shear_factor=0.3):
        # Extract list of all filenames sorted
        image_files = sorted(gl.glob(input_images_path + '/*.*'))
        label_files = sorted(gl.glob(input_labels_path + '/*.*'))
        # Go through each image and create patches
        for f_idx in range(len(image_files)):
            cur_img = io.imread(image_files[f_idx])
            cur_lab = io.imread(label_files[f_idx])
            # Generate the location of the patches for each image
            row, col = random.randint(0, (
                                      cur_img.shape[0]-patch_shape[0])-1
                                      ), random.randint(0, (
                                      cur_img.shape[1]-patch_shape[1])-1)
            pair_seen = set((row, col))
            rand_pair = [[row, col]]
            for n in range(1, n_patches):
                row, col = random.randint(0, (
                                          cur_img.shape[0]-patch_shape[0])-1
                                          ), random.randint(0, (
                                          cur_img.shape[1]-patch_shape[1])-1)
                while (row, col) in pair_seen:
                    row, col = random.randint(0, (
                                              cur_img.shape[0]-patch_shape[0])-1
                                              ), random.randint(0, (
                                              cur_img.shape[1]-patch_shape[1])-1)
                pair_seen.add((row, col))
                rand_pair.append([row, col])
                
        # Write the patches to new files
        for n in range(0, len(rand_pair)):
            # Crop patch n
            img_patch_n = cur_img[rand_pair[n][0]:(rand_pair[n][0] + patch_shape[0]), 
                                  rand_pair[n][1]:(rand_pair[n][1] + patch_shape[1])
                                  ].astype(image.dtype)
            lab_patch_n = cur_img[rand_pair[n][0]:(rand_pair[n][0] + patch_shape[0]), 
                                  rand_pair[n][1]:(rand_pair[n][1] + patch_shape[1])
                                  ].astype(image.dtype)
            # Save patch n to file (assume 3 letter extension)
            img_patch_filename = output_images_path + '/' + image_files[f_idx][:-4] + '_patch_' + str(n) + image_files[f_idx][-4:]
            lab_patch_filename = output_labels_path + '/' + label_files[f_idx][:-4] + '_patch_' + str(n) + label_files[f_idx][-4:]
            io.imsave(fname=img_patch_filename, arr=img_patch_n)
            io.imsave(fname=lab_patch_filename, arr=lab_patch_n)

            # Check to see if to augment
            if random.uniform(0, 1) < prob_augment:
                # Rotation
                if random.uniform(0, 1) < rotate_prob:
                    self._rotate(img_patch_n, img_patch_filename)
                    self._rotate(lab_patch_filename, lab_patch_filename)
                # Horizontal flip
                if random.uniform(0, 1) < hflip_prob:
                    self._horizontal_flip(img_patch_n, img_patch_filename)
                    self._horizontal_flip(lab_patch_filename, lab_patch_filename)
                # Vertical flip
                if random.uniform(0, 1) < vflip_prob:
                    self._vertical_flip(img_patch_n, img_patch_filename)
                    self._vertical_flip(lab_patch_filename, lab_patch_filename)
                # Scaling
                if random.uniform(0, 1) < scale_prob:
                    self._scale(scale_factor, img_patch_n, img_patch_filename)
                    self._scale(scale_factor, lab_patch_filename, lab_patch_filename)
                # Shearing
                if random.uniform(0, 1) < scale_prob:
                    self._shear(shear_factor, img_patch_n, img_patch_filename)
                    self._shear(shear_factor, lab_patch_filename, lab_patch_filename)


    def _rotate(self, image, patch_filename):
        new_filename = patch_filename[:-4] + "_hfl" + patch_filename[-4:]
        rotate_tf = trf.SimilarityTransform(rotation=np.deg2rad(180))
                new_imag = trf.warp(image, inverse_map=rotate_tf, mode='reflect', 
                           preserve_range=True).astype(image.dtype)
        io.imsave(fname=new_filename, arr=new_imag)

    def _horizontal_flip(self, image, patch_filename):
        new_filename = patch_filename[:-4] + "_hfl" + patch_filename[-4:]
        new_imag = image[:, ::-1]
        io.imsave(fname=new_filename, arr=new_imag)

    def _vertical_flip(self, image, patch_filename):
        new_filename = patch_filename[:-4] + "_hfl" + patch_filename[-4:]
        new_imag = image[::-1, :]
        io.imsave(fname=new_filename, arr=new_imag)

    def _scale(self, scale_factor, image, patch_filename):
        if scale_factor is None or scale_factor < 1:
            scale_factor = 1.0
        new_filename = patch_filename[:-4] + "_hfl" + patch_filename[-4:]
        new_imag = trf.rescale(
                        image, scale_factor, mode='reflect', preserve_range=True)
        left = int((new_imag.shape[0]-image.shape[0])/2)
        right = left+image.shape[0]
        bottom = int((new_imag.shape[1]-image.shape[0])/2)
        top = bottom+image.shape[1]
        crop_imag = new_imag[bottom:top, left:right].astype(image.dtype)
        io.imsave(fname=new_filename, arr=new_imag)

    def _shear(self, shear_factor, image, patch_filename):
        if shear_factor is None or shear_factor < 0.1 or shear_factor > 0.5:
            shear_factor = 0.3
        new_filename = patch_filename[:-4] + "_hfl" + patch_filename[-4:]
        image = io.imread('{}/{}'.format(file[0], file[1]))
                affine_tf = trf.AffineTransform(shear=shear_factor)
                new_imag = trf.warp(image, inverse_map=affine_tf, mode='reflect', 
                           preserve_range=True).astype(image.dtype)
        io.imsave(fname=new_filename, arr=new_imag)
        