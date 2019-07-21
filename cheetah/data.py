
import os
import sys
import random
import numpy as np
from skimage import io


def load_image (filename, normalization='max', in_array=False):
    '''Read data files and return numpy array'''
    image = io.imread(filename)
    # Normalize image
    if normalization == 'max':
        image = image / image.max()
    elif normalization == 'mean':
        image = image / image.mean()
    # Make sure shape of data correct (width, height, channel)
    if len(image.shape) != 3:
        image = image[:, :, np.newaxis]
    if in_array:
        return image  
    else:
        image = image[np.newaxis, :, :, :]
        return image


def load_label_set (filename, n_classes, in_array=False):
    '''Read label files and perform one-hot encoding,
    single file which contains all classes''' 
    # 0 is background, 1...n are the label classes
    image = io.imread(filename)
    if len(image.shape) != 3:
        image = image[:, :, np.newaxis]
    if image.max() >= 255:   # if image is black / white (8, 16 or 24 bit)
        image = image / image.max()
    image = np.around(image)  # round float type data (data augmentation)
    one_hot_labels = []
    for g in range(0, n_classes):
        class_i = np.ones(image.shape, dtype='float32') * (image == g)
        one_hot_labels.append(class_i)
    # Move background class to last index
    one_hot_labels = one_hot_labels[1:] + [one_hot_labels[0]]
    one_hot_labels = np.concatenate(one_hot_labels, axis=2)
    if in_array:
        return one_hot_labels
    else:
        one_hot_labels = one_hot_labels[np.newaxis, :, :, :]
        return one_hot_labels
        

def load_images (path, normalization='max', file_type=('.tif', '.png')):
    images = []
    # Walk recursively through the given path looking for files
    abs_path = os.path.abspath(path)
    for root, subdirs, files in os.walk(abs_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            # Ignore files that are incorrect type or hidden
            if file_path.lower().endswith(file_type) and not filename.startswith('.'):
                images.append(load_image(file_path, 
                                         normalization=normalization,
                                         in_array=True))
    if len(images) == 0:
        print('No image files found.')
        return None
    else:
        image_array = np.zeros((len(images), *images[0].shape), dtype='float32')
        for i in range(image_array.shape[0]):
            image_array[i, :, :, :] = images[i]
        return image_array


def load_label_sets (path, n_classes, file_type=['.tif', '.png']):
    label_sets = []
    # Walk recursively through the given path looking for files
    abs_path = os.path.abspath(path)
    for root, subdirs, files in os.walk(abs_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            # Ignore files that are incorrect type or hidden
            if file_path.lower().endswith(file_type) and not filename.startswith('.'):
                label_sets.append(load_label_set(file_path, n_classes, in_array=True))
    if len(label_sets) == 0:
        print('No label files found.')
        return None
    else:
        label_set_array = np.zeros((len(label_sets), *label_sets[0].shape), dtype='float32')
        for i in range(label_set_array.shape[0]):
            label_set_array[i, :, :, :] = label_sets[i]
        return label_set_array


def predict_convert(main_path, folder_name, normalization='max',
                    filename='dataset'):
    '''
    Data coverter (to .npz file) for images to predict
    main_path: path to main folder
    folder_name: data folder with images (can contain subfolders)
    normalization: divides image by image max or mean
                    default = 'max'
    filename: filename of created .npz file
              default = 'dataset'
    '''
    # Initialze directory
    path = join(main_path, folder_name)
    subdir = [f for f in listdir(path) if isdir(join(path, f))]
    input_path = []
    if len(subdir) == 0:
        input_path.append([path])
    else:
        for i in range(0, len(subdir)):
            temp_path = join(path, subdir[i])
            input_path.append([temp_path])
    # Initialize folders
    input_data = []
    for i in range(0, len(input_path)):
        input_data.append([f for f in listdir(input_path[i][0]) if isfile(
                           join(input_path[i][0], f))])
    # Check if all folders contain the same number of files
    it = iter(input_data)
    len_entry = len(next(it))
    if not all(len(n) == len_entry for n in it):
        raise ValueError('Not all lists have same length!')
    # Load first data set (to determine input shape)
    single_input = get_data_multiple(0, input_path, input_data, normalization)
    data_X = np.zeros((len(input_data[0]), *single_input.shape), 
                      dtype='float32')
    # Fill input data array
    for i in range(data_X.shape[0]):
        data_X[i, :, :, :] = get_data_multiple(i, input_path, input_data, normalization)
    # Save dataset to file
    np.savez(join(path, filename + '.npz'), data_X=data_X)
    del data_X  # free up memory


def get_data_multiple(k, input_path, input_data, normalization):
    '''Read data files and return numpy array'''
    input_arrays = []
    for h in range(0, len(input_data)):
        image = io.imread('{}/{}'.format(
                input_path[h][0], input_data[h][k]))
        # Normalize image
        if normalization == 'max':
            image = image / image.max()
        elif normalization == 'mean':
            image = image / image.mean()
        if len(image.shape) != 3:
            image = image[:, :, np.newaxis]
        input_arrays.append(image)
    single_input = np.concatenate(input_arrays, axis=2)
    return single_input   
