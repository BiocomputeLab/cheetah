# Combine a number of separate label images generated from MATLAB into a single image.
# Author: Thomas E. Gorochowski <thomas.gorochowski@bristol.ac.uk>

import numpy as np
import skimage as sk
import glob as gl

import random
import colorsys

import scipy

import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

import skimage.measure
from skimage import io
from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate
from skimage.morphology import remove_small_objects

###################################################################################################

# This should be where the raw images are held
data_path = './_data'

# Where the combined label files should be saved
output_path = './_output'

images = []
files = sorted(gl.glob(data_path + '/*.png'))
for f in files:
    cur_image = io.imread(f)
    cur_mask = cur_image > 0
    images.append(cur_mask)
 
###################################################################################################

# Indexes in the region property array
ID = 0
CENTROID_X = 1
CENTROID_Y = 2
AREA = 3
BBOX_MIN_X = 4
BBOX_MIN_Y = 5
BBOX_MAX_X = 6
BBOX_MAX_Y = 7
ORIENTATION = 8
MAJOR_LENGTH = 9
MINOR_LENGTH = 10

def extract_mask_labels (img, background=0, connectivity=1, min_size=15):
    # Remove small regions (noise)
    if min_size is not None:
        remove_small_objects(img, min_size=min_size, connectivity=connectivity, in_place=True)
    # Extract connected components in mask and region properties
    label_img, num_of_labels = skimage.measure.label(img, background=background, connectivity=connectivity, return_num=True)
    props = regionprops_table(label_img, properties=('centroid',
                                                     'area',
                                                     'bbox',
                                                     'orientation',
                                                     'major_axis_length',
                                                     'minor_axis_length'))
    # Reformat the properties of each region into an array
    new_props = np.zeros((num_of_labels+1, 11))
    new_props[:, ID] = range(num_of_labels+1)
    new_props[1:, CENTROID_X] = props['centroid-1']
    new_props[1:, CENTROID_Y] = props['centroid-0']
    new_props[1:, AREA] = props['area']
    new_props[1:, BBOX_MIN_X] = props['bbox-0']
    new_props[1:, BBOX_MIN_Y] = props['bbox-1']
    new_props[1:, BBOX_MAX_X] = props['bbox-2']
    new_props[1:, BBOX_MAX_Y] = props['bbox-3']
    new_props[1:, ORIENTATION] = props['orientation']
    new_props[1:, MAJOR_LENGTH] = props['major_axis_length']
    new_props[1:, MINOR_LENGTH] = props['minor_axis_length']
    return label_img, new_props

def relabel_image (img, props, img_prev, props_prev, max_dist=10):
    # The new relabelled data
    props_new = np.copy(props)
    img_new = np.zeros_like(img)
    # Calculate pairwise distance between all points
    dist_mat = scipy.spatial.distance.cdist(props[1:, CENTROID_X:CENTROID_Y], props_prev[1:, CENTROID_X:CENTROID_Y], metric='euclidean')
    new_r_id = np.max(props[:, 0])+1
    num_to_process = props.shape[0]-1
    for idx in range(props.shape[0]-1):
        min_dist = dist_mat.min()
        min_dist_idxs = np.where(dist_mat == dist_mat.min())
        min_dist_idx = min_dist_idxs[0][0]
        min_dist_prev_idx = min_dist_idxs[1][0]
        if min_dist > max_dist:
            props_new[min_dist_idx+1, 0] = new_r_id
            new_r_id = new_r_id+1
        else:
            props_new[min_dist_idx+1, 0] = props_prev[min_dist_prev_idx+1, 0]
            # This ID has now been matched so block out
            dist_mat[min_dist_idx, :] = np.ones_like(dist_mat[min_dist_idx, :])*99999999
            dist_mat[:, min_dist_prev_idx] = np.ones_like(dist_mat[:, min_dist_prev_idx])*99999999
    # Create new image with correct labeling
    for idx in range(props_new.shape[0]-1):
        r_idx = idx+1
        img_new[img == props[r_idx, 0]] = props_new[r_idx, 0]
    return img_new, props_new

def plot_labelled_image (img, props, clist, output_filename=None, details=False):
    fig = plt.figure(figsize=(7.0, 7.0))
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0])
    ax.imshow(img, cmap=clist, vmax=MAX_CELLS)
    if details == True:
        for idx in range(props.shape[0]-1):
            region_idx = idx+1
            x0 = props[region_idx, CENTROID_X]
            y0 = props[region_idx, CENTROID_Y]
            orientation = props[region_idx, ORIENTATION]
            x2 = x0 - math.sin(orientation) * 0.5 * props[region_idx, MAJOR_LENGTH]
            y2 = y0 - math.cos(orientation) * 0.5 * props[region_idx, MAJOR_LENGTH]
            ax.plot((x0, x2), (y0, y2), '-r', linewidth=2, alpha=0.8)
            ax.plot(x0, y0, '.r', markersize=12, alpha=0.8)
            minr = props[region_idx, BBOX_MIN_X]-1
            minc = props[region_idx, BBOX_MIN_Y]-1
            maxr = props[region_idx, BBOX_MAX_X]
            maxc = props[region_idx, BBOX_MAX_Y]
            bx = (minc, maxc, maxc, minc, minc)
            by = (minr, minr, maxr, maxr, minr)
            ax.plot(bx, by, '-w', linewidth=1.0, alpha=0.5)
    # Sort out the formatting of the plot (fill entire frame)
    plt.subplots_adjust(hspace=.0 , wspace=.0, left=.1, right=.9, top=.9, bottom=.1)
    if output_filename == None:
        plt.show()
    else:
        fig.savefig(output_filename, transparent=True)
        plt.close('all')

def generate_cmap (max_number, seed=1):
    random.seed(seed)
    clist = [ (0,0,0) ]
    for idx in range(max_number):
        # Generate bright colours
        h,s,l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
        r,g,b = [int(256*i) for i in colorsys.hls_to_rgb(h,l,s)]
        clist.append( (r/256.0,g/256.0,b/256.0) )
    cm = LinearSegmentedColormap.from_list('cheetah', clist, N=max_number)
    return cm

###################################################################################################

# Colourmap to use (pre-generate so colours are fixed)
MAX_CELLS = 100000
cm = generate_cmap(MAX_CELLS, seed=1)

label, props = extract_mask_labels(images[0], background=0, connectivity=1, min_size=15)
plot_labelled_image(label, props, cm, output_filename='_output/image_0.png', details=False)
for idx in range(len(images)-1):
    new_label, new_props = extract_mask_labels(images[idx+1], background=0, connectivity=1, min_size=15)
    relabel, reprops = relabel_image(new_label, new_props, label, props, max_dist=10)
    plot_labelled_image(relabel, reprops, cm, output_filename='_output/image_'+str(idx+1)+'.png', details=False)
    label = relabel
    props = reprops

###################################################################################################
