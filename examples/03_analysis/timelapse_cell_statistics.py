
import cheetah as ch
import math
import numpy as np
import glob as gl
import skimage.io
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

###################################################################################################

# This should be where the raw images are held
data_path = './data'

# Where the combined label files should be saved
output_path = './output'

def load_sorted_image_files (path, filetype='png'):
    images = []
    files = sorted(gl.glob(path + '/*.'+ filetype))
    for f in files:
        cur_image = skimage.io.imread(f)
        images.append(cur_image)
    return images

def load_sorted_mask_files (path, filetype='png'):
    images = []
    files = sorted(gl.glob(path + '/*.'+ filetype))
    for f in files:
        cur_image = skimage.io.imread(f)
        cur_mask = cur_image > 0
        images.append(cur_mask)
    return images

mask_imgs = load_sorted_mask_files(data_path+'/timelapse_masks', filetype='png')
fluo_imgs = load_sorted_image_files(data_path+'/timelapse_fluorescence', filetype='png')
phase_imgs = load_sorted_image_files(data_path+'/timelapse_phase_contrast', filetype='png')

###################################################################################################

"""
# Colourmap to use (pre-generate so colours are fixed)
MAX_CELLS = 1000000
cm = ch.generate_cmap(max_number=MAX_CELLS, seed=1)

label, props = ch.extract_mask_labels(mask_imgs[0], background=0, connectivity=1, min_size=15)
ch.plot_labelled_image(label, props, cm, output_filename=output_path+'/labelled_timelapse/image_0.png', details=True, max_number=MAX_CELLS)
for idx in range(len(mask_imgs)-1):
    new_label, new_props = ch.extract_mask_labels(mask_imgs[idx+1], background=0, connectivity=1, min_size=15)
    relabel, reprops = ch.relabel_image(new_label, new_props, label, props, max_dist=10)
    ch.plot_labelled_image(relabel, reprops, cm, output_filename=output_path+'/labelled_timelapse/image_'+str(idx+1)+'.png', details=True)
    label = relabel
    props = reprops
"""

###################################################################################################

# Colourmap to use (pre-generate so colours are fixed)
MAX_CELLS = 100000
cm = ch.generate_cmap(max_number=MAX_CELLS, seed=1)

label, props = ch.extract_mask_labels(mask_imgs[0], background=0, connectivity=1, min_size=15)

for img_idx in range(len(mask_imgs)-80): #REMOVE WHEN WORKING (SET TO -1)
    new_label, new_props = ch.extract_mask_labels(mask_imgs[img_idx+1], background=0, connectivity=1, min_size=15)
    relabel, reprops = ch.relabel_image(new_label, new_props, label, props, max_dist=10)    
    label = relabel
    props = reprops

    cell_num = ch.cell_count(reprops)
    #ss_avg_int = ch.single_cell_avg_intensities(fluo_imgs[idx], relabel, reprops)
    #avg_int = ch.mask_avg_intensity(fluo_imgs[idx], relabel, reprops)
    #print(idx, cell_num, avg_int, ss_avg_int)

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 3)
    
    #################################################################################
    # Phase contrast
    ax = plt.subplot(gs[0])
    ax.imshow(phase_imgs[img_idx+1], cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])

    #################################################################################
    # Mask coloured
    ax = plt.subplot(gs[3])
    ax.imshow(relabel, cmap=cm, vmax=MAX_CELLS)
    for idx in range(props.shape[0]-1):
        region_idx = idx+1
        x0 = props[region_idx, ch.CENTROID_X]
        y0 = props[region_idx, ch.CENTROID_Y]
        orientation = props[region_idx, ch.ORIENTATION]
        x2 = x0 - math.sin(orientation) * 0.5 * props[region_idx, ch.MAJOR_LENGTH]
        y2 = y0 - math.cos(orientation) * 0.5 * props[region_idx, ch.MAJOR_LENGTH]
        ax.plot((x0, x2), (y0, y2), '-r', linewidth=1.2, alpha=0.8)
        ax.plot(x0, y0, '.r', markersize=7, alpha=0.8)
        minr = props[region_idx, ch.BBOX_MIN_X]-1
        minc = props[region_idx, ch.BBOX_MIN_Y]-1
        maxr = props[region_idx, ch.BBOX_MAX_X]
        maxc = props[region_idx, ch.BBOX_MAX_Y]
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        ax.plot(bx, by, '-w', linewidth=1.0, alpha=0.5)
    ax.set_xticks([])
    ax.set_yticks([])

    #################################################################################
    # Orientation plot
    ax = plt.subplot(gs[4], projection='polar')
    n_numbers = 100
    bins_number = 20  # the [0, 360) interval will be subdivided into this
    # number of equal bins
    bins = np.linspace(0.0, 2 * np.pi, bins_number + 1)
    #angles = 2 * np.pi * np.random.rand(n_numbers)
    angles = reprops[1:, ch.ORIENTATION]
    n, _, _ = plt.hist(angles, bins)
    width = 2 * np.pi / bins_number
    #ax = plt.subplot(1, 1, 1, projection='polar')
    bars = ax.bar(bins[:bins_number], n, width=width, bottom=0.0)
    ax.set_theta_offset(np.pi/2.0)
    for bar in bars:
        bar.set_alpha(0.5)

    # Format the subplots and save
    plt.subplots_adjust(hspace=.08, wspace=.08, left=.02, right=.98, top=.98, bottom=.02)
    fig.savefig('./output/analysis_timelapse/out'+str(img_idx+1)+'.png') #, transparent=True)
    plt.close('all')

###################################################################################################
