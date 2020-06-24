 
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

mask_imgs = load_sorted_mask_files(data_path+'/timelapse_mask', filetype='png')
fluo_imgs = load_sorted_image_files(data_path+'/timelapse_fluorescence', filetype='png')
phase_imgs = load_sorted_image_files(data_path+'/timelapse_phase_contrast', filetype='png')

###################################################################################################

cmap = {}
cmap['green']     = ( 78/255.0, 178/255.0, 101/255.0)
cmap['l_green']   = (144/255.0, 201/255.0, 135/255.0)
cmap['vl_green']  = (202/255.0, 224/255.0, 171/255.0)
cmap['blue']      = ( 25/255.0, 101/255.0, 176/255.0)
cmap['l_blue']    = ( 82/255.0, 137/255.0, 199/255.0)
cmap['vl_blue']   = (123/255.0, 175/255.0, 222/255.0)

fig_params_black = {'ytick.color' : 'k',
              'xtick.color' : 'k',
              'axes.labelcolor' : 'w',
              'axes.edgecolor' : 'k'}
fig_params_white = {'ytick.color' : 'w',
              'xtick.color' : 'w',
              'axes.labelcolor' : 'w',
              'axes.edgecolor' : 'w'}

# Colourmap to use (pre-generate so colours are fixed)
MAX_CELLS = 100000
cm = ch.generate_cmap(max_number=MAX_CELLS, seed=1)

label, props = ch.extract_mask_labels(mask_imgs[0], background=0, connectivity=1, min_size=15)
t_list = [0]
count_list = []
fluor_ss_avg_list = []
fluor_ss_low_list = []
fluor_ss_high_list = []
fluor_avg_list = []

for img_idx in range(len(mask_imgs)-2): #REMOVE WHEN WORKING (SET TO -2)
    new_label, new_props = ch.extract_mask_labels(mask_imgs[img_idx+1], background=0, connectivity=1, min_size=15)
    relabel, reprops = ch.relabel_image(new_label, new_props, label, props, max_dist=10)    
    label = relabel
    props = reprops

    count_list.append(ch.cell_count(reprops))
    fluor_avg_list.append(ch.mask_avg_intensity(fluo_imgs[img_idx+1], relabel, reprops))
    ss_avg_int = ch.single_cell_avg_intensities(fluo_imgs[img_idx+1], relabel, reprops)
    
    fluor_ss_avg_list.append(np.mean(ss_avg_int))
    fluor_ss_low_list.append(np.percentile(ss_avg_int, 25))
    fluor_ss_high_list.append(np.percentile(ss_avg_int, 75))

    fig = plt.figure(figsize=(9, 12), facecolor='black')
    gs = gridspec.GridSpec(4, 3)
    plt.rcParams.update(fig_params_black)

    #################################################################################
    # Cells - Phase contrast
    ax = plt.subplot(gs[0])
    ax.imshow(phase_imgs[img_idx+1], cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])

    #################################################################################
    # Cells - Fluorescence
    ax = plt.subplot(gs[3])
    ax.imshow(fluo_imgs[img_idx+1], cmap='viridis')
    ax.set_xticks([])
    ax.set_yticks([])

    #################################################################################
    # Mask coloured with bounding boxes and orientations
    ax = plt.subplot(gs[0:2, 1:])
    ax.imshow(relabel, cmap=cm, vmax=MAX_CELLS)
    for idx in range(props.shape[0]-1):
        region_idx = idx+1
        x0 = props[region_idx, ch.CENTROID_X]
        y0 = props[region_idx, ch.CENTROID_Y]
        orientation = props[region_idx, ch.ORIENTATION]
        x2 = x0 - math.sin(orientation) * 0.5 * props[region_idx, ch.MAJOR_LENGTH]
        y2 = y0 - math.cos(orientation) * 0.5 * props[region_idx, ch.MAJOR_LENGTH]
        ax.plot((x0, x2), (y0, y2), '-r', linewidth=1.1, alpha=0.8)
        ax.plot(x0, y0, '.r', markersize=6, alpha=0.8)
        minr = props[region_idx, ch.BBOX_MIN_X]-1
        minc = props[region_idx, ch.BBOX_MIN_Y]-1
        maxr = props[region_idx, ch.BBOX_MAX_X]
        maxc = props[region_idx, ch.BBOX_MAX_Y]
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        ax.plot(bx, by, '-w', linewidth=1.0, alpha=0.3)
    ax.set_xlim([0,255])
    ax.set_ylim([255,0])
    ax.set_xticks([])
    ax.set_yticks([])

    #################################################################################
    # Cell Count Profile
    
    plt.rcParams.update(fig_params_white)
    ax = plt.subplot(gs[2, :])
    ax.plot(t_list, count_list, color=cmap['l_blue'], linewidth=4)
    ax.set_ylabel('Cell count')
    ax.set_xlim([0, 1390])
    ax.set_ylim([0, 300])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_xticks([])

    #################################################################################
    # Fluorescence Profile
    ax = plt.subplot(gs[3, :])

    ax.fill_between(t_list, fluor_ss_low_list, fluor_ss_high_list, color=cmap['green'], alpha=0.3, zorder=-10)
    ax.plot(t_list, fluor_ss_avg_list, color=cmap['green'], linewidth=4, zorder=-5)

    ax.set_ylabel('GFP (a.u.)')
    ax.set_xlabel('Time (min)')
    #ax.set_yscale('log')
    ax.set_xlim([0, 1390])
    ax.set_ylim([0, 26000])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    """
    #################################################################################
    # Orientation plot - We don't use, but could be added easily
    ax = plt.subplot(gs[3], projection='polar')
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
    ax.set_xticks([])
    ax.set_yticks([])
    """

    # Format the subplots and save
    plt.subplots_adjust(hspace=.04, wspace=.02, left=.09, right=.99, top=.99, bottom=.04)
    fig.savefig('./output/analysis_timelapse/out'+str(img_idx+1)+'.png', facecolor=fig.get_facecolor(), transparent=True)
    plt.close('all')
    # Add the new time point
    t_list.append(t_list[-1]+5)

print(t_list[-1], max(fluor_ss_high_list), max(fluor_avg_list))
###################################################################################################
