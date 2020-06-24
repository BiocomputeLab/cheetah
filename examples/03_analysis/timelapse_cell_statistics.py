
import cheetah as ch
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

def load_sorted_files (path, filetype='png'):
    images = []
    files = sorted(gl.glob(path + '/*.'+ filetype))
    for f in files:
        cur_image = skimage.io.imread(f)
        cur_mask = cur_image > 0
        images.append(cur_mask)
    return images

mask_imgs = load_sorted_files(data_path+'/timelapse_masks', filetype='png')
fluo_imgs = load_sorted_files(data_path+'/timelapse_fluorescence', filetype='png')

###################################################################################################

# Colourmap to use (pre-generate so colours are fixed)
MAX_CELLS = 100000
cm = ch.generate_cmap(max_number=MAX_CELLS, seed=1)

label, props = ch.extract_mask_labels(mask_imgs[0], background=0, connectivity=1, min_size=15)
ch.plot_labelled_image(label, props, cm, output_filename=output_path+'/labelled_timelapse/image_0.png', details=True, max_number=MAX_CELLS)
for idx in range(len(mask_imgs)-1):
    new_label, new_props = ch.extract_mask_labels(mask_imgs[idx+1], background=0, connectivity=1, min_size=15)
    relabel, reprops = ch.relabel_image(new_label, new_props, label, props, max_dist=10)
    ch.plot_labelled_image(relabel, reprops, cm, output_filename=output_path+'/labelled_timelapse/image_'+str(idx+1)+'.png', details=True)
    label = relabel
    props = reprops

###################################################################################################



###################################################################################################
