
from skimage import io
from skimage.color import rgb2gray
import numpy as np
import glob as gl

###############################################################################
# CONVERT IMAGES TO GREYSCALE AND RELABEL

def convert_to_greyscale (filename_in, filename_out):
	img_color = io.imread(filename_in)
	img_greyscale = rgb2gray(img_color)
	io.imsave(fname=filename_out, arr=img_greyscale)

def convert_to_labels (filename_in, filename_out):
	img_color = io.imread(filename_in)
	img_greyscale = rgb2gray(img_color)
	label_map = {}
	label_map[0.0] = 0
	cur_label = 1
	unique_vals = np.unique(img_greyscale)
	for u in unique_vals:
		if u not in label_map.keys():
			label_map[u] = cur_label
			cur_label = cur_label+1
	img_labels = np.empty_like(img_greyscale, dtype='uint8')
	for c in label_map.keys():
		img_labels[img_greyscale == c] = label_map[c]
	io.imsave(fname=filename_out, arr=img_labels)
	
def process_dir (files_path, output_path, fn_to_apply):
	files = gl.glob(files_path)
	for f in files:
		filename = (f.split('/')[-1]).strip()
		filename_out = output_path + '/' + filename
		fn_to_apply(f, filename_out)

# Convert images to greyscale
process_dir('./data/train/images/*.png', './data/train/images_converted', convert_to_greyscale)
process_dir('./data/validate/images/*.png', './data/validate/images_converted', convert_to_greyscale)

# Convert colour labels to integers
process_dir('./data/train/labels/*.png', './data/train/labels_converted', convert_to_labels)
process_dir('./data/validate/labels/*.png', './data/validate/labels_converted', convert_to_labels)


###############################################################################
