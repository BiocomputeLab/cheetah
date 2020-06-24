
# Everything we need is in the cheetah package
import cheetah as ch

import numpy as np
from skimage import io
import glob as gl
import os
# Turn off TensorFlow warnings (should be fixed in future)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load a single image and corrisponding label set (ground truth)
timelapse_images, filenames = ch.load_images('./data/timelapse', normalization='max')

# Predict the labels of new image with no ground truth
predicted_labels = seg_predict.predict(timelapse_images)

output_path = './output/timelapse'

for idx in range(np.size(predicted_labels, axis=0)):
	cur_image = timelapse_images[idx]
	cur_labels = predicted_labels[idx]

	m_temp = np.argmax(cur_labels, axis=-1) + 1
	pred_mask_cells = (m_temp*(m_temp==2)) * (1.0/2)

	filename = filenames[idx].split('/')[-1]

	io.imsave(output_path + '/' + filename[:-4] + '_cells.png', pred_mask_cells)
	io.imsave(output_path + '/' + filename[:-4] + '_image.png', cur_image)

###############################################################################