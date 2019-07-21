
# Everything we need is in the cheetah package
import cheetah as ch

# Load a single image and corrisponding label set (ground truth)
single_image = ch.load_image('./data/test_image.png', normalization='max')
print(single_image.dtype)
print(single_image.shape)

single_label_set = ch.load_label_set('./data/test_image_labels.png', 2)
print(single_label_set.dtype)
print(single_label_set.shape)

images = ch.load_images('./data/train/images', normalization='max', file_type=('.png', '.tif'))
print(images.dtype)
print(images.shape)

label_sets = ch.load_label_sets('./data/train/labels', 2, file_type=('.png', '.tif'))
print(label_sets.dtype)
print(label_sets.shape)
