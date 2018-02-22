import numpy as np
import PIL.Image as Image
import glob
from utils.parse_img import images_to_arrays
import pickle
from utils.parse_img import rand_rotate_and_crop


# char_dirs = glob.glob('/home/super/Documents/devanagari/Test/*')
char_dirs = glob.glob('/home/super/Documents/devanagari/temp/*')
# char_dirs = glob.glob('/home/super/Documents/devanagari/Train/*')
if len(char_dirs) == 0:
	print('not a dir')
	quit()

all_data = []

for char in char_dirs:

	img_paths_per_char = glob.glob(char + '/*.png')

	processed_imgs = rand_rotate_and_crop(img_paths_per_char, rots_per_img = 10, crops_per_rot = 1, rotation_factor=5)
	print(len(processed_imgs))
	all_data = all_data + images_to_arrays(processed_imgs)



print(len(all_data))

print('dumping...')
pickle.dump(all_data, open('data/train_data.p', 'wb'))
