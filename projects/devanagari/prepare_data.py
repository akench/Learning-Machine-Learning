import numpy as np
import PIL.Image as Image
import glob
from utils.parse_img import images_to_arrays
import pickle


# char_dirs = glob.glob('/home/super/Documents/devanagari/Test/*')
# char_dirs = glob.glob('/home/super/Documents/devanagari/digits_train/*')
char_dirs = glob.glob('/home/super/Documents/devanagari/Train/*')
if len(char_dirs) == 0:
	print('not a dir')
	quit()

all_data = []

for char in char_dirs:

	img_paths_per_char = glob.glob(char + '/*.png')


	image_objs = []
	for path in img_paths_per_char:

		with Image.open(path) as im:
			all_data = all_data + images_to_arrays([im])

	print(len(all_data))


print('dumping...')
pickle.dump(all_data, open('data/train_data.p', 'wb'))
