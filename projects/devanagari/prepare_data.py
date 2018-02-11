import numpy as np
import PIL.Image as Image
import glob
from utils.parse_img import images_to_arrays
import pickle


char_dirs = glob.glob('/home/super/Downloads/devanagari/Test/*')


all_data = []

for char in char_dirs:

	img_paths_per_char = glob.glob(char + '/*.png')


	image_objs = []
	for path in img_paths_per_char:

		im = Image.open(path)
		image_objs.append(im)


	data = images_to_arrays(image_objs)
	all_data = all_data + data
	print(len(all_data))



pickle.dump(all_data, open('data/train_data.p', 'wb'))
