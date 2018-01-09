import PIL.ImageOps
from PIL import Image
import numpy as np
from random import *

def resize_crop(img, save_path = None, crop_type = 'center', size = 28):


	if not isinstance(img, Image.Image):
		try:
			img = Image.open(img).convert('L')
		except IOError:
			print('file_path', img)
			print('File not found')
			return None

	else:
		img = img.convert('L')
	#makes it a square, crops out extra parts
	# shorter_side = min(img.size[0], img.size[1])
	# horizontal_padding = (shorter_side-img.size[0]) // 2
	# vertical_padding = (shorter_side-img.size[1]) // 2
	# img = img.crop((-horizontal_padding,
	# -vertical_padding,
	# img.size[0] + horizontal_padding,
	# img.size[1] + vertical_padding))


	w = img.size[0]
	h  = img.size[1]
	#img is now horizontal, width is longer than height
	if w < h:
		img = img.rotate(-90, expand = True)
		w, h = h, w

	if crop_type == 'center':
		center_w = w // 2
		center_h = h // 2

		img = img.crop(
			(
				center_w - h // 2,
				0,
				center_w + h // 2,
				h
			)
		)
	elif crop_type == 'random':

		start_width  = randint(0, w - h)
		img = img.crop(
			(
				start_width,
				0,
				start_width + h,
				h
			)
		)
	elif crop_type is None:
		pass
	else:
		raise ValueError("Invalid crop type")



	img = img.resize((size, size), PIL.Image.ANTIALIAS)

	# print('resized img to %d by %d' % (img.size[0], img.size[1]))

	if save_path is not None:
		img.save(save_path)

	return img


def images_to_arrays(image_arr):
	'''
	@param list of image objects
	@return list of list of pixels for each image
			each image is in each row
	'''

	ret = [np.array(img).reshape((1, 784)).squeeze() for img in image_arr]
	return ret




def normalize_data(data):

	'''
	@param 2D array with arr storing each image, and arr[i] storing pixels of image i
	@return normalized data, mean of data, standard deviation of data
	'''
	m = np.mean(data, axis = 0)
	sd = np.std(data, axis = 0)

	data -= m
	data /= sd

	return data, m, sd
