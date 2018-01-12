import PIL.ImageOps
from PIL import Image
import numpy as np
from random import *

def resize_crop(img, save_path = None, crop_type = 'center', size = 28):
	'''
	@param img : image to crop and resize
	@param save_path : where to save image, if not None
	@param crop_type : 'center' or 'random'
	@param size : width of resized img
	@return cropped and resized image
	'''

	if not isinstance(img, Image.Image):
		try:
			img = Image.open(img).convert('L')
		except IOError:
			raise ValueError('file %s not found' % (img))

	else:
		img = img.convert('L')


	w = img.size[0]
	h  = img.size[1]

	center_w = w // 2
	center_h = h // 2

	#if pix is horizontal
	if w > h:

		if crop_type == 'center':

			img = img.crop(
				(
					center_w - h // 2,
					0,
					center_w + h // 2,
					h
				)
			)
		elif crop_type == 'random':

			start_width  = randint(w//8, w - h)
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


	#img is vertical
	elif w < h:
		if crop_type == 'center':

			img = img.crop(
				(
					0,
					center_h - w // 2,
					w,
					center_h + w // 2
				)
			)
		elif crop_type == 'random':

			start_height  = randint(h // 8, h - w)
			img = img.crop(
				(
					0,
					start_height,
					w,
					start_height + w
				)
			)
		elif crop_type is None:
			pass
		else:
			raise ValueError("Invalid crop type")


	img = img.resize((size, size), PIL.Image.ANTIALIAS)

	if save_path is not None:
		img.save(save_path)

	return img



def images_to_arrays(image_list):
	'''
	@param list of image objects
	@return list of list of pixels for each image
			each image is in each row
	'''

	ret = [np.array(img).reshape((1, img.size[0] * img.size[1])).squeeze() for img in image_list]
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

def rand_rotate_and_crop(file_paths_list, rots_per_img = 10, crops_per_rot = 5):

	'''
	@param list of file paths with images to process
	@param number of random rotations per image
	@param number of random crops per rotation
	@return list of Pillow image objects
		number of processed images = 2 * len(file_paths_list) * rots_per_img * crops_per_rot
	'''

	step = 0
	processed_images = []

	for path in file_paths_list:
		img = Image.open(path)

		for _ in range(rots_per_img):

			rot = int(gauss(0, 1.4) * 13)
			rotated = img.rotate(rot)

			f = randint(0,1)
			if f == 0:
				rotated = rotated.transpose(Image.FLIP_LEFT_RIGHT)

			for _ in range(crops_per_rot):
				i = resize_crop(img = rotated, crop_type='random')
				processed_images.append(i)
				step += 1

	return processed_images
