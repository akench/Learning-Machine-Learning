import PIL.ImageOps
from PIL import Image
import numpy as np
from random import *

def resize_crop(img, save_path = None, crop_type = 'center', size = None, grey = True):
	'''
	Args:
		img : image to crop and resize
	 	save_path : where to save image, if not None
		crop_type : 'center' or 'random'
	 	size : width of resized img
	Returns:
		cropped and resized image
	'''

	if not isinstance(img, Image.Image):
		try:
			img = Image.open(img)
		except IOError:
			raise ValueError('file %s not found' % (img))

	if grey:
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

			if w//8 > w-h:
				start_width = randint(0, w - h)
			else:
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

			if h // 8 > h - w:
				start_height = randint(0, h-w)
			else:
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


	if size is not None:
		img = img.resize((size, size), PIL.Image.ANTIALIAS)

	if save_path is not None:
		img.save(save_path)

	return img



def images_to_arrays(image_list):
	'''
	Args:
		list of image objects
	Returns:
		list of list of pixels for each image
		...each image is in each row
	'''
	return [np.array(img).flatten() for img in image_list]





def normalize_data(data):

	'''
	Args:
		2D array with arr storing each image, and arr[i] storing pixels of image i
	Returns:
		normalized data, mean of data, standard deviation of data
	'''
	m = np.mean(data, axis = 0)
	std = np.std(data, axis = 0)

	data -= m
	data /= std

	return data, m, std


def stretch_img(img, stretch_type, factor):
	'''
	Args:
		img : PIL image object
		stretch_type : w or h, horizontal or vertical stretch?
		factor : stretch factor
	Returns:
		stretched image
	'''
	w, h = img.size

	if stretch_type == 'w':
		return img.resize((int(w * factor), h))
	else:
		return img.resize((w, int(h * factor)))


def salt_and_pepper(image, prob):
	import math
	'''
	Args:
		img : PIL image object
		prob : probability of salt and pepper being added
	Returns:
		Image with salt and pepper
	'''
	arr = images_to_arrays([image])
	arr = np.reshape(arr, (image.size))
	output = np.zeros(image.size, np.uint8)
	thres = 1 - prob
	for i in range(image.size[0]):
		for j in range(image.size[1]):
			rdn = random()
			if rdn < prob:
				output[i][j] = randint(0, 127)
			elif rdn > thres:
				output[i][j] = randint(128, 255)
			else:
				output[i][j] = arr[i][j]

	output = Image.fromarray(output)
	return output


def rand_rotate_and_crop(file_paths_list, rots_per_img = 10, crops_per_rot = 5):

	'''
	Args:
		list of file paths with images to process
		number of random rotations per image
		number of random crops per rotation
	Returns:
		list of Pillow image objects
			number of processed images = 2 * len(file_paths_list) * rots_per_img * crops_per_rot
	'''

	step = 0
	processed_images = []

	for path in file_paths_list:
		print(path)
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
