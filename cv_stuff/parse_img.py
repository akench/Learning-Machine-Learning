import PIL.ImageOps
from PIL import Image
import numpy as np
from random import *

def resize_crop(img, save_path = None, crop_type = 'center', size = 28):



	if not isinstance(img, Image.Image):
		try:
			img = Image.open(img).convert('L')
		except IOError:
			print('file_path', file_path)
			print('File not found')
			return None


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
		img = img.rotate(-90)
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



	img = img.resize((size, size), PIL.Image.ANTIALIAS)

	# print('resized img to %d by %d' % (img.size[0], img.size[1]))

	if save_path is not None:
		img.save(save_path)



	return img
