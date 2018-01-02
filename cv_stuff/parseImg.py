import PIL.ImageOps
from PIL import Image
import numpy as np


def resize(file_path):
	try:
		img = Image.open(file_path).convert('L')
	except IOError:
		print('File not found')
		quit()

	
	# #adds padding to make it a square
	# longer_side = max(img.size[0], img.size[1])
	# horizontal_padding = (longer_side-img.size[0]) // 2
	# vertical_padding = (longer_side-img.size[1]) // 2
	# img = img.crop((-horizontal_padding, -vertical_padding, img.size[0] + horizontal_padding, img.size[1] + vertical_padding))


	# #resizes image
	# basewidth = 28
	# wpercent = (basewidth / float(img.size[0]))
	# hsize = int((float(img.size[1]) * float(wpercent)))
	# img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)

	img = img.resize((28, 28), PIL.Image.ANTIALIAS)



	# path = ''
	# exten = ''
	# for i in range(len(file_path)):
	# 	if(file_path[i] != '.'):
	# 		path += file_path[i]
	# 	else:
	# 		exten = file_path[i : ]
	# 		break	

	# path = path + '_test' + exten

	# print('resized img to %d by %d' % (img.size[0], img.size[1]))

	img.save(file_path)
	return img