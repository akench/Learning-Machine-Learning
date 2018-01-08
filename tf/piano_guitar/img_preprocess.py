from cv_stuff.parse_img import resize_crop, images_to_arrays, normalize_data
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import glob
from random import *
import pickle



def preprocess(file_paths_list, rots_per_img = 10, crops_per_rot = 5):

    step = 0
    processed_images = []

    for path in file_paths_list:
        img = Image.open(path)

        for _ in range(rots_per_img):

            rot = int(gauss(0, 1.4) * 13)
            rotated = img.rotate(rot)

            f = randint(0, 1)
            if f == 1:
                rotated = rotated.transpose(Image.FLIP_LEFT_RIGHT)

            for _ in range(crops_per_rot):
                i = resize_crop(img = rotated, crop_type='random')
                processed_images.append(i)
                step += 1

    return processed_images


def make_data():
    piano_paths = glob.glob('org_data/guitar/*.jpg')
    guitar_paths = glob.glob('org_data/piano/*.jpg')

    piano_imgs = preprocess(piano_paths)
    guitar_imgs = preprocess(guitar_paths)

    piano_data = images_to_arrays(piano_imgs)
    guitar_data = images_to_arrays(guitar_imgs)

    piano_norm, mean, sd = normalize_data(piano_data)
    guitar_norm, mean, sd = normalize_data(guitar_data)

    pickle.dump(piano_norm, open('processed_data/piano_data.p', 'wb'))
    pickle.dump(guitar_norm, open('processed_data/guitar_data.p', 'wb'))

    print('Success! :)')


make_data()






# quit()
# print(arr)
#

# plt.gray()
# plt.imshow(arr)
# plt.show()
