from cv_stuff.parse_img import resize_crop, images_to_arrays
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import glob
from random import *
import pickle



# >>> X -= np.mean(X, axis = 0) # zero-center
# >>> X /= np.std(X, axis = 0) # normalize





def preprocess(file_paths_list, rots_per_img = 10, crops_per_rot = 5):

    step = 0
    for path in files:
        img = Image.open(path)

        for _ in range(rots_per_img):

            rot = int(gauss(0, 1.4) * 13)
            rotated = img.rotate(rot)

            f = randint(0, 1)
            if f == 1:
                rotated = rotated.transpose(Image.FLIP_LEFT_RIGHT)

            for _ in range(crops_per_rot):
                i = resize_crop(img = rotated, crop_type='random')
                step += 1



piano_paths = glob.glob('org_data/p*.jpg')
print(piano_paths)







# quit()
# print(arr)
#

# plt.gray()
# plt.imshow(arr)
# plt.show()

# im = Image.open('org_data/pianopic.jpg')
#
#
# im = im.rotate(45)
# print(im)
#
# im = resize_crop(img = im, save_path = 'processed_data/test.jpg')

# im.save('processed_data/piano_test.jpg')
# im = resize_crop(img = 'processed_data/piano_test.jpg', save_path = 'processed_data/piano_crop.jpg')
