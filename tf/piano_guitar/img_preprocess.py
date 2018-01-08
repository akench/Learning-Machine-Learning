from cv_stuff.parse_img import resize_crop
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import glob
from random import *



# >>> X -= np.mean(X, axis = 0) # zero-center
# >>> X /= np.std(X, axis = 0) # normalize


files = glob.glob('org_data/*.jpg')
step = 0

for path in files:
    img = Image.open(path)

    gausses = []
    for _ in range(10):

        rot = int(gauss(0, 1.4) * 13)
        gausses.append(rot)
        rotated = img.rotate(rot)

        f = randint(0, 1)
        if f == 1:
            rotated = rotated.transpose(Image.FLIP_LEFT_RIGHT)

        for _ in range(5):
            i = resize_crop(img = rotated, save_path = 'processed_data/test_' + str(step) + '.jpg', crop_type='random')
            step += 1

    gausses.sort()
    print(gausses)


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
