from cv_stuff.parse_img import resize_crop
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

for i in range(5):
    img = resize_crop(img = 'org_data/pianopic.jpg', save_path = 'processed_data/test_' + str(i) + '.jpg',
        crop_type='random')
#
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
