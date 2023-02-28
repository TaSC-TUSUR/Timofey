'''
    Конвертер изображений
    должен преобразовывать изображения в удобные для нейросети
    (высококонтрастные черно-белые 28 на 28 пикселей)
'''

import numpy as np
from PIL import Image, ImageOps
im = Image.open('images/raw/test_image5.jpg')

im = im.convert('P')
arr = np.array(im)
arr[arr>135] = 255
arr[arr<=135] = 0
arr = np.array(Image.fromarray(arr))
im = ImageOps.invert(Image.fromarray(arr)).resize((28, 28))

im.save('images/assets/test.jpg')