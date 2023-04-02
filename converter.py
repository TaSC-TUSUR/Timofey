'''
    Конвертер изображений
    должен преобразовывать изображения в удобные для нейросети
    (высококонтрастные черно-белые 28 на 28 пикселей)
    
'''


import numpy as np
import os
from PIL import Image, ImageFilter
import re
def rename_ims(directory):
    i = 0
    for filename in os.listdir(directory):
        os.rename(f'{directory}/{filename}',f'{directory}/{i}-temp.jpg')
        i += 1
    i = 0
    for filename in os.listdir(directory):
            os.rename(f'{directory}/{filename}',f'{directory}/test_image{i}.jpg')
            i+=1
    print(f"Renamed {directory} done")

for i in range(0, 10):
    rename_ims(f'images/raw/{i}')
rename_ims(f'images/raw/stuff')

def imageprepare(argv):

        im = Image.open(argv, mode="r").convert('L')
        width = float(im.size[0])
        height = float(im.size[1])
        newImage = Image.new('L', (28, 28), (255))  # создание белого холста размером 28x28 пикселей

        if width > height:  # check which dimension is bigger
            # Width is bigger. Width becomes 20 pixels.
            nheight = int(round((20.0 / width * height), 0))  # изменение размера высоты в соответствии с соотношением ширины
            if (nheight == 0):  # rare case but minimum is 1 pixel
                nheight = 1
                # resize and sharpen
            img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
            wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
            newImage.paste(img, (4, wtop))  # paste resized image on white canvas
        else:
            # Height is bigger. Heigth becomes 20 pixels.
            nwidth = int(round((20.0 / height * width), 0))  # изменение размера ширины в соответствии с соотношением высоты
            if (nwidth == 0):  # rare case but minimum is 1 pixel
                nwidth = 1
                # resize and sharpen
            img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
            wleft = int(round(((28 - nwidth) / 2), 0))  # Подсчёт вертикальных позиций
            newImage.paste(img, (wleft, 4))  # вставка изображения с измененным размером на белый холст

        # newImage.save("sample.png

        tv = list(newImage.getdata())  # получение значений в пикселях

        # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
        tva = [(255 - x) * 1.0 / 255.0 for x in tv]
        print(tva)

        return tva
