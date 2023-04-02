'''
    Конвертер изображений
    должен преобразовывать изображения в удобные для нейросети
    (высококонтрастные черно-белые 28 на 28 пикселей)
'''


import numpy as np
import numpy
import os
from PIL import Image, ImageOps, ImageFilter
from matplotlib import pyplot as plt
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

# for i in range(0, 10):
#     rename_ims(f'images/raw/{i}')
# rename_ims(f'images/raw/stuff')

# def convert_im(path):
#
#     img = Image.open(path)
#     arr = np.array(img)
#     ش = len(arr) #Высота
#     س = len(arr[0]) #Ширина
#
#     # Мы надеемся на то, что фон белый или серый, т.к. у него соотношение цветов примерно одно и тоже
#     # Например r,g,b = 130 126 123 - это оттенок серого, максимальная разница 130-123 = 7, что относительно немного
#     # А у цвета r,g,b = 30 50 223 - это оттенок синего, максимальная разница = 223-30 = 193, что очень много
#
#     aver_rgb_sum = 0
#     for i in range(ش):
#         for j in range(س):
#             r, g, b = map(int, arr[i, j])
#             aver_rgb_sum += r + g + b
#
#     aver_rgb_sum /= ش * س * 3
#     τ = aver_rgb_sum
#
#     for i in range(ش):
#         for j in range(س):
#             cur_sum = sum(arr[i, j])/3
#             # if cur_sum < n:
#             #     pass
#             if cur_sum > τ/1.7:
#                 arr[i, j] = [255, 255, 255]
#             else:
#                 arr[i, j] = [0, 0, 0]
#
#     arr = np.array(Image.fromarray(arr))
#     img = ImageOps.invert(Image.fromarray(arr)).resize((28, 28))  # Меняем размер изображения на 28*28 пикселей, как в базе данных
#     img = img.convert('L')  # Делаем чёрно-белое изображение
#
#     img.save('images/assets/active_test.jpg')
#     arr = np.array(img)
#     # print(arr)
#     return arr  # Надо будет что-то придумать
#
#     # return img_array # Надо будет что-то придумать
#
# convert_im("images/raw/9/test_image8.jpg")
def imageprepare(argv):

    im = Image.open(argv, mode = "r").convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

    # newImage.save("sample.png

    tv = list(newImage.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    print(tva)

    return tva

 #Это чтобы смотреть во что конвертировалось
# x=[imageprepare('images/raw/8/test_image8.jpg')]#file path here          #Тут путь до картиночки которую смотришь
# print(len(x))# mnist IMAGES are 28x28=784 pixels
# print(x[0])
# newArr = [[0 for d in range(28)]for y in range(28)]
# k = 0
# for i in range(28):
#     for j in range(28):
#         newArr[i][j] = x[0][k]
#         k+=1
# for i in range(28):
#     for j in range(28):
#         print(newArr[i][j])
#     print("\n")
# plt.imshow(newArr,interpolation='nearest')
# plt.savefig('images/assets/active_test2.jpg')                            #Тут путь по которому сохраняешь конвертированную картиночку
# plt.show()