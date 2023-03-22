'''
    Конвертер изображений
    должен преобразовывать изображения в удобные для нейросети
    (высококонтрастные черно-белые 28 на 28 пикселей)
'''


import numpy as np
import os
from PIL import Image, ImageOps
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

def convert_im(path):

    τ = 45  # Допуск цвета

    img = Image.open(path)
    arr = np.array(img)
    ش = len(arr) #Высота
    س = len(arr[0]) #Ширина

    # Мы надеемся на то, что фон белый или серый, т.к. у него соотношение цветов примерно одно и тоже
    # Например r,g,b = 130 126 123 - это оттенок серого, максимальная разница 130-123 = 7, что относительно немного
    # А у цвета r,g,b = 30 50 223 - это оттенок синего, максимальная разница = 223-30 = 193, что очень много

    aver_rgb_sum = 0
    for i in range(ش):
        for j in range(س):
            r, g, b = map(int, arr[i, j])
            aver_rgb_sum += r + g + b

    aver_rgb_sum /= ش * س * 3
    τ = aver_rgb_sum

    for i in range(ش):
        for j in range(س):
            cur_sum = sum(arr[i, j])/3
            # if cur_sum < n:
            #     pass
            if cur_sum > τ/1.7:
                arr[i, j] = [255, 255, 255]
            else:
                arr[i, j] = [0, 0, 0]

    arr = np.array(Image.fromarray(arr))
    img = ImageOps.invert(Image.fromarray(arr)).resize((28, 28))  # Меняем размер изображения на 28*28 пикселей, как в базе данных
    img = img.convert('L')  # Делаем чёрно-белое изображение

    img.save('images/assets/active_test.jpg')
    arr = np.array(img)
    # print(arr)
    return arr  # Надо будет что-то придумать

    # return img_array # Надо будет что-то придумать
