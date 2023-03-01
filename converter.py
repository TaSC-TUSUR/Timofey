'''
    Конвертер изображений
    должен преобразовывать изображения в удобные для нейросети
    (высококонтрастные черно-белые 28 на 28 пикселей)
'''

import numpy as np
from PIL import Image, ImageOps

def convert_im(path):
    η = 215 #Ползунок для изменения вхождения чёрного
    τ = 45 # Допуск цвета

    img = Image.open(path)
    arr = np.array(img)
    im_ht = len(arr)
    im_wt = len(arr[0])

    # Мы блять надеемся на то, фон белый или серый, а у него соотношение цветов приерно одно и тоже,
    # Например r,g,b = 130 126 123 - это сероватый цвет максимальная разница 130-123 = 7, что немного
    # А у цвета r,g,b = 30 50 223 - это один из синих максимальная разница = 223-30 = 193, что дохуя

    aver_rgb_sum = 0
    for i in range(im_ht):
        for j in range(im_wt):
            r, g, b = map(int, arr[i, j])
            aver_rgb_sum += r+g+b

    aver_rgb_sum /= im_ht*im_wt

    τ = aver_rgb_sum

    for i in range(im_ht):
        for j in range(im_wt):
            if abs(r - g) <= τ \
                and abs(r - b) <= τ\
                and abs(g - b) <= τ\
                and sum(arr[i, j]) > η: # Проверим ещё черный цвет

                    arr[i, j] = [255, 255, 255]
            else:
                arr[i, j] = [0, 0, 0]

    arr = np.array(Image.fromarray(arr))
    img = ImageOps.invert(Image.fromarray(arr)).resize((28, 28)) # Делаем его 28*28 как в базе данных
    img = img.convert('L') # Делаем чб

    img.save('images/assets/test.jpg')
    arr = np.array(img)
    # print(arr)
    return arr  # Надо будет что-то придумать

    # return img_array # Надо будет что-то придумать

convert_im('images/raw/test_image2.jpg')
