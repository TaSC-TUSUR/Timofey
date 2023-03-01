'''
    Конвертер изображений
    должен преобразовывать изображения в удобные для нейросети
    (высококонтрастные черно-белые 28 на 28 пикселей)
'''

import numpy as np
from PIL import Image, ImageOps

def convert_im(path):
    n = 150
    img = Image.open(path)
    arr = np.array(img)

    # Мы блять надеемся на то, фон белый или серый, а у него соотношение цветов приерно одно и тоже,
    # Например r,g,b = 130 126 123 - это сероватый цвет максимальная разница 130-123 = 7, что немного
    # А у цвета r,g,b = 30 50 223 - это один из синих максимальная разница = 223-30 = 193, что дохуя
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            r, g, b = map(int, arr[i, j])
            if abs(r - g) <= 30 \
                and abs(r - b) <= 30\
                and abs(g - b) <= 30\
                and sum(arr[i, j]) > n: # Проверим ещё черный цвет

                    arr[i, j] = [255, 255, 255]
            else:
                arr[i, j] = [0, 0, 0]

    arr = np.array(Image.fromarray(arr))
    img = ImageOps.invert(Image.fromarray(arr)).resize((28, 28)) # Делаем его 28*28 как в базе данных
    img = img.convert('L') # Делаем чб

    img.save('images/assets/test.jpg')

    # return img_array # Надо будет что-то придумать

convert_im('images/raw/test_image1.jpg')
