'''
    Конвертер изображений
    должен преобразовывать изображения в удобные для нейросети
    (высококонтрастные черно-белые 28 на 28 пикселей)
'''


import numpy as np
from PIL import Image, ImageOps


def convert_im(path):
    τ = 45  # Допуск цвета

    img = Image.open(path)
    arr = np.array(img)
    ش = len(arr) #Высота
    س = len(arr[0]) #Ширина

    # Мы блять надеемся на то, фон белый или серый, а у него соотношение цветов приерно одно и тоже,
    # Например r,g,b = 130 126 123 - это сероватый цвет максимальная разница 130-123 = 7, что немного
    # А у цвета r,g,b = 30 50 223 - это один из синих максимальная разница = 223-30 = 193, что дохуя

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
    img = ImageOps.invert(Image.fromarray(arr)).resize((28, 28))  # Делаем его 28*28 как в базе данных
    img = img.convert('L')  # Делаем чб

    img.save('images/assets/active_test.jpg')
    arr = np.array(img)
    # print(arr)
    return arr  # Надо будет что-то придумать

    # return img_array # Надо будет что-то придумать
