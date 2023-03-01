import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from converter import convert_im
from keras.layers import Dense, Flatten
from keras.datasets import mnist
from pprint import pprint

from PIL import Image


(x_train, y_train), (x_test, y_test) = mnist.load_data()  # Выгрузка баз данных мниста(70 000 рукописных цифр)

x_train = x_train / 255 # Стандартизация данных, картинки у нас черно-белые,
x_test = x_test / 255   # поэтому данные мы будем хранить как 0-черный, 1-белый

# im = Image.fromarray(x_train[5])
# im.show()

# ----------------------------------------------------------
# Далее мы записываем каждую цифру как массив из десяти
# Например цифру 4 как [0,0,0,0,1,0,0,0,0,0]
#              а 6 как [0,0,0,0,0,0,1,0,0,0]
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)
# ----------------------------------------------------------

# ----------------------------------------------------------
# Создаем модель нейросети, где
# *слово Sequential означает последовательную(то есть состоящую из нескольких слоёв)
#
# *Flatten преобразует матрицу(нашу картинку) размером 28 на 28 и, имеющую черно-белую цветовую схему, в один слой,
#   состоящий из (28*28) = 784 нейронов (можно забить хуй)
#
# *Dense создает новый слой, где units - кол-во нейронов, activation - метод активации
#
model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1)),  # Входной слой
    Dense(units=128, activation='relu'),  # Скрытый слой(тут происходит вся магия) 128 число наугад, можно ставить разные
    Dense(units=10, activation='softmax') # Выходной слой состоит из 10 нейронов, тк каждый отвечает за свою цифру
])
# ----------------------------------------------------------

# ----------------------------------------------------------
# Компиляция модели
# loss - критерий качества
# optimizer - говорит сам за себя, оптимизируем нейросеть
# метрика - метрика(определяем насколько мы правильно в % предсказали цифры)
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
# ----------------------------------------------------------

# ----------------------------------------------------------
# Запуск процесса обучения.
# x_train - множество на котором тренируем.
# y_train_cat - ожидаемый результат (который хотим получить).
# batch_size - количество картинок после которых коэффициенты будут пересмотрены.
# epochs - количество эпох обучения
# validation_split - выбор сколько % картинок пойдут на валидацию (они уйдут на evaluate)
model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)
print('\n')
# ----------------------------------------------------------

# ----------------------------------------------------------
# Валидация результата(проверка корректности)
model.evaluate(x_test, y_test_cat)
print('\n')
# ----------------------------------------------------------

# ----------------------------------------------------------
# Тест конкретной цифры
n = 1  # id в выборке mnist
convert_im('images/raw/test_image2.jpg')
im = Image.open('images/assets/test.jpg')
data = np.array(im)
x = np.expand_dims(data, axis=0)
res = model.predict(x)
print(res)
print('Я тут подумал, я те бля отвечаю это -',np.argmax(res))
plt.imshow(data, cmap=plt.cm.binary)
plt.show()
# ----------------------------------------------------------

# ----------------------------------------------------------
# Руки ещё пока не дошли
# # Распознавание всей тестовой выборки
# pred = model.predict(x_test)
# pred = np.argmax(pred, axis=1)
#
# print(pred.shape)
#
# print(pred[:20])
# print(y_test[:20])
#
# # Выделение неверных вариантов
# mask = pred == y_test
# print(mask[:10])
#
# x_false = x_test[~mask]
# y_false = x_test[~mask]
#
# print(x_false.shape)
#
# # Вывод первых 25 неверных результатов
# plt.figure(figsize=(10,5))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(x_false[i], cmap=plt.cm.binary)
#
# plt.show()
