import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Dense, Flatten
from keras.datasets import mnist
from pprint import pprint
from PIL import Image
from converter import convert_im

def create_model():

    # Создаем модель нейросети, где
    # *слово Sequential означает последовательную(то есть состоящую из нескольких слоёв)
    #
    # *Flatten преобразует матрицу(нашу картинку) размером 28 на 28 и, имеющую черно-белую цветовую схему, в один слой,
    #   состоящий из (28*28) = 784 нейронов (можно забить хуй)
    #
    # *Dense создает новый слой, где units - кол-во нейронов, activation - метод активации

    global model
    model = keras.Sequential([
        Flatten(input_shape=(28, 28, 1)),       # Входной слой
        Dense(units=128, activation='relu'),     # Скрытый слой(тут происходит вся магия) 128 число наугад, можно ставить разные
        Dense(units=10, activation='softmax')     # Выходной слой состоит из 10 нейронов, тк каждый отвечает за свою цифру
    ])

    # Компиляция модели
    # loss - критерий качества
    # optimizer - говорит сам за себя, оптимизируем нейросеть
    # метрика - метрика(определяем насколько мы правильно в % предсказали цифры)
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    return model

def train(model, x_train, x_test, y_train_cat, y_test_cat):

    # Запуск процесса обучения.
    # x_train - множество на котором тренируем.
    # y_train_cat - ожидаемый результат (который хотим получить).
    # batch_size - количество картинок после которых коэффициенты будут пересмотрены.
    # epochs - количество эпох обучения
    # validation_split - выбор сколько % картинок пойдут на валидацию (они уйдут на evaluate)
    model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)
    print("\n")

    # Валидация результата(проверка корректности)
    model.evaluate(x_test, y_test_cat)
    print("\n")
    pass

def main():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()  # Выгрузка баз данных мниста(70 000 рукописных цифр)

    x_train = x_train / 255  # Стандартизация данных, картинки у нас черно-белые,
    x_test = x_test / 255  # поэтому данные мы будем хранить как 0-черный, 1-белый

    # Далее мы записываем каждую цифру как массив из десяти
    # Например цифру 4 как [0,0,0,0,1,0,0,0,0,0]
    #              а 6 как [0,0,0,0,0,0,1,0,0,0]
    y_train_cat = keras.utils.to_categorical(y_train, 10)
    y_test_cat = keras.utils.to_categorical(y_test, 10)

    # Выполнение функций
    create_model()
    train(model, x_train, x_test, y_train_cat, y_test_cat)

    # Тест
    for i in range(1,7):
        convert_im(f'images/raw/test_image{i}.jpg')
        im = Image.open('images/assets/test.jpg')
        data = np.array(im)
        ξ = np.expand_dims(data, axis=0)
        res = model.predict(ξ)
        print(res)
        print(f'Я тут подумал, я те бля отвечаю {i} это -', np.argmax(res))
        plt.imshow(data, cmap=plt.cm.binary)
        plt.show()

    return 0

if __name__ == '__main__':
    main()