import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow
from tensorflow import keras
from keras.layers import Dense, Flatten, Conv2D, Dropout,BatchNormalization
from keras.datasets import mnist
from PIL import Image
from converter import convert_im

# from tensorflow.python.client import device_lib
print(tensorflow.config.list_physical_devices('GPU'))


# model_path = 'model/trained_model_5k'
model_path = 'model/trained_model'

def create_model():
    '''
    Создаем модель нейросети, где
    *слово Sequential означает последовательную(то есть состоящую из нескольких слоёв)
    
    *Flatten преобразует матрицу(нашу картинку) размером 28 на 28 и, имеющую черно-белую цветовую схему, в один слой,
    состоящий из (28*28) = 784 нейронов (можно забить хуй)
    
    *Dense создает новый слой, где units - кол-во нейронов, activation - метод активации
    '''

    global model
    model = keras.Sequential([
        #Flatten(input_shape=(28, 28, 1)),       # Входной слой
        Conv2D(28, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)), # Входной и скрытый слой
        Conv2D(56, kernel_size=(3, 3), activation='relu'), # Скрытый слой(тут происходит вся магия, однако продвинутая)
        Conv2D(112, kernel_size=(3, 3), activation='relu'), # Скрытый слой(тут происходит вся магия, однако продвинутая)
        BatchNormalization(),
        Flatten(),
        Dense(units=128, activation='relu'), # Скрытый слой(тут происходит вся магия) 128 число наугад, можно ставить разные
        Dropout(0.2),
        Dense(units=10, activation='softmax')     # Выходной слой состоит из 10 нейронов, тк каждый отвечает за свою цифру
    ])

    # Компиляция модели
    # loss - критерий качества
    # optimizer - говорит сам за себя, оптимизируем нейросеть
    # метрика - метрика(определяем насколько мы правильно в % предсказали цифры)
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    return model

def train(model, x_train, x_test, y_train_cat, y_test_cat):
    '''
    Запуск процесса обучения.
    x_train - множество на котором тренируем.
    y_train_cat - ожидаемый результат (который хотим получить).
    batch_size - количество картинок после которых коэффициенты будут пересмотрены.
    epochs - количество эпох обучения
    validation_split - выбор сколько % картинок пойдут на валидацию (они уйдут на evaluate)
    '''
    model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)
    print("\n")
    

    # Валидация результата(проверка корректности)
    model.evaluate(x_test, y_test_cat)
    print("\n")
    pass

def main():
    global model , model_path# Костыль чтобы не ругалось
    (x_train, y_train), (x_test, y_test) = mnist.load_data()  # Выгрузка баз данных мниста(70 000 рукописных цифр)

    x_train = x_train / 255  # Стандартизация данных, картинки у нас черно-белые,
    x_test = x_test / 255  # поэтому данные мы будем хранить как 0-черный, 1-белый

    # Далее мы записываем каждую цифру как массив из десяти
    # Например цифру 4 как [0,0,0,0,1,0,0,0,0,0]
    #              а 6 как [0,0,0,0,0,0,1,0,0,0]
    y_train_cat = keras.utils.to_categorical(y_train, 10)
    y_test_cat = keras.utils.to_categorical(y_test, 10)

    # Выполнение функций

    # Проверка сохранённости модели
    if (not os.path.exists(model_path)):
        print('-|| Модель не найдена, создаю новую...')
        create_model()
        print('-|| Модель cоздана')
        print('-|| Обучаю модель...')
        train(model, x_train, x_test, y_train_cat, y_test_cat)
        print('-|| Модель обучена')
        print('-|| Сохраняю модель...')
        model.save(model_path)
        print('-|| Модель сохранена')
    else:
        print('-|| Модель найдена, загружаю...')
        model = keras.models.load_model(model_path)
        print('-|| Модель загружена')

    # Тест
    accuracy = np.array([0.0]*10)
    for i in range(0,10):
        cnt_files = 0
        for filename in os.listdir(f'images/raw/{i}'):
            # convert_im(f'images/raw/1/test_image{i}.jpg')#Исполнения конвертора в коде
            convert_im(f'images/raw/{i}/{filename}')
            im = Image.open('images/assets/active_test.jpg')
            data = np.array(im)
            ξ = np.expand_dims(data, axis=0)
            res = model.predict(ξ)


            if(i == np.argmax(res)):
                accuracy[i] += 1
            else:
                print(res)
                print(f'Я тут подумал, я отвечаю {i} это -', np.argmax(res))
                plt.imshow(data, cmap=plt.cm.binary, label=str(np.argmax(res)))
                plt.show()

            cnt_files+=1


        accuracy[i] /= (cnt_files)
    print('Я всё посчитал, таков итог')
    print(accuracy, np.mean(accuracy))
    return 0

if __name__ == '__main__':
    main()
    pass