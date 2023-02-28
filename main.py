import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Dense, Flatten
from keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test =  x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

model = keras. Sequential([
    Flatten(input_shape=(28,28,1)),
    Dense(units=128, activation='relu'),
    Dense(units=10, activation='softmax')
        ])

model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)
print("\n")

model.evaluate(x_test, y_test_cat)
print("\n")

# Тест конкретной цифры хуй
n = 1 # id в выборке mnist
x = np.expand_dims(x_test[n], axis=0)
res = model.predict(x)
print( res )
print( np.argmax(res) )

plt.imshow(x_test[n], cmap=plt.cm.binary)
plt.show()

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
