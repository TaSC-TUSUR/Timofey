<h1 align="center">Timofey (R) v.0.1.1 01.03.2023</h1>
<h3 align="center">Полносвязная нейронная сеть, распознающая цифры / Fully connected number recognition neural network</h3>
<p align="center"><img src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&amp;logo=TensorFlow&amp;logoColor=white" alt="TensorFlow"></p>
<h2>Описание</h2>
Распознавание цифр с помощью нейронной сети в Python — важная составляющая искусственного интеллекта. Это используется для распознавания изображений и текста, а также для машинного обучения, классификации объектов и предсказания будущих событий. В этой статье мы обсудим нейронные сети и подготовим нашу самую первую модель нейронной сети на Python для распознавания цифр. Для обучения нейронной сети мы будем использовать набор данных MNIST, содержащий изображения печатных цифр.
<br><br>
Digit recognition using a neural network in Python is an important component of artificial intelligence. This is used for image and text recognition, as well as for machine learning, object classification and prediction of future events. In this article, we will discuss neural networks and prepare our very first neural network model in Python for digit recognition. To train a neural network, we will use the MNIST dataset containing images of printed digits.
<br><br>
Вся информация о лицензии написано в файле LICENSE / All information about the license is written in the LICENSE file

<br>

<h2>История версий</h2>
<h3>Version 0.1</h3>
<ul>
 <li>Конвентер (conventer.py) для перевода изображений в удобный для нейросети вид;
  <ul>
    <li>Переводит изображение в высококонтрастное чёрно-белое, после чего сохраняется отдельным файлом;</li>
  </ul>
 </li>
 <li>Главная программа (main.py);
  <ul>
    <li>Берёт изображение, созданное конвентером, для обработки, в результате выводит ответ — число;</li>
  </ul>
 </li>
 <li>Добавлена папка images для удобного и структурированного хранения изображений, используемых для обучения и подачи (images/raw) и для использования самой программой (images/assets);</li>
 <li>Переработана структура файла README.MD для удобного ознакомления;
  <ul><li>Добавлены пункты "Описание" и "История версий".</li></ul>
</li>
</ul>
