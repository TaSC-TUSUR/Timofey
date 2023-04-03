<!-- Версию менять в бейджике (ссылка в заголовке ниже) -->
<h1 align="center">Timofey® 03.04.2023 <img src="https://img.shields.io/badge/version-v1.0.0-blue" alt="version"></h1> 
<h3 align="center">Полносвязная нейронная сеть, распознающая цифры / Fully connected number recognition neural network</h3>
<p align="center"><img src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&amp;logo=TensorFlow&amp;logoColor=white" alt="TensorFlow">
<img src="https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&amp;logo=Keras&amp;logoColor=white" alt="Keras"></p>
<h2>Описание</h2>
Распознавание цифр с помощью нейронной сети в Python — важная составляющая искусственного интеллекта. Это используется для распознавания изображений и текста, а также для машинного обучения, классификации объектов и предсказания будущих событий. В этой статье мы обсудим нейронные сети и подготовим нашу самую первую модель нейронной сети на Python для распознавания цифр. Для обучения нейронной сети мы будем использовать набор данных MNIST, содержащий изображения печатных цифр.
<br><br>
Digit recognition using a neural network in Python is an important component of artificial intelligence. This is used for image and text recognition, as well as for machine learning, object classification and prediction of future events. In this article, we will discuss neural networks and prepare our very first neural network model in Python for digit recognition. To train a neural network, we will use the MNIST dataset containing images of printed digits.

<br><br>

<h2>История версий</h2>
<h3>Version 1.0.0</h3>
<ul>
 <li>Релиз!</li>
 <li>Конвертер (conventer.py), предназначен для перевода изображений в удобный для нейросети вид;
  <ul>
    <li>Переводит изображение в высококонтрастное чёрно-белое, после чего сохраняется отдельным файлом;</li>
  </ul>
 </li>
 <li>Главная программа (main.py);
  <ul>
    <li>Берёт изображение, созданное конвентером, для обработки, в результате строит таблицу весов и выводит ответ — число, вероятность которого наиболее высока;</li>
    <li>Присутсвует сохранение модели обучения (папка /model); 
    <ul>
      <li>Нейросети теперь не требуется заново обучаться при каждом запуске;</li>
    </ul>
    </li>
  </ul>
 </li>
 <li>Папка images предназначена для удобного и структурированного хранения изображений, используемых для обучения и подачи (images/raw) и для использования самой программой (images/assets);</li>
</ul> 

<br>

<h2>Лицензия</h2>
  <a href="https://github.com/TaSC-TUSUR/Timofey/blob/master/LICENSE" target="_blank">
        Apache License 2.0
      </a>

