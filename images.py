# # pip install tensorflow keras
# # pip install pillow
#
#
# import os
# import tensorflow as tf
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
# import numpy as np
#
# # Загрузите предобученную модель MobileNetV2 с весами ImageNet
# model = MobileNetV2(weights='imagenet')
#
#
# def classify_image(image_path):
#     # Загрузите изображение и преобразуйте его к размерам, подходящим для модели
#     img = image.load_img(image_path, target_size=(224, 224))
#
#     # Преобразуйте изображение в массив numpy и добавьте размерность для пакета данных
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#
#     # Предобработка изображения для модели
#     img_array = preprocess_input(img_array)
#
#     # Сделайте предсказание с помощью модели
#     predictions = model.predict(img_array)
#
#     # Расшифруйте предсказания
#     decoded_predictions = decode_predictions(predictions, top=3)[0]
#
#     # Найдите классификацию кошки или собаки в предсказаниях
#     for prediction in decoded_predictions:
#         if 'dog' in prediction[0]:
#             return 'Собака'
#         elif 'cat' in prediction[0]:
#             return 'Кошка'
#
#     return 'Не удалось определить'
#
#
# # Путь к папке с изображениями
# image_folder_path = r'C:\Users\vmon_\OneDrive\Изображения\pictures'
#
# # Список имен файлов изображений (например, image1.jpg, image2.jpg, и т.д.)
# image_filenames = [
#     'cat119.jpg'
#
# ]
#
# # Пройдемся по каждому изображению и классифицируем его
# for filename in image_filenames:
#     # Полный путь к изображению
#     image_path = os.path.join(image_folder_path, filename)
#
#     # Классификация изображения
#     result = classify_image(image_path)
#
#     print(f'Изображение {filename}: {result}')
#
#
# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
#
# # параметры обучения
# batch_size = 32
# epochs = 10
#
# # путь к папке с изображениями
# image_dir = 'C:/Users/vmon_/OneDrive/Изображения/pictures'
#
# # размер изображений
# img_rows, img_cols = 224, 224
#
# # создание модели CNN
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_rows, img_cols, 3)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(2, activation='softmax'))
#
# # компиляция модели
# model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
#
# # генерация обучающих данных
# train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
# train_generator = train_datagen.flow_from_directory(image_dir, target_size=(img_rows, img_cols), batch_size=batch_size, class_mode='categorical', subset='training')
#
# # обучение модели
# model.fit(train_generator, steps_per_epoch=train_generator.samples // batch_size, epochs=epochs)
#
# # сохранение обученной модели
# model.save('my_model.h5')
#
# # генерация тестовых данных
# test_datagen = ImageDataGenerator(rescale=1./255)
# test_generator = test_datagen.flow_from_directory(image_dir, target_size=(img_rows, img_cols), batch_size=batch_size, class_mode='categorical', subset='validation')
#
# # оценка модели на тестовых данных
# test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
# print('Test accuracy:', test_acc)
#
# # обход всех файлов в папке
# for filename in os.listdir(image_dir):
#     # пропуск неизображений
#     if not filename.endswith('.jpg') and not filename.endswith('.png'):
#         continue
#     # загрузка изображения
#     img = tf.keras.preprocessing.image.load_img(os.path.join(image_dir, filename), target_size=(img_rows, img_cols))
#     # предобработка изображения
#     img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
#     img = np.expand_dims(img, axis=0)
#     # предсказание класса
#     pred = model.predict(img)
#     # определение класса
#     if pred[0][0] > pred[0][1]:
#         print(filename, ': кошка')
#     else:
#         print(filename, ': собака')
#
# import os
# import tensorflow as tf
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
# import numpy as np
#
# # Загрузите предобученную модель VGG16 с весами ImageNet
# model = VGG16(weights='imagenet')
#
#
# def classify_image(image_path):
#     # Загрузите изображение и преобразуйте его к размерам, подходящим для модели
#     img = image.load_img(image_path, target_size=(224, 224))
#
#     # Преобразуйте изображение в массив numpy и добавьте размерность для пакета данных
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#
#     # Предобработка изображения для модели
#     img_array = preprocess_input(img_array)
#
#     # Сделайте предсказание с помощью модели
#     predictions = model.predict(img_array)
#
#     # Расшифруйте предсказания
#     decoded_predictions = decode_predictions(predictions, top=3)[0]
#
#     # Найдите классификацию кошки или собаки в предсказаниях
#     for prediction in decoded_predictions:
#         if 'dog' in prediction[0]:
#             return 'Собака'
#         elif 'cat' in prediction[0]:
#             return 'Кошка'
#
#     return 'Не удалось определить'
#
#
# # Путь к папке с изображениями
# image_folder_path = r'C:\Users\vmon_\OneDrive\Изображения\pictures'
#
# # Список имен файлов изображений (например, image1.jpg, image2.jpg, и т.д.)
# image_filenames = [
#     'cat119.jpg',
#     'dog119.jpg',
#     'cat120.jpg',
#     'cat121.jpg',
#     'cat.jpg'
#
# ]
#
# # Пройдемся по каждому изображению и классифицируем его
# for filename in image_filenames:
#     # Полный путь к изображению
#     image_path = os.path.join(image_folder_path, filename)
#
#     # Классификация изображения
#     result = classify_image(image_path)
#
#     print(f'Изображение {filename}: {result}')

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# параметры обучения
batch_size = 32
epochs = 5

# путь к папке с изображениями
train_dir = 'E:/PycharmProjects/another_PJ/ZeroOne/base/train'
test_dir = 'E:/PycharmProjects/another_PJ/ZeroOne/base/test'

# размер изображений
img_rows, img_cols = 224, 224

# создание модели CNN
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_rows, img_cols, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

# компиляция модели
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# генерация обучающих данных
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# генератор данных для обучения
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical'
)

# расчет количества шагов на эпоху
steps_per_epoch = train_generator.samples // batch_size

# обучение модели
for epoch in range(epochs):
    train_generator.reset()  # Сброс генератора данных
    model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=1)

# сохранение обученной модели в формате Keras
model.save('my_model.keras')

# генерация тестовых данных
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical'
)

# оценка модели на тестовых данных
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print('Test accuracy:', test_acc)

# обход всех файлов в папке
for subdir in ['cats', 'dogs']:  # Обход папок cats и dogs в директории test_dir
    subdir_path = os.path.join(test_dir, subdir)
    for filename in os.listdir(subdir_path):
        # пропуск неизображений
        if not filename.endswith(('.jpg', '.png')):
            continue
        # загрузка изображения
        img = tf.keras.preprocessing.image.load_img(
            os.path.join(subdir_path, filename),
            target_size=(img_rows, img_cols)
        )
        # предобработка изображения
        img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        # предсказание класса
        pred = model.predict(img)
        # определение класса
        if pred[0][0] > pred[0][1]:
            print(filename, ': кошка')
        else:
            print(filename, ': собака')

