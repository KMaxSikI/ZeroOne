# # pip install tensorflow keras
# # pip install pillow


# Импорт необходимых библиотек и модулей
import os  # Работа с файловой системой и путями
import numpy as np  # Работа с массивами данных и численными операциями
import tensorflow as tf  # Основная библиотека для машинного обучения
from tensorflow.keras.models import Sequential  # Импорт класса для создания последовательной модели
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense  # Импорт слоев для архитектуры модели
from tensorflow.keras.optimizers import Adam  # Импорт оптимизатора Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Импорт класса для генерации данных изображений

# Определение параметров обучения
batch_size = 32  # Размер партии данных для каждой итерации обучения
epochs = 5  # Количество эпох обучения (проходов по обучающему набору данных)

# Путь к папке с изображениями для обучения и тестирования
train_dir = 'E:/PycharmProjects/another_PJ/ZeroOne/base/train'  # Папка с обучающими данными
test_dir = 'E:/PycharmProjects/another_PJ/ZeroOne/base/test'  # Папка с тестовыми данными

# Определение размера изображений для входа в модель
img_rows, img_cols = 224, 224  # Ширина и высота изображений в пикселях

# создание модели CNN
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_rows, img_cols, 3)))  # Добавление первого слоя сверточной сети (Conv2D) с 32 фильтрами, размером ядра 3x3 и активацией ReLU
# Указывается входная форма изображения (img_rows, img_cols, 3)
model.add(MaxPooling2D(pool_size=(2, 2)))  # Добавление слоя объединения (MaxPooling2D) с размером пула 2x2 для уменьшения размерности изображения
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu')) # Добавление второго слоя сверточной сети (Conv2D) с 64 фильтрами, размером ядра 3x3 и активацией ReLU
model.add(MaxPooling2D(pool_size=(2, 2))) # Добавление второго слоя объединения (MaxPooling2D) с размером пула 2x2
model.add(Flatten()) # Преобразование данных в одномерный вектор для передачи в полносвязные слои
model.add(Dense(128, activation='relu')) # Добавление полносвязного слоя с 128 нейронами и активацией ReLU
model.add(Dense(2, activation='softmax')) # Добавление выходного слоя с 2 нейронами для классификации на 2 класса (кошки и собаки) с активацией softmax

# Компиляция модели с функцией потерь 'categorical_crossentropy' для задачи классификации на несколько классов
# Используется оптимизатор Adam с заданной скоростью обучения (learning_rate=0.001)
# В качестве метрики для оценки модели используется точность (accuracy)
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Создание объекта генератора данных для обучения
# Этот объект позволяет применять аугментацию данных для увеличения разнообразия обучающих изображений
# Параметр rescale=1./255 используется для масштабирования значений пикселей из диапазона [0, 255] в [0, 1]
# Параметры shear_range=0.2, zoom_range=0.2 и horizontal_flip=True указывают на случайные преобразования для аугментации данных
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# Создание генератора данных для обучения с использованием объекта train_datagen
# Путь к папке с обучающими данными указывается в train_dir
# Параметр target_size=(img_rows, img_cols) задает размерность изображений
# Параметр batch_size указывает на размер батча (количество изображений, обрабатываемых за один раз)
# Параметр class_mode='categorical' указывает, что метки классов в данных представлены в формате one-hot encoding
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical'
)

# Расчет количества шагов на эпоху (steps_per_epoch)
# Оно вычисляется как целочисленное деление общего числа образцов в train_generator (train_generator.samples) на размер батча (batch_size)
# Этот параметр определяет, сколько итераций (шагов) должно быть выполнено за одну эпоху обучения
steps_per_epoch = train_generator.samples // batch_size

# обучение модели
for epoch in range(epochs):
    train_generator.reset()  # Сброс генератора данных, чтобы начать каждую эпоху заново
    # Обучение модели с использованием генератора данных train_generator
    # Указывается количество шагов на эпоху (steps_per_epoch) и количество эпох (1)
    model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=1)

# Сохранение обученной модели в формате Keras
# Файл модели будет сохранен в текущем каталоге с именем 'my_model.keras'
model.save('my_model.keras')

# Создание объекта генератора тестовых данных
# Параметр rescale=1./255 используется для масштабирования значений пикселей из диапазона [0, 255] в [0, 1]
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Создание генератора данных для тестирования с использованием объекта test_datagen
# Путь к папке с тестовыми данными указывается в test_dir
# Параметр target_size=(img_rows, img_cols) задает размерность изображений
# Параметр batch_size указывает на размер батча (количество изображений, обрабатываемых за один раз)
# Параметр class_mode='categorical' указывает, что метки классов в данных представлены в формате one-hot encoding
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical'
)

# Оценка модели на тестовых данных
# Функция model.evaluate() оценивает модель на тестовом наборе данных test_generator
# Аргумент steps указывает количество шагов (батчей) для выполнения оценки: деление общего числа образцов в тестовом наборе (test_generator.samples) на размер батча (batch_size)
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print('Test accuracy:', test_acc)


# Обходит папки cats и dogs в директории test_dir
for subdir in ['cats', 'dogs']:  # Обход папок cats и dogs в директории test_dir
    subdir_path = os.path.join(test_dir, subdir) # Строит путь к каждой папке (cats, dogs)
    for filename in os.listdir(subdir_path):
        if not filename.endswith(('.jpg', '.png')): # Пропускает файлы, которые не являются изображениями
            continue
        # Загрузка и изменение размера изображения до (img_rows, img_cols)
        img = tf.keras.preprocessing.image.load_img(
            os.path.join(subdir_path, filename),
            target_size=(img_rows, img_cols)
        )

        img = tf.keras.preprocessing.image.img_to_array(img) / 255.0 # Преобразует изображение в массив NumPy и масштабирует его в диапазон [0, 1]
        img = np.expand_dims(img, axis=0) # Добавляет ось к массиву, чтобы сделать его соответствующим размерности (1, img_rows, img_cols, 3)
        pred = model.predict(img) # Предсказание класса для изображения
        if pred[0][0] > pred[0][1]: # Сравнивает значения вероятностей для классов "cats" и "dogs"
            print(filename, ': кошка') # Выводит имя файла и определенный класс "кошка"
        else:
            print(filename, ': собака') # Выводит имя файла и определенный класс "собака"
