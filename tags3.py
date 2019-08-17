# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 20:52:56 2019

@author: Ринат
"""

import cv2
import pandas as pd
from matplotlib import pyplot as plt
import os
import numpy as np
from sklearn.model_selection import train_test_split

data_path = 'data/'
# используем данные из папок
shelf_images = 'images/ShelfImages/'
product_images = 'images/ProductImagesFromShelves/'

# прочитаем все наименования фото из ShelfImages
jpg_files = [f for f in os.listdir(f'{shelf_images}') if f.endswith('JPG')]
photos_df = pd.DataFrame([[f, f[:6], f[7:14]] for f in jpg_files], 
                         columns=['file', 'shelf_id', 'planogram_id'])
photos_df.head()

#jpg_files2 = [f for i in range(11) for f in os.listdir(f'{product_images}{i}') if f.endswith('png')]
#products_df = pd.DataFrame(
#    [[f[:18], f[:6], f[7:14], i, *map(int, f[19:-4].split('_'))] for i in range(11) for f in jpg_files2],
#    columns=['file', 'shelf_id', 'planogram_id', 
#             'category', 'xmin', 'ymin', 'w', 'h'])
products_df = pd.DataFrame(
    [[f[:18], f[:6], f[7:14], i, *map(int, f[19:-4].split('_'))] 
     for i in range(11) 
     for f in os.listdir(f'{product_images}{i}') if f.endswith('png')],
    columns=['file', 'shelf_id', 'planogram_id', 
             'category', 'xmin', 'ymin', 'w', 'h'])

products_df['xmax'] = products_df['xmin'] + products_df['w']
products_df['ymax'] = products_df['ymin'] + products_df['h']
products_df.head()

#Train/Validation/Test Split
#Делим по полкам!
# получим уникальные полки
shelves = list(set(photos_df['shelf_id'].values))
# использем train_test_split для разделения данных
shelves_train, shelves_validation, _, _ = train_test_split(
    shelves, shelves, test_size=0.3, random_state=6)
# пометим данные в df флагом is_train flag
def is_train(shelf_id): return shelf_id in shelves_train

photos_df['is_train'] = photos_df.shelf_id.apply(is_train)
products_df['is_train'] = products_df.shelf_id.apply(is_train)

# Данные содержат 11 классов. 
# Класс 0 мусор
# Класс 1 - Marlboro, 2 - Kent, 3 - Camel и тд
# Визуализируем, что в данные для обучения попали все классы
df = products_df[products_df.category != 0].\
         groupby(['category', 'is_train'])['category'].\
         count().unstack('is_train').fillna(0)
df.plot(kind='barh', stacked=True)

# сохраним данные
photos_df.to_pickle(f'{data_path}photos.pkl')
products_df.to_pickle(f'{data_path}products.pkl')

# функция для отрисовки полок с прямоуголниками продуктов
def draw_shelf_photo(file):
    file_products_df = products_df[products_df.file == file]
    coordinates = file_products_df[['xmin', 'ymin', 'xmax', 'ymax']].values
    im = cv2.imread(f'{shelf_images}{file}')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)    
    for xmin, ymin, xmax, ymax in coordinates:
        cv2.rectangle(im, (xmin, ymin), (xmax, ymax), (0, 255, 0), 5)
    plt.imshow(im)

# проверим данные
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
draw_shelf_photo('C3_P07_N1_S6_1.JPG')

# Распрознавание брэндов
# упростим задачу, - распознаем лишь 10 из них
from IPython.display import Image
Image('docs/images/brands.png', width=300)

# Попробуем использовать CNN для данной задачи.
# Используем https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py
# в качестве основы
# Keras это высокоуровневое API для построения нейронных сетей

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import pandas as pd
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import BatchNormalization
from keras.regularizers import l2
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras import backend as K

# загрузим данные с предыдущего шага
photos_df = pd.read_pickle(f'{data_path}photos.pkl')
products_df =  pd.read_pickle(f'{data_path}products.pkl')

# нейронные сети работают с вх данными одного размера, нам необходимо привести
# все изображения к одному. Размеры - это что-то вроде метапараметров, можно 
# попробовать различные варианты. Логично, что чем больше размеры
# тем лучше результаты. К сожалению это не так вследствие 
# переобучения. Чем больше параметров на входе НС, тем больше шанс
# переобучения
num_classes = 10
SHAPE_WIDTH = 80
SHAPE_HEIGHT = 120

# меняем размеры с исходного на SHAPE_WIDTH x SHAPE_HEIGHT
def resize_pack(pack):
    fx_ratio = SHAPE_WIDTH / pack.shape[1]
    fy_ratio = SHAPE_HEIGHT / pack.shape[0]    
    pack = cv2.resize(pack, (0, 0), fx=fx_ratio, fy=fy_ratio)
    return pack[0:SHAPE_HEIGHT, 0:SHAPE_WIDTH]

# x - картинка, y - класс, f - флаг тренировчного датасета
x, y, f = [], [], []
for file, is_train in photos_df[['file', 'is_train']].values:
    photo_rects = products_df[products_df.file == file]
    rects_data = photo_rects[['category', 'xmin', 'ymin', 'xmax', 'ymax']]
    im = cv2.imread(f'{shelf_images}{file}')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    for category, xmin, ymin, xmax, ymax in rects_data.values:
        if category == 0:
            continue
        pack = resize_pack(np.array(im[ymin:ymax, xmin:xmax]))
        x.append(pack)
        f.append(is_train)
        y.append(category - 1)

# выведем SHAPE_WIDTH x SHAPE_HEIGHT картинку, 
# очень сложно распознать даже глазами, что получится у НС
plt.imshow(x[60])
# делим данные на train/validation основываясь на флаге is_train flag
x = np.array(x)
y = np.array(y)
f = np.array(f)
x_train, x_validation, y_train, y_validation = x[f], x[~f], y[f], y[~f]
# сохраним картинки для валидации
x_validation_images = x_validation

# конвертируем y_train и y_validation в "one-hot" (категорийные) массивы
y_train = keras.utils.to_categorical(y_train, num_classes)
y_validation = keras.utils.to_categorical(y_validation, num_classes)
# нормализуем x_train, x_validation
x_train = x_train.astype('float32')
x_validation = x_validation.astype('float32')
x_train /= 255
x_validation /= 255

# что в итоге имеем
print('размер x_train:', x_train.shape)
print('размер y_train:', y_train.shape)
print(x_train.shape[0], 'количество примеров в train')
print(x_validation.shape[0], 'количество примеров в validation')

# Построим ResNet CNN. Практически не меняем пример в keras
# шаг обучения
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 5:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr
# описание слоя
def resnet_layer(inputs,
                 num_filters=16, # число фильтров
                 kernel_size=3,  # размер ядра
                 strides=1,      # шаг свертки
                 activation='relu', # функция активации
                 batch_normalization=True, # нормализация пакета
                 conv_first=True):         
    
    # сверточный (convolutional) слой 
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same', # метод обработки краев
                  kernel_initializer='he_normal', # инициализатор
                  kernel_regularizer=l2(1e-4))    # регуляризатор

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

# описываем сеть
def resnet_v1(input_shape, depth, num_classes=10):
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # описание модели
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=x_train.shape[1:]) # вход
    x = resnet_layer(inputs=inputs) # 1й слой 120х80х16
    # доопределим стек остаточных слоев
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # первый слой но не первый стэк
                strides = 2  # обратная свертка
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides) # 2й слой 60х40х32
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None) # 3й слой 30х20х64
            if stack > 0 and res_block == 0:  # первый слой но не первый стэк
                # линейная проекция остаточных прямых связей чтобы соотвествовать
                # изменению размеров
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # добавляем классификатор сверху
    # v1 не использует BN после последнего быстрого соединения -ReLU
    x = AveragePooling2D(pool_size=8)(x) # усреднение и суммирование
    y = Flatten()(x)                     # плоский слой
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # определяем модель
    model = Model(inputs=inputs, outputs=outputs)
    return model

n = 3
version = 1
if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2
model_type = 'ResNet%dv%d' % (depth, version)

model = resnet_v1(input_shape=x_train.shape[1:], depth=depth, num_classes=num_classes)
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)), metrics=['accuracy'])

# посмотрим архитектуру модели и какие тренировочные параметры она имеет
model.summary()

# Предобработка и приращение данных в реальном времени:
datagen = ImageDataGenerator(
    featurewise_center=False,  # установим среднее 0 по всему датасету
    samplewise_center=False,   # установим среднее 0 по каждому примеру
    featurewise_std_normalization=False,  # делим вх данные на станд отклон датасета
    samplewise_std_normalization=False,   # делим вх данные на станд отклон по каждому примеру
    zca_whitening=False,  # применим белый шум ZCA 
    rotation_range=5,  # произвольно поворачиваем картинки (градусы, 0 to 180)
    width_shift_range=0.1,  # произвольный сдвиг по горизонтали (% от общего числа данных)
    height_shift_range=0.1, # произвольный сдвиг по вертикали (% от общего числа данных)
    horizontal_flip=False,  # произвольно переворачиваем  по горизонтали
    vertical_flip=False)    # произвольно переворачиваем  по вертикали
datagen.fit(x_train)

# запустим процесс обучения, 15 эпох достаточно
batch_size = 50
epochs = 15
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    validation_data=(x_validation, y_validation),
                    epochs=epochs, verbose=1, workers=4, 
                    callbacks=[LearningRateScheduler(lr_schedule)],
                    steps_per_epoch=x_train.shape[0]/batch_size)

# проверяем результат на валидации
scores = model.evaluate(x_validation, y_validation, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# рисуем матрицу отклонений
def plot_confusion_matrix(cm, classes, normalize=False, 
                          title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
# рисуем матрицу отклонений для проверки качества распознавания
y_validation_cls = np.argmax(y_validation, axis=1)
y_validation_predict = model.predict(x_validation)
y_validation_predict_cls = np.argmax(y_validation_predict, axis=1) 

fig = plt.gcf()
fig.set_size_inches(10, 10)
cnf_matrix = confusion_matrix(y_validation_cls, y_validation_predict_cls)
plot_confusion_matrix(cnf_matrix, [f'C{i+1}' for i in range(num_classes)], 
                      title='Confusion matrix', normalize=True)  

# посмотрим самые репрезентативные случаи матрицы отклонений
power = np.array([y_validation_predict[i][y_validation_predict_cls[i]] 
                  for i in range(len(y_validation_predict_cls))])


margin = 5
width = num_classes * SHAPE_WIDTH + (num_classes - 1) * margin
height = num_classes * SHAPE_HEIGHT + (num_classes - 1) * margin
confusion_image = np.zeros((height, width, 3), dtype='i')
for i in range(num_classes):
    for j in range(num_classes):
        flags = [(y_validation_cls == i) & (y_validation_predict_cls == j)]
        if not np.any(flags):
            continue
        max_cell_power = np.max(power[flags])
        index = np.argmax(flags & (power == max_cell_power))
        ymin, xmin = (SHAPE_HEIGHT+margin) * i, (SHAPE_WIDTH+margin) * j
        ymax, xmax = ymin + SHAPE_HEIGHT, xmin + SHAPE_WIDTH
        confusion_image[ymin:ymax, xmin:xmax, :] = x_validation_images[index]
        
fig = plt.gcf()
fig.set_size_inches(20, 20)
plt.imshow(confusion_image)   

# Как сверточная НС видит мир)
def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

n = 4

margin = 5
width = n * SHAPE_WIDTH + (n - 1) * margin
height = n * SHAPE_HEIGHT + (n - 1) * margin
stitched_filters = np.zeros((height, width, 3), dtype='i')


input_img = model.input
# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])
layer_name = 'conv2d_5'
for i in range(n):
    for j in range(n):
        filter_index = i * n + j
        layer_output = layer_dict[layer_name].output
        loss = K.mean(layer_output[:, :, :, filter_index])
        grads = K.gradients(loss, input_img)[0]
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
        iterate = K.function([input_img], [loss, grads])
        input_img_data = np.random.random((1, SHAPE_HEIGHT, SHAPE_WIDTH, 3))
        input_img_data = input_img_data * 20 + 128.
        step = 1.
        for k in range(20):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step
        img = deprocess_image(input_img_data[0])
        ymin, xmin = (SHAPE_HEIGHT+margin) * i, (SHAPE_WIDTH+margin) * j
        ymax, xmax = ymin + SHAPE_HEIGHT, xmin + SHAPE_WIDTH
        stitched_filters[ymin:ymax, xmin:xmax, :] = img

fig = plt.gcf()
fig.set_size_inches(15, 15)
plt.imshow(stitched_filters)








