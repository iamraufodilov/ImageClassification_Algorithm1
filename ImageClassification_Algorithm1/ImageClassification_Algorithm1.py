# loading libraries
import numpy as np
import pandas as pd
import cv2
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from matplotlib import pyplot as plt
import matplotlib.image as img

#let check five image from file directory
plt.figure(figsize=(20,20))
test_folder='G:/rauf/STEPBYSTEP/Data/IntelImageClassification/seg_train/seg_train/forest'
for i in range(5):
    file = np.random.choice(os.listdir(test_folder))
    image_path = os.path.join(test_folder, file)
    my_img = img.imread(image_path)
    ax= plt.subplot(1,5,i+1)
    ax.title.set_text(file)
    plt.imshow(my_img)

# for one picture independent
my_path = 'G:/rauf/STEPBYSTEP/Data/IntelImageClassification/seg_train/seg_train/forest/8.jpg'
this_img = img.imread(my_path)
print(this_img)
plotted_img = plt.imshow(this_img)
print(plotted_img)

# setting image dimension
IMG_WIDTH=200
IMG_HEIGHT=200
img_folder = 'G:/rauf/STEPBYSTEP/Data/IntelImageClassification/seg_train/seg_train'

def create_dataset_tf(img_folder):
    class_name = []
    tf_image_data_array = []

    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
            image = os.path.join(img_folder, dir1, file)
            image = tf.io.read_file(image)
            image = tf.io.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, (200,200))
            image = tf.cast(image/255., tf.float32)
            tf_image_data_array.append(image)
            class_name.append(dir1)
    return tf.stack(tf_image_data_array, axis=0), class_name

tf_img_data, class_name = create_dataset_tf(img_folder)
print(type(tf_img_data))
print(type(class_name))

target_dict = {k: v for v, k in enumerate(np.unique(class_name))}
print(target_dict)

target_val = [target_dict[class_name[i]] for i in range(len(class_name))]

model = keras.layers.Sequential([
    tf.keras.layers.InputLayer(input_shape=(200,200,3)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2,2), activation='relu'),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2,2), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32)
    ])

model.summary()

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x=tf_img_data, y=tf.cast(list(map(int,target_val)),tf.int32), epochs=2)