import os
import cv2
from os import system, name
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical


# define our clear function
def clear():
    # for windows
    if name == 'nt':
        _ = system('cls')

    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')

# load the model
model = load_model('mnist_model.keras', compile=False)

# create a function to open the image
def open_image():
    folder_path = r"C:\Users\Raven\PycharmProjects\mnist_dataset\test_data"
    for filename in os.listdir(folder_path):
        img = image.load_img(os.path.join(folder_path, filename), color_mode="grayscale", target_size=(28, 28))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img.astype('float32') / 255
        img = 1 - img
        prediction = model.predict(img)
        print(filename)
        print(np.argmax(prediction))
        print("\n")

open_image()
