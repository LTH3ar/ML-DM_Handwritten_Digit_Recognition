import os
import joblib
import cv2
from os import system, name

# define our clear function
def clear():
 
    # for windows
    if name == 'nt':
        _ = system('cls')
 
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')

# load the model
model = joblib.load("model/digit_recognizer(0.9354838)")

# create a function to open the image
def open_image(img_path):
    im = cv2.imread(img_path)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (15, 15), 0)

    # Threshold the image
    ret, im_th = cv2.threshold(im_gray, 100, 255, cv2.THRESH_BINARY)

    X = []

    for i in range(im_th.shape[0]):
        for j in range(im_th.shape[1]):
            k = im_th[i, j]
            if k > 100:
                k = 1
            else:
                k = 0
            X.append(k)

    predictions = model.predict([X])
    return predictions[0]

# scan local folder for images
dirList = os.listdir("test_data")
test_lst = []
for img_path in dirList:
    result = open_image("test_data/" + img_path)
    test_lst.append(str(f"{img_path} Output: {result}"))

clear()

for result in test_lst:
    print(result)



