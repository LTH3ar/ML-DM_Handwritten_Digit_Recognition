import tkinter as tk
from tkinter import filedialog
import joblib
import cv2
import numpy as np
from PIL import Image, ImageTk

# a simple GUI for the digit recognizer
# load the model
model = joblib.load("model/digit_recognizer(0.90322)")

# create a window
window = tk.Tk()
window.title("Digit Recognizer")
window.geometry("300x500")

# create a area to display the image
image_area = tk.Label(window)
image_area.grid(row=0, column=0, padx=10, pady=10)

# create a area to display the prediction
prediction_area = tk.Label(window, text="Prediction: None")
prediction_area.grid(row=1, column=0, padx=10, pady=10)

# create a function to open the image
def open_image():
    img_path = filedialog.askopenfilename(title="Select an Image", filetypes=(("PNG Images", "*.png"), ("All Files", "*.*")))
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

    # Display the image
    img = Image.open(img_path)
    img = img.resize((280, 280))
    img = ImageTk.PhotoImage(img)
    image_area.configure(image=img)
    image_area.image = img

    # Display the prediction
    prediction_area.configure(text="Prediction: " + str(predictions[0]))

# create a button to open the image
open_image_button = tk.Button(window, text="Open Image", command=open_image)
open_image_button.grid(row=2, column=0, padx=10, pady=10)

# run the window
window.mainloop()

