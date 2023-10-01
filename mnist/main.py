import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load a pre-trained MNIST digit classification model
model = load_model('mnist_model.keras', compile=False)

# Create a function to preprocess the input image
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28 pixels (MNIST image size)
    img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img = 1 - img  # Invert colors (MNIST digits are white on a black background)
    img = img.reshape(1, 28, 28, 1)  # Reshape for model input
    return img

# Create a function to predict the digit
def predict_digit(image_path):
    preprocessed_img = preprocess_image(image_path)
    prediction = model.predict(preprocessed_img)
    return np.argmax(prediction)

# Create a function to clear the old image area
def clear_image_area():
    for widget in root.winfo_children():
        if isinstance(widget, tk.Label) and widget != import_button:
            widget.destroy()

# Create a GUI application
def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        clear_image_area()  # Clear old image area
        prediction = predict_digit(file_path)
        display_image(file_path, prediction)

def display_image(image_path, prediction):
    image = Image.open(image_path)
    image.thumbnail((200, 200))  # Resize image for display
    img_label = ImageTk.PhotoImage(image=image)

    # Create a label to display the image
    image_label = tk.Label(root, image=img_label)
    image_label.image = img_label
    image_label.pack()

    # Create a label to display the prediction
    prediction_label = tk.Label(root, text=f'Predicted Digit: {prediction}', font=("Helvetica", 18))
    prediction_label.pack()

# Create the main application window
root = tk.Tk()
root.title("MNIST Digit Classifier")

# Create a button to open an image
import_button = tk.Button(root, text="Import Image", command=open_image)
import_button.pack()

# Start the GUI main loop
root.mainloop()
