import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
import joblib
from sklearn.svm import SVC
from sklearn import metrics


# Load the dataset
data = pd.read_csv('dataset.csv')
data = shuffle(data)

X = data.drop(["label"], axis=1)
Y = data["label"]

# Choose an index for the image you want to display
idx = 120

# Reshape the image to 280x280
img = X.loc[idx].values.reshape(280, 280)

# Display the image
"""
plt.imshow(img, cmap='gray')  # Use 'gray' colormap for grayscale images
plt.title(f"Label: {Y[idx]}")  # Display the corresponding label as the title
plt.show()
"""
acc_score = float(0)
run_count = 0

# Split the dataset into training and testing sets
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)

# Create the model
classifier = SVC(kernel="linear", C=2.0, random_state=6)

# run the model until the accuracy is greater than 90%
while acc_score < 0.9 and run_count < 10:

    classifier.fit(train_x,train_y)

    prediction=classifier.predict(test_x)
    acc_score = metrics.accuracy_score(prediction,test_y)
    print("Accuracy= ", acc_score)
    run_count += 1

# Save the model
if acc_score > 0.9:
    joblib.dump(classifier, "model/digit_recognizer")
    print("Model dumped!")
else:
    print("Model not dumped!")