# Generate dataset
import cv2
import csv
import glob

header = ["label"]
for i in range(0, 280 * 280):
    header.append("pixel" + str(i))
with open('dataset.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

for label in range(10):
    dirList = glob.glob("captured_images/" + str(label) + "/*.png")

    for img_path in dirList:
        im = cv2.imread(img_path)
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_gray = cv2.GaussianBlur(im_gray, (15, 15), 0)
        roi = cv2.resize(im_gray, (280, 280), interpolation=cv2.INTER_AREA)

        data = []
        data.append(label)

        # Flatten the 280x280 image into a 1D array
        # converts pixel values to 0 or 1 based on a threshold of 100
        for pixel_value in roi.ravel():
            if pixel_value > 100:
                pixel_value = 1
            else:
                pixel_value = 0
            data.append(pixel_value)

        with open('dataset.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data)
