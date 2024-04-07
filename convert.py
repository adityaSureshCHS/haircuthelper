import cv2
import matplotlib.pyplot as plt
import numpy as np
global getX 
global getY
global width
global height
imagePath = 'images/men/testing_set/rectangular/1.jpg'
image = cv2.imread(imagePath)
print(image.shape)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
face = face_classifier.detectMultiScale(
    hsv, scaleFactor=1.1, minNeighbors=5, minSize=(40,40)
)

for (x, y, w, h) in face:
    getX = x
    getY = y
    width = w
    height = h
    cv2.rectangle(image, (x - w, y - h - h), (x + w + w, y + h + h), (0, 255, 0), 4)

img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(20, 10))
plt.imshow(img_rgb)
plt.axis('off')
plt.show()