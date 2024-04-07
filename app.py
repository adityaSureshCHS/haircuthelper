import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from flask import Flask, url_for, request, render_template, redirect
app = Flask(__name__)

training_list = []
directoryoval1 = 'images/men/training_set/oval'
directoryoval2 = 'images/women/training_set/oval'
directoryrectangular1 = 'images/men/training_set/rectangular'
directoryrectangular2 = 'images/women/training_set/rectangular'
directoryround1 = 'images/men/training_set/round'
directoryround2 = 'images/women/training_set/round'
directorysquare1 = 'images/men/training_set/square'
directorysquare2 = 'images/women/training_set/square'
size1 = 0
size2 = 0
size3 = 0
size4 = 0

for files in os.listdir(directoryoval1):
    f = os.path.relpath(files)
    image = cv2.imread(f, 0)
    training_list.append(image)
    size1+=1
for files in os.listdir(directoryoval2):
    f = os.path.join(directoryoval2, files)
    image = cv2.imread(f, 0)
    image = image.flatten()
    training_list.append(image)
    size1+=1

for files in os.listdir(directoryrectangular1):
    f = os.path.join(directoryrectangular1, files)
    image = cv2.imread(f, 0)
    image = image.flatten()
    training_list.append(image)
    size2+=1
for files in os.listdir(directoryrectangular2):
    f = os.path.join(directoryrectangular2, files)
    image = cv2.imread(f, 0)
    image = image.flatten()
    training_list.append(image)
    size2+=1

for files in os.listdir(directoryround1):
    f = os.path.join(directoryround1, files)
    image = cv2.imread(f, 0)
    image = image.flatten()
    training_list.append(image)
    size3+=1
for files in os.listdir(directoryround2):
    f = os.path.join(directoryround2, files)
    image = cv2.imread(f, 0)
    image = image.flatten()
    training_list.append(image)
    size3+=1

for files in os.listdir(directorysquare1):
    f = os.path.join(directorysquare1, files)
    image = cv2.imread(f, 0)
    image = image.flatten()
    training_list.append(image)
    size4+=1
for files in os.listdir(directorysquare2):
    f = os.path.join(directorysquare2, files)
    image = cv2.imread(f, 0)
    image = image.flatten()
    training_list.append(image)
    size4+=1

train_labels = []
for i in range(size1):
    train_labels.append("oval")
for i in range(size2):
    train_labels.append("rectangular")
for i in range(size3):
    train_labels.append("round")
for i in range(size4):
    train_labels.append("square")

knn = cv2.ml.KNearest_create()
knn.train(training_list, cv2.ml.ROW_SAMPLE, train_labels)

'''testing_list = []
directoryoval3 = 'images/men/testing_set/oval'
directoryoval4 = 'images/women/testing_set/oval'
directoryrectangular3 = 'images/men/testing_set/rectangular'
directoryrectangular4 = 'images/women/testing_set/rectangular'
directoryround3 = 'images/men/testing_set/round'
directoryround4 = 'images/women/testing_set/round'
directorysquare3 = 'images/men/testing_set/square'
directorysquare4 = 'images/women/testing_set/square'
size1 = 0
size2 = 0
size3 = 0
size4 = 0


for files in os.listdir(directoryoval3):
    f = os.path.join(directoryoval3, files)
    testing_list.append(f)
    size1+=1
for files in os.listdir(directoryoval4):
    f = os.path.join(directoryoval4, files)
    testing_list.append(f)
    size1+=1

for files in os.listdir(directoryrectangular3):
    f = os.path.join(directoryrectangular3, files)
    testing_list.append(f)
    size2+=1
for files in os.listdir(directoryrectangular4):
    f = os.path.join(directoryrectangular4, files)
    testing_list.append(f)
    size2+=1

for files in os.listdir(directoryround3):
    f = os.path.join(directoryround3, files)
    testing_list.append(f)
    size3+=1
for files in os.listdir(directoryround4):
    f = os.path.join(directoryround4, files)
    testing_list.append(f)
    size3+=1

for files in os.listdir(directorysquare3):
    f = os.path.join(directorysquare3, files)
    testing_list.append(f)
    size4+=1
for files in os.listdir(directorysquare4):
    f = os.path.join(directorysquare4, files)
    testing_list.append(f)
    size4+=1

test_labels = []
for i in size1:
    test_labels.append("oval")
for i in size2:
    test_labels.append("rectangular")
for i in size3:
    test_labels.append("round")
for i in size4:
    test_labels.append("square")

result = knn.findNearest(testing_list, k=5)
matches = result == test_labels
correct = np.count_nonzero(matches)
accuracy = correct * 100.0 / result.size
print("Accuracy: {:.2f}%".format(accuracy))
'''

@app.route("/")
def mainpage():
    return render_template("mainpage.html", upload = url_for("upload"))
@app.route("/upload")
def upload():
    return render_template("uploadphoto.html", results = url_for("results"))

@app.route("/results", methods=["POST", "GET"])
def results():
    if request.method == 'POST':
        return render_template("results.html")
    else:
        return render_template("results.html")

