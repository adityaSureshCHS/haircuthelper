import numpy as np
import pandas as pd


image_data = np.load('images/women/testing_set','images/men/testing_set','images/women/training_set','images/men/training_set')

labels = ['oval','rectangular','round','square']

import matplotlib.pyplot as plt


from skimage.color import rgb2gray

gray_data = rgb2gray(image_data)



seen = set()
for i, label in enumerate (labels):
    if label in seen:
        continue
    seen.add(label)
   

    from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(gray_data,labels,test_size=0.2,random_state = 42)



import cv2

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    elif height is None:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

from sklearn.metrics import confusion_matrix
gray_data.shape
reshaped_gray_data = gray_data.reshape((-1,64*64))
reshaped_gray_data.shape
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(labels)
y = le.transform(labels)
X_train, X_test, y_train, y_test = train_test_split(reshaped_gray_data,y,test_size=0.2,random_state = 42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)



from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression()

logistic_model.fit(X_train, y_train)

predictions = logistic_model.predict(X_test)

from sklearn.metrics import accuracy_score

score = accuracy_score(y_test, predictions)
print('Logistic Regression Model Accuracy: {:.2%}'.format(score))
matrix = confusion_matrix(y_test, predictions)
print(matrix.diagonal()/matrix.sum(axis=1))

print(le.classes_)

print(set(labels))

from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier


neigh = KNeighborsClassifier(n_neighbors=1)

neigh.fit(X_train,y_train)

prediction = neigh.predict(X_test)


score = accuracy_score(y_test, prediction)
print('K Nearest Neighbor Model Accuracy: {:.2%}'.format(score))

matrix = confusion_matrix(y_test, predictions)
print(matrix.diagonal()/matrix.sum(axis=1))

print(le.classes_)

