import train

import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

import pandas as pd

import os



import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torch.cuda.amp import GradScaler
from sklearn.metrics import accuracy_score
import resize
import safe_pil_loader

if(images is not NotImplemented and labels is not NotImplemented):
        #test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        #labels = ['oval','rectangular','round','square']

        #import matplotlib.pyplot as plt
        #print(image_data.shape

        from skimage.color import rgb2gray
        #gray_data = rgb2gray(image_data)

        #seen = set()
        #for i, label in enumerate (labels):
        #   if label in seen:
        #      continue
        # seen.add(label)#

        from sklearn.model_selection import train_test_split
        #print(images.len())
        X_train, X_test, y_train, y_test = train_test_split(images,labels,train_size=0.8,test_size=0.2,random_state = 42)



        import cv2

            

        from sklearn.metrics import confusion_matrix
        images.shape
        reshaped_gray_data = images.reshape((-1,64*64))
        reshaped_gray_data.shape
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        le.fit(labels)
        y = le.transform(labels)
        #X_train, X_test, y_train, y_test = train_test_split(reshaped_gray_data,y,test_size=0.2,random_state = 42)

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



