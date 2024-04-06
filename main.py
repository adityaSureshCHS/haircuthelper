import cv2
import numpy as np
from flask import Flask, url_for, request, render_tempalte, redirect

app = Flask(__name__)

@app.route("/")
def mainpage():
    return render_template("mainpage.html")

image = cv2.imread('images/cut5.jpeg', 0)
cv2.imshow(image)
