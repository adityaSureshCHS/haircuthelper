import cv2
import numpy as np
from flask import Flask, url_for, request, render_template, redirect

app = Flask(__name__)

@app.route("/")
def mainpage():
    return render_template("mainpage.html")

def display_image():
    image = cv2.imread('images/cut5.jpeg', 0)
    cv2.imshow("text", image)
    cv2.waitKey(5)
    cv2.destroyAllWindows()
    
display_image()