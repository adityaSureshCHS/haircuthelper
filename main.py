import cv2
import numpy as np
image = cv2.imread('cut5.jpeg')
cv2.imwrite('cut5.jpeg', image)
print("Image saved successfully.")