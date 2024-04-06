import cv2
import numpy as np
image = cv2.imread('images/cut5.jpeg', 0)
cv2.imshow('test', image);
cv2.waitKey(0)
cv2.destroyAllWindows()
