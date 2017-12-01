# using shape context

from matplotlib import pyplot as plt
import glob
import numpy as np
import cv2
names = []
names = glob.glob("/home/shruti/Sift_data/*.jpg")

l = len(names)


a = cv2.imread(names[0]);
b = cv2.imread(names[1]);

imgray_a = cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
ret_a,thresh_a = cv2.threshold(imgray_a,127,255,0)

imgray_b = cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)
ret_b,thresh_b = cv2.threshold(imgray_b,127,255,0)


# find contours
_, ca, _ = cv2.findContours(thresh_a, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
_, cb, _ = cv2.findContours(thresh_b, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
print(np.shape(ca[0]), np.shape(cb[0]))

# ShapeContext

sd = cv2.createShapeContextDistanceExtractor()

d = sd.computeDistance(ca[0],cb[0])
print (d)
