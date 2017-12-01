# Using SIFT descriptors to compare images from given directory. Every two consecutive images are compared with each other.

import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
names = []

## Replace with ur directory
names = glob.glob("/home/shruti/Sift_data/*.jpg")
l = len(names)

i=0
while(i<l):
	
	img1 = cv2.imread(names[i],0)          # queryImage
	img2 = cv2.imread(names[i+1],0) # trainImage
	i = i+2
  
	# Initiate SIFT detector
	orb = cv2.ORB_create()

	# find the keypoints and descriptors with SIFT
	kp1, des1 = orb.detectAndCompute(img1,None)
	kp2, des2 = orb.detectAndCompute(img2,None)

	# create BFMatcher object
  # u can replace cv2.NORM_HAMMING with any other distance as per your choice
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	# Match descriptors.
	matches = bf.match(des1,des2)

	# Sort them in the order of their distance.
	matches = sorted(matches, key = lambda x:x.distance)
	c = 2
	while(c<=16):
		img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:c],None,flags=2)
		c = c+2
		plt.imshow(img3),plt.show()


##  second code

'''
import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
names = []
names = glob.glob("/home/shruti/RD-cvpr06/*.jpg")
l = len(names)

i=0
while(i<l):
	
	img1 = cv2.imread(names[i],0)          # queryImage
	img2 = cv2.imread(names[i+1],0) # trainImage
	i = i+2
	# Initiate SIFT detector
	sift = cv2.xfeatures2d.SIFT_create()

	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)

	# BFMatcher with default params
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1,des2, k=2)

	# Apply ratio test
	good = []
	for m,n in matches:
	    if m.distance < 0.75*n.distance:
		good.append([m])

	c = 2
	while(c<=16):
		comp = good[0:c]
		# cv2.drawMatchesKnn expects list of lists as matches.
		img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,comp,None,flags=2)
		c = c+2
		plt.imshow(img3),plt.show()
		
'''
