# converts an image to histogarm

im1 = cv2.imread('r0101.jpg',0)
hist1 = cv2.calcHist([im1],[0],None,[256],[0,256])

im2 = cv2.imread('r0102.jpg',0)
hist2 = cv2.calcHist([im2],[0],None,[256],[0,256])

# compares above 2 histogarms using chi squared metric

hist1 = np.asarray([1,2,3,4] , dtype = np.float32)
hist2 = np.asarray([2,5,3,4] , dtype = np.float32)
a=cv2.compareHist(hist1,hist2,cv2.HISTCMP_CHISQR)
print a

