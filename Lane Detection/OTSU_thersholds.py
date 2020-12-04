import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
path = r'J:\ESD Project\imgs\00549.ppm'
img = cv.imread(path,0)

ret1,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
print(ret1)

ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
print(ret2)

blurred_img = cv.GaussianBlur(img,(5,5),0)
ret3,th3 = cv.threshold(blurred_img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
print(ret3)

filtered_image1 = cv.Canny(img, threshold1=0.3*ret1, threshold2=ret1)
filtered_image2 = cv.Canny(img, threshold1=0.3*ret2, threshold2=ret2)
filtered_image3 = cv.Canny(img, threshold1=0.3*ret3, threshold2=ret3)

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(filtered_image1,cmap = 'gray')
plt.title('Global Thresholding'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(filtered_image2,cmap = 'gray')
plt.title('Otsu Thresholding'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(filtered_image3,cmap = 'gray')
plt.title('Otsu Thresholding with gaussian blurring'), plt.xticks([]), plt.yticks([])


plt.show()