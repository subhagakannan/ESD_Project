import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
path = r'J:\ESD Project\imgs\00241.ppm'
img_raw = cv.imread(path,0)

roi = cv.selectROI(img_raw)

roi_cropped = img_raw[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
blur = cv.GaussianBlur(roi_cropped,(5,5),0)

ret,th = cv.threshold(roi_cropped,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
print(ret)

filtered_image = cv.Canny(blur, threshold1=0.5*ret, threshold2=ret)

plt.subplot(2,2,1),plt.imshow(img_raw,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(roi_cropped,cmap = 'gray')
plt.title('Cropped image'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(filtered_image,cmap = 'gray')
plt.title('Image after canny filtering'), plt.xticks([]), plt.yticks([])

plt.show()
