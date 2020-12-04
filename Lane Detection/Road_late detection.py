import cv2
import numpy as np
import matplotlib.pylab as plt

image = cv2.imread('Test_image1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print(image.shape)
height = image.shape[0]
width = image.shape[1]

region_of_interest_vertices = [(0,height),(620,480),(1100,height)]

def region_of_interest(img,vertices):
    mask = np.zeros_like(image)
    channel_count = img.shape[2]
    match_mask_color = (255,)* channel_count
    cv2.fillPoly(mask,vertices,match_mask_color)
    masked_image = cv2.bitwise_and (image, mask)
    return masked_image

cropped_image = region_of_interest(image, np.array([region_of_interest_vertices], np.int32))
gray = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)
canny = cv2.Canny(gray, 60, 250)

cv2.imshow("result",canny)
plt.imshow(image)
plt.show()