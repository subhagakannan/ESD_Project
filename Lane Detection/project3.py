import cv2
import matplotlib.pylab as plt
import numpy as np

def region_of_interest(img,vertices):
    mask =np.zeros_like(image)
    #channel_count = img.shape[2]
    #match_mask_color = 150
    #cv2.fillPoly(mask,vertices,match_mask_color)
    masked_image = cv2.bitwise_and (image,mask)
    return masked_image

def drow_the_lines(image,lines):
    img =np.copy(image)
    blank_image = np.zeros((image.shape[0],image.shape[1],3),dtype=np.uint8)
    line_color = [0,255,0]
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(blank_image,(x1,y1),(x2,y2),(0,200,0),thickness=3)
    img =cv2.addWeighted(img,0.8,blank_image,1,0.0)
    return img

image = cv2.imread('road1.jpg')
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

print(image.shape)
height = image.shape[0]
width = image.shape[1]
region_of_interest_vertices = [(0,height),(630,480),(1500,height)]

gray_image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
canny_image = cv2.Canny(gray_image,800,400)
cropped_image = region_of_interest(image,np.array([region_of_interest_vertices], np.int32))
lines =cv2.HoughLinesP (canny_image,
                        rho = 1,
                        theta = np.pi/180,
                        threshold = 40,
                        lines = np.array([]),
                        minLineLength = 60,
                        maxLineGap = 10)

image_lines = drow_the_lines(cropped_image,lines)
#cv2.imshow("result",canny_image)
#cv2.imshow("result",cropped_image)
plt.imshow(image_lines)
plt.show()
#except AssertionError as msg:
#print(msg)