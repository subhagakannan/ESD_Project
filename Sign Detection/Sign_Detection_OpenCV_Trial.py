import cv2
import numpy as np

def cnts_find(binary_image_red):
    cont_Saver = []

    ( cnts, _) = cv2.findContours(binary_image_red.copy(), cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE )  # finding contours of conected component
    for d in cnts:
        if cv2.contourArea(d) > 700:
            (x, y, w, h) = cv2.boundingRect(d)
            print(x+y+w+h)
            if ((w / h) < 1.21 and (w / h) > 0.59):
                cont_Saver.append([cv2.contourArea(d), x, y, w, h])
    return cont_Saver

#img path

imgMain = cv2.imread('stop8.jpg')

imgHSV = cv2.cvtColor(imgMain,cv2.COLOR_BGR2HSV)

# red mask
lower_red_1 = np.array([0, 70, 50])
upper_red_1 = np.array([10, 255, 255])
mask_1 = cv2.inRange(imgHSV, lower_red_1, upper_red_1)
lower_red_2 = np.array([170, 70, 50])
upper_red_2 = np.array([180, 255, 255])
mask_2 = cv2.inRange(imgHSV, lower_red_2, upper_red_2)
mask = cv2.bitwise_or(mask_1, mask_2)
redMask = cv2.bitwise_and(imgMain, imgMain, mask=mask)
#redMask1=cv2.resize(redMask, (480, 480))
#cv2.imshow("red mask", redMask1)

# separating channels
r_channel1 = redMask[:, :, 2]
g_channel1 = redMask[:, :, 1]
b_channel1 = redMask[:, :, 0]

#contour detection
cont_Saver=cnts_find(r_channel1)
sign_images = []
print ("Total Contours Found: ",len(cont_Saver))
if len(cont_Saver)>0:
    cont_Saver=np.array(cont_Saver)
    cont_Saver=cont_Saver[cont_Saver[:,0].argsort()].astype(int)
    for conta in range(len(cont_Saver)):
        cont_area,x, y, w, h=cont_Saver[len(cont_Saver)-conta-1]
        #getting the boundry of rectangle around the contours.
        image_found=imgMain[y:y+h,x:x+w]
        crop_image0=cv2.resize(image_found, (72, 72))
        sign_images.append(image_found)
        cv2.imshow("result", crop_image0)
        cv2.waitKey(0)
