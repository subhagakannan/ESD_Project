#Software for Lane Detection Prototype I 
#Author: Saranyaa Suresh, Juliet Eldo, Subhaga Kannan, Dhanush Lingeswaran
#Description: The Prototype I SW detects the straight lanes in which the Autonomous vehicle is driving. 
#Canny filter works with fixed thresholds. This has to be checked if it works with real-time data. Fine tuning has to be done incase of failure 
#The Region of lanes being present is done by the user. These values should be taekn from calibration data in the next release.
#The length and the angle of lanes must be fine tuned after real time testing to support the camera features.

import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt
path = r'J:\\ESD Project\\test_video\\test_video_1.MOV'
i=1

#Method to draw the lines obtained from Hough transform
def draw_the_lines(image,lines,roi):
    img =np.copy(image)
    blank_image = np.zeros((image.shape[0],image.shape[1],3),dtype=np.uint8)
    line_color = [0,255,0]
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv.line(blank_image,((x1+int(roi[0])),(y1+int(roi[1]))),((x2+int(roi[0])),(y2+int(roi[1]))),(0,200,0),thickness=10)
    img =cv.addWeighted(img,0.8,blank_image,1,0.0)
    return img

#Caputuring the video

cap = cv.VideoCapture(path)
while (cap.isOpened()):
    ret, frame = cap.read()
	
	#Selecting the area where the lanes are likely to be present
    if i==1:
        roi=cv.selectROI(frame)
        print(roi)
        i=2
		
    LongLines = [];
	
	#Obtaining the cropped frames
    roi_cropped = frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
	
	#converting the frames to gray scale
    gray=cv.cvtColor(roi_cropped,cv.COLOR_BGR2GRAY) 
	
	#Applying canny filter for edge detection with fixed thresholds
    canny = cv.Canny(gray, threshold1=70, threshold2=175) 
    
    #Applying Hough Transform to obtain the straight lines
    lines =cv.HoughLinesP (canny,
                        rho = 1,
                        theta = np.pi/180,
                        threshold = 40,
                        lines = np.array([]),
                        minLineLength = 60,
                        maxLineGap = 5)
	
	#Filtering out the lane lines using length and angle
    for line in lines:
        for x1,y1,x2,y2 in line:
            length = math.sqrt(((x2-x1)**2)+((y2-y1)**2));
            angle = math.degrees(math.atan((x1-x2)/(y1-y2)));
            if(length >= 150 and ((angle>(-80) and angle<(-30)) or ((angle <80) and (angle>30)))):
                print("angle : "+str(math.degrees(math.atan((x1-x2)/(y1-y2)))))
                print("length :"+str(length));
                LongLines.append(line);
    print(LongLines);
   
    image_lines = draw_the_lines(frame,LongLines,roi)
    if ret == True:
        cv.imshow("video", image_lines)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv.destroyAllWindows()
