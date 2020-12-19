#Software for Lane Detection Prototype I 
#Author: Saranyaa Suresh, Juliet Eldo, Subhaga Kannan, Dhanush Lingeswaran
#Description: The Prototype I SW detects the straight lanes in which the Autonomous vehicle is driving. 
#Canny filter works with fixed thresholds. This has to be checked if it works with real-time data. Fine tuning has to be done incase of failure.
#The Region of lanes being present, length and angles of the lanes are taken from calibration data.
#The length and the angle of lanes must be fine tuned after real time testing to support the camera features.

import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt
path = r'E:\\ESD_Project\\Takeover.M4v'

#Method to draw the lines obtained from Hough transform
def draw_the_lines(image,lines,roi):
    img =np.copy(image)
    blank_image = np.zeros((image.shape[0],image.shape[1],3),dtype=np.uint8)
    line_color = [0,255,0]
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv.line(blank_image,((x1+int(roi[0])),(y1+int(roi[1]))),((x2+int(roi[0])),(y2+int(roi[1]))),(0,200,0),thickness=5)
    img =cv.addWeighted(img,0.8,blank_image,1,0.0)
    return img
#Method to Calibrate ROI
def Calibration(frame_cal,roi_CalFunc):
    print(type(roi_CalFunc[0]));
    start_point = (int(roi_CalFunc[0]),int(roi_CalFunc[1]))
    end_point = (int(roi_CalFunc[2]),int(roi_CalFunc[3]))
    print(start_point)
    print(end_point)
    color = (0, 255, 0) 
    # Line thickness of 2 px 
    thickness = 2
    cv.rectangle(img=frame_cal, pt1=(int(roi_CalFunc[0]), int(roi_CalFunc[1])), pt2=((int(roi_CalFunc[0])+int(roi_CalFunc[2])),(int(roi_CalFunc[1])+int(roi_CalFunc[3]))), color=(0, 0, 255), thickness=2)
    roi_CalOut=cv.selectROI(frame_cal)
    f= open("ROI_Cal.txt","w")
    f.write(str(roi_CalOut[0])+"\n")
    f.write(str(roi_CalOut[1])+"\n")
    f.write(str(roi_CalOut[2])+"\n")
    f.write(str(roi_CalOut[3])+"\n")
    return roi_CalOut
def elements(array):
    return array.ndim and array.size
#Caputuring the video
cap = cv.VideoCapture(path)

#Read the calibration data for ROI, length and angles
roi_Cal = [x.strip() for x in open("ROI_Cal.txt","r")]
len_Cal = [x.strip() for x in open("Len_Cal.txt","r")]
ang_Cal = [x.strip() for x in open("Ang_Cal.txt","r")]
roi = roi_Cal
while (cap.isOpened()):
    print("Press 's' to enter the Calibration mode")
    ret, Orgframe = cap.read()
    frame = cv.resize(Orgframe,(480,320))
    if cv.waitKey(10) & 0xFF == ord('s'):
        roi = Calibration(frame,roi)
    
    #Selecting the area where the lanes are likely to be present
    LongLines = [];
    
    #Obtaining the cropped frames
    roi_cropped = frame[int(roi[1]):(int(roi[1])+int(roi[3])), int(roi[0]):(int(roi[0])+int(roi[2]))]
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
    lines = np.array(lines)
    
    #Filtering out the lane lines using length and angle
    if elements(lines) != 0:
        for line in lines:
            for x1,y1,x2,y2 in line:
                length = math.sqrt(((x2-x1)**2)+((y2-y1)**2));
                angle = math.degrees(math.atan((x1-x2)/(y1-y2)));
                if((length >= 30)and ((angle>(int(ang_Cal[0])) and angle<(int(ang_Cal[1]))) or ((angle <(int(ang_Cal[2]))) and (angle>(int(ang_Cal[3])))))):
                    #print("angle : "+str(math.degrees(math.atan((x1-x2)/(y1-y2)))))
                    #print("length :"+str(length));
                    LongLines.append(line);
    else:
        print("no lines");
    #print(LongLines);
    if (len(LongLines) != 0):
        image_lines = draw_the_lines(frame,LongLines,roi)
    else:
        image_lines = frame
    if ret == True:
        cv.imshow("video", image_lines)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv.destroyAllWindows()
