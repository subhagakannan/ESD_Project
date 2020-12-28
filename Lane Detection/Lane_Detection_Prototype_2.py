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
from operator import itemgetter
import shapely
from shapely.geometry import LineString, Point

path = r'E:\ESD_Project\Takeover.m4v'

#Method to draw the lines obtained from Hough transform
def draw_the_lines(image,lines,roi):
    img =np.copy(image)
    blank_image = np.zeros((image.shape[0],image.shape[1],3),dtype=np.uint8)
    line_color = [0,255,0]
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv.line(blank_image,((x1+int(roi[0])),(y1+int(roi[1]))),((x2+int(roi[0])),(y2+int(roi[1]))),(0,255,0),thickness=10)
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
    cv.destroyAllWindows()
    return roi_CalOut
    
def elements(array):
    return array.ndim and array.size
    
def FindLeftRight(DistanceArray):
    DistPos = []
    DistNeg = []
    for Dist in DistanceArray: 
        if(Dist>0):
            DistPos.append(Dist)
        else:
            DistNeg.append(Dist)
    if DistNeg :
        RightLaneIdx = DistanceArray.index(max(DistNeg))
    else:
        RightLaneIdx =None
    if DistPos :
        LeftLaneIdx = DistanceArray.index(min(DistPos))
    else:
        LeftLaneIdx = None
    return LeftLaneIdx,RightLaneIdx
            
                
#Caputuring the video
cap = cv.VideoCapture(path)

#Read the calibration data for ROI, length and angles
roi_Cal = [x.strip() for x in open("ROI_Cal.txt","r")]
len_Cal = [x.strip() for x in open("Len_Cal.txt","r")]
ang_Cal = [x.strip() for x in open("Ang_Cal.txt","r")]
roi = roi_Cal
PrevLeftLane =  np.zeros((1,4),dtype=int)
PrevRightLane = np.zeros((1,4),dtype=int);
LErrorFrames =0
RErrorFrames =0
firstFrame =1
while (cap.isOpened()):
    #print("Press 's' to enter the Calibration mode")
    ret, Orgframe = cap.read()
    frame = cv.resize(Orgframe,(800,600))
    if cv.waitKey(10) & 0xFF == ord('s'):
        roi = Calibration(frame,roi)
    
    #Selecting the area where the lanes are likely to be present
    LongLines = [];
    distFrmCenter = [];
    selectline =[];

    
    #Obtaining the cropped frames
    roi_cropped = frame[int(roi[1]):(int(roi[1])+int(roi[3])), int(roi[0]):(int(roi[0])+int(roi[2]))]
    numrows = len(roi_cropped)
    #converting the frames to gray scale
    gray=cv.cvtColor(roi_cropped,cv.COLOR_BGR2GRAY)
    
    
    
    #Applying canny filter for edge detection with fixed thresholds
    canny = cv.Canny(gray, threshold1=70, threshold2=170) 
    
    #Applying Hough Transform to obtain the straight lines
    lines =cv.HoughLinesP (canny,
                        rho = 1,
                        theta = np.pi/180,
                        threshold = 40,
                        lines = np.array([]),
                        minLineLength = 5,
                        maxLineGap = 10)
    #print(lines)
    #print('sorted lines')
    #line_list = lines.tolist()

    #line_list.sort(key=itemgetter(0))
    #print(line_list)
    lines = np.array(lines)
    xaxis = LineString([(0,0),(int(roi[2]),0)])
    centerpoint = abs((int(roi[0]) - int(roi[2])))/2;
    #print(centerpoint);
    #Filtering out the lane lines using length and angle
    #line2 = LineString([(int(roi[1]),line[0][1]), (line[0][2],line[0][3])])
    #                    int(roi[1])
    prevx1 =0;
    if elements(lines) != 0:
        line_list = lines.tolist()
        line_list.sort(key=itemgetter(0))
        lines = np.array(line_list)
        for line in lines:
            for x1,y1,x2,y2 in line:
                length = math.sqrt(((line[0,2]-line[0,0])**2)+((line[0,3]-line[0,1])**2));
                angle = math.degrees(math.atan((x1-x2)/(y1-y2)));
                #deviation = abs(x1-prevx1);
                deviation = 1;
                #print(deviation)
                if(deviation>0):
                    if((length>50)and(angle>(int(ang_Cal[0])) and angle<(int(ang_Cal[1]))) or ((angle <(int(ang_Cal[2]))) and (angle>(int(ang_Cal[3]))))):
                        #print("angle : "+str(math.degrees(math.atan((x1-x2)/(y1-y2)))))
                        #print("length :"+str(length));
                        line1 = LineString([(line[0,0],line[0,1]), (line[0,2],line[0,3])])
                        #if(line1.intersects(xaxis)):
                        z = np.polyfit((y1,y2),(x1,x2), 1)
                        f = np.poly1d(z)
                        #print(round(f(0)))
                        #int_pt = line1.intersection(xaxis)
                        #print(int_pt.x)
                        #print(int_pt.y)
                        if(y1 > y2):
                            line[0,0] = x2
                            line[0,1] = y2
                            line[0,2] = x1
                            line[0,3] = y1
                        #print(line)
                        if(f(0)>=0):
                            line[0,0] = round(f(0))
                            line[0,1] = 0
                        if(f(numrows)>=0):
                            line[0,2] = round(f(numrows))
                            line[0,3] = numrows
                        #print(line)
                        #print(length)
                        if((line[0,3] == numrows) and (line[0,1] == 0)):
                           # print(line)
                            LongLines.append(line);
                            distFrmCenter.append((centerpoint - line[0][2]));
                            #print(distFrmCenter)
                        #print(deviation);
                        #print(x1)
                        #print(x2) 
                        prevx1 = x1;    
    else:
        print("no lines");
    #print(LongLines);
    if (len(LongLines) != 0):
        LeftLaneIdx,RightLaneIdx = FindLeftRight(distFrmCenter);
        if((LeftLaneIdx != None)):
            if((((PrevLeftLane[0][0] - LongLines[LeftLaneIdx][0][2]) < 0) or (firstFrame == 1)) and ((centerpoint-LongLines[LeftLaneIdx][0][2])>0)):
                selectline.append(LongLines[LeftLaneIdx]);
                PrevLeftLane = LongLines[LeftLaneIdx]
            elif((centerpoint-PrevLeftLane[0][2])>0):
                #print('élse')
                selectline.append(PrevLeftLane);
                LErrorFrames+=1
            if((LErrorFrames>=20) or (firstFrame==1)):
                #print('entered')
                PrevLeftLane = LongLines[LeftLaneIdx]
                LErrorFrames = 0
        elif(LErrorFrames<20):
            selectline.append(PrevLeftLane);
            LErrorFrames+=1
        if(RightLaneIdx != None):
            if((((PrevRightLane[0][0] - LongLines[RightLaneIdx][0][0]) > 0) or (firstFrame == 1)) and ((centerpoint-LongLines[RightLaneIdx][0][0])<0)):
                selectline.append(LongLines[RightLaneIdx]);
                PrevRightLane = LongLines[RightLaneIdx]
            elif((centerpoint-LongLines[RightLaneIdx][0][0])<0):
                #print('élse')
                selectline.append(PrevRightLane);
                RErrorFrames+=1
            if((RErrorFrames>=20) or (firstFrame==1)):
                #print('entered')
                PrevRightLane = LongLines[RightLaneIdx]
                RErrorFrames = 0
        elif(RErrorFrames<20):
            selectline.append(PrevRightLane);
            RErrorFrames+=1
        print(selectline)
    #    
    #if(len(selectline) !=0 ):
    #    if(len(selectline[0][0]) >3):
    #        image_lines = draw_the_lines(frame,LongLines,roi)
    #    else:
    #        image_lines = frame
    if(len(selectline) !=0 ):
        image_lines = draw_the_lines(frame,selectline,roi)
        LeftLaneIdx,RightLaneIdx = FindLeftRight(distFrmCenter);
    else:
        image_lines = frame
    if ret == True:
        firstFrame =0
        cv.namedWindow('frame2',cv.WND_PROP_FULLSCREEN)
        cv.setWindowProperty('frame2', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        cv.imshow("frame2", image_lines)
        #print(distFrmCenter);
        print('-------------------------------------------------------------------');
        if cv.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv.destroyAllWindows()
