import cv2
import numpy as np
import os
from keras.models import load_model
import time
import pandas as pd

################################################
frameWidth= 600         # CAMERA RESOLUTION
frameHeight = 600
brightness = 180
threshold = 0.80       # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
model = load_model("E:\\ESD_Project\\MyLeNetModel")
#imgMain = cv2.imread('D:\\Downloads\\TrainIJCNN2013\\00595.ppm')
path = r'Takeover.m4v'
labelFile = "D:\\Downloads\\labels.csv"
ClassData=pd.read_csv(labelFile)
ClassList = ClassData['Name'].to_list()
cap = cv2.VideoCapture(path)
################################################
def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img

def preprocessing(img):
    img = grayscale(img)
    #img = cv2.equalizeHist(img)
    img = img/255
    return img
        
def Calssify(CrpdImage):
    # PROCESS IMAGE
    img = np.asarray(CrpdImage)
    #img = cv2.resize(img, (60, 60))
    img = preprocessing(img)
    #cv2.imshow("Processed Image", img)
    img = img.reshape(1, 60, 60, 1)
    #cv2.putText(CrpdImage, "CLASS: " , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    #cv2.putText(CrpdImage, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    # PREDICT IMAGE
    predictions = model.predict(img)
    classIndex = model.predict_classes(img)
    probabilityValue =np.amax(predictions)
    if probabilityValue > threshold:
        className =  ClassList[int(classIndex)]
        print(className)
        print("probability :"+ str(probabilityValue))
        #cv2.putText(CrpdImage,str(classIndex)+" "+str(getCalssName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        #cv2.putText(CrpdImage, str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        #cv2.imshow("Result", CrpdImage)
    else:
        className = "None"
        #print(className)
    
    #cv2.waitKey(1)
    return className


def cnts_find(binary_image_red):
    cont_Saver = []
    ( cnts, _) = cv2.findContours(binary_image_red.copy(), cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)  # finding contours of conected component
    for d in cnts:
        if (cv2.contourArea(d) > 150 and cv2.contourArea(d) < 200):
            (x, y, w, h) = cv2.boundingRect(d)
            #print(w / h)
            if ((w / h) < 1.00 and (w / h) > 0.7 and (w>10 and w<50)):
                cont_Saver.append([cv2.contourArea(d), x, y, w, h])
    return cont_Saver

def Draw_bboxes(bboxes, image):
    
    image = image
    box = bboxes
    figure = cv2.rectangle(image, (box[0],box[1]), (box[2],box[3]), (0,255,0), 1)
    return figure
    
def HSV_Proc(imgMain):
    imgHSV = cv2.cvtColor(imgMain,cv2.COLOR_BGR2HSV)
    # red mask
    lower_red_1 = np.array([0, 50, 30])
    upper_red_1 = np.array([10, 255, 255])
    mask_1 = cv2.inRange(imgHSV, lower_red_1, upper_red_1)
    lower_red_2 = np.array([170, 50, 30])
    upper_red_2 = np.array([180, 255, 255])
    mask_2 = cv2.inRange(imgHSV, lower_red_2, upper_red_2)
    mask = cv2.bitwise_or(mask_1, mask_2)
    redMask = cv2.bitwise_and(imgMain, imgMain, mask=mask)
    
    #bluemask
    lowerBlue = np.array([100, 70, 50])  #hsv low values
    upperBlue = np.array([135, 255, 255])  #hsv high values
    maskb = cv2.inRange(imgHSV, lowerBlue, upperBlue)
    blueMask = cv2.bitwise_and(imgMain, imgMain, mask=maskb)
    
    # yellow mask
    lower_yellow_1 = np.array([5, 50, 30])
    upper_yellow_1 = np.array([32, 255, 255])
    masky = cv2.inRange(imgHSV, lower_yellow_1, upper_yellow_1)
    yellowMask = cv2.bitwise_and(imgMain, imgMain, mask=masky)
    
    # separating channels
    r_channel1 = redMask[:, :, 2]
    b_channel1 = blueMask[:, :, 0]
    y_channel1 = yellowMask[:, :, 2]
    
    tot_channel = (r_channel1+ b_channel1+ y_channel1)
    #contour detection
    cont_Saver=cnts_find(tot_channel)
    bboxes =[]
    #print ("Total Contours Found: ",len(cont_Saver))
    if len(cont_Saver)>0:
        cont_Saver=np.array(cont_Saver)
        cont_Saver=cont_Saver[cont_Saver[:,0].argsort()].astype(int)
        counter =0
        for conta in range(len(cont_Saver)):
            cont_area,x, y, w, h=cont_Saver[len(cont_Saver)-conta-1]
            #getting the boundry of rectangle around the contours.
            image_found=imgMain[abs(y-10):y+h+10,abs(x-10):x+w+10]
            crop_image0=cv2.resize(image_found, (60, 60))
            #sign_images.append(image_found)
            #DetectedClass.append(Calssify(crop_image0))
            ClassName = Calssify(crop_image0)
            #ClassName = "q"
            if(ClassName != "None"):
                imgMain = Draw_bboxes((abs(x-10),abs(y-10),x+w+10,y+h+10),imgMain);
                cv2.putText(imgMain,str(ClassName), (120, 35+(counter*50)), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                counter = counter+1
    return imgMain
    
##img path
#while (cap.isOpened()):
#    start_time = time.time() # start time of the loop
#    ret, frame = cap.read()
#    frame = cv2.resize(frame,(800,600))
#    frame = HSV_Proc(frame)
#    if ret == True:
#        cv2.namedWindow('frame2',cv2.WND_PROP_FULLSCREEN)
#        cv2.setWindowProperty('frame2', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#        cv2.imshow("frame2", frame)
#        if cv2.waitKey(10) & 0xFF == ord('q'):
#            break
#    print("FPS: ", round(1.0 / (time.time() - start_time))) # FPS = 1 / time to process loop
#