#import cv2

#cap=cv2.VideoCapture("Test Video.mp4")

#while True:
   # success,img =cap.read()
    #cv2.imshow("video",img)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
        # break
import cv2 as cv

# The video feed is read in as a VideoCapture object
cap = cv.VideoCapture("testvideo3.mp4")
while (cap.isOpened()):

    ret, frame = cap.read()
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    canny=cv.Canny(gray,450,280)
    if ret == True:
         cv.imshow("video",frame)
         cv.imshow("video",gray)
         cv.imshow("video", canny)

         if cv.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv.destroyAllWindows()