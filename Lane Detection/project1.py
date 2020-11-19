import cv2

img=cv2.imread("road.jpeg")
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgCanny = cv2.Canny(img,600,430)

cv2.imshow("Image",img)
cv2.imshow("Gray Image",imgGray)
cv2.imshow("Canny Image",imgCanny)

cv2.waitKey()
