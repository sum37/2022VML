import cv2
import time
import HandTrackingModule as htm
import os
import numpy as np

brushThickness=15


folderPath="./hand_detection/header"
myList=os.listdir(folderPath)
overlayList=[]
for imPath in myList:
    image=cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
header=overlayList[0]
drawColor=(0,0,0)

wCam, hCam = 1280, 728

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector=htm.handDetector(detectionCon=0.75)
xp, yp=0,0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    #1. Import image
    success, img = cap.read()
    img=cv2.flip(img,1)
    
    #2. Find Hand Landmarks
    img = detector.findHands(img)
    lmList=detector.findPosition(img, draw=False)
    
    if len(lmList)!=0:
        # tip of index and middle fingers
        x1, y1=lmList[8][1:]
        x2, y2=lmList[12][1:]
        
    
        #3. Check which fingers are up
        fingers=detector.fingersUp()
        #4. If Selection mode - Two fingeres are up
        if fingers[1] and fingers[2]:
            xp, yp =0,0
            if y1<108:
                #checkin for the click
                if 0<x1<116:
                    header=overlayList[1]
                    drawColor=(0,0,198)
                elif 116<x1<233:
                    header=overlayList[2]
                    drawColor=(0,0,255)
                elif 233<x1<349:
                    header=overlayList[3]
                    drawColor=(0,192,255)
                elif 349<x1<465:
                    header=overlayList[4]
                    drawColor=(0,255,255)
                elif 465<x1<582:
                    header=overlayList[5]
                    drawColor=(80,208,146)
                elif 582<x1<698:
                    header=overlayList[6]
                    drawColor=(80,176,0)
                elif 698<x1<815:
                    header=overlayList[7]
                    drawColor=(240,176,0)
                elif 815<x1<931:
                    header=overlayList[8]
                    drawColor=(192,112,0)
                elif 931<x1<1047:
                    header=overlayList[9]
                    drawColor=(96,32,0)
                elif 1047<x1<1164:
                    header=overlayList[10]
                    drawColor=(160,48,112)
                elif 1164<x1<1280:
                    header=overlayList[11]
                    drawColor=(0,0,0)
            cv2.rectangle(img, (x1,y1-25), (x2,y2+25), drawColor, cv2.FILLED)
 
    
        #5. Drawing mode - Index finger is up
        if fingers[1] and fingers[2]==False:
            cv2.circle(img, (x1,y1-15), 15, drawColor, cv2.FILLED)
            if xp==0 and yp==0:
                xp, yp =x1, y1
            cv2.line(img,(xp,yp), (x1,y1), drawColor, brushThickness)
            cv2.line(imgCanvas,(xp,yp), (x1,y1), drawColor, brushThickness)
            xp, yp=x1,y1
            
            
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_or(img,imgCanvas)
    #
    img[0:108, 0:1280]=header
    cv2.imshow("Image", img)
    cv2.imshow("ImageCanvas", imgCanvas)
    cv2.waitKey(1)
    