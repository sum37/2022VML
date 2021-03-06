import cv2
import time
import HandTrackingModule as htm

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime=0

detector=htm.handDetector(detectionCon=0.75)

tipIds=[4,8,12,16,20] #4: 엄지 끝 8: 검지끝 12: 중지 끝 ...

while True:
    success, img = cap.read()
    img=detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    
    if len(lmList)!=0:
        fingers=[]
        
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        for id in range(1, 5):
            if lmList[tipIds[id]][2]<lmList[tipIds[id]-2][2]:
                fingers.append(1) #open이면 1
            else: 
                fingers.append(0) #close면 0 
        totalfinger=fingers.count(1)
        print(totalfinger)

    
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    
    
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3,(255,0,255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    
