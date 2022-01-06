import sys
import os
import dlib
import glob
import cv2
import numpy as np

ESC_KEY = 27

def swapRGB2BGR(rgb, img):
    r, g, b = cv2.split(img)
    bgr = cv2.merge([b,g,r])
    return bgr

if len(sys.argv) != 3:
    print(
        "Give the path to the trained shape predictor model as the first "
        "argument and then the directory containing the facial images.\n"
        "For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        "    ./face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../examples/faces\n"
        "You can download a trained facial shape predictor from:\n"
        "    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    exit()

predictor_path = sys.argv[1]
faces_folder_path = sys.argv[2]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#create window
cv2.namedWindow('Face_1')
cv2.namedWindow('Face_2')
cv2.namedWindow('Combine')

#load two images    
img1 = dlib.load_rgb_image("./me.jpg")      
img2 = dlib.load_rgb_image("./winter.jpg")

#match picture size
width=img1.shape[1]
height=img1.shape[0]
img2=cv2.resize(img2, (width, height))

#eye detecting, img1
cvImg1 = swapRGB2BGR(img1, img1)    
dets1 = detector(img1, 1)

print("Number of faces detected: {}".format(len(dets1)))
    
left_x_1=0
left_y_1=0
right_x_1=0
right_y_1=0
    
for k, d in enumerate(dets1):   
    shape = predictor(img1, d) ##shape size가 68개 
    print(shape.num_parts)
    for i in range(0, shape.num_parts):
        if i>=36 and i<=41:
            x = shape.part(i).x
            y = shape.part(i).y
            print(str(x) + " " + str(y))
            cv2.putText(cvImg1, str(i), (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (0, 255, 0))
            left_x_1+=x
            left_y_1+=y
        if i>=42 and i<=47:
            x = shape.part(i).x
            y = shape.part(i).y
            print(str(x) + " " + str(y))
            cv2.putText(cvImg1, str(i), (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (0, 255, 0))
            right_x_1+=x
            right_y_1+=y
        print(left_x_1/6)
        print(left_y_1/6)
        print(right_x_1/6)
        print(right_y_1/6)   
        
    cv2.circle(cvImg1, (int(left_x_1/6),int(left_y_1/6)), 3, (0, 0, 255), -1)
    cv2.circle(cvImg1, (int(right_x_1/6),int(right_y_1/6)), 3, (0, 0, 255), -1)
                       
    cv2.imshow('Face_1', cvImg1)

#eye detecting, img2
cvImg2 = swapRGB2BGR(img2, img2)    
dets2 = detector(img2, 1)

left_x_2=0
left_y_2=0
right_x_2=0
right_y_2=0
    
for k, d in enumerate(dets2):   
    shape = predictor(img2, d) ##shape size가 68개 
    print(shape.num_parts)
    for i in range(0, shape.num_parts):
        if i>=36 and i<=41:
            x = shape.part(i).x
            y = shape.part(i).y
            print(str(x) + " " + str(y))
            cv2.putText(cvImg2, str(i), (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (0, 255, 0))
            left_x_2+=x
            left_y_2+=y
        if i>=42 and i<=47:
            x = shape.part(i).x
            y = shape.part(i).y
            print(str(x) + " " + str(y))
            cv2.putText(cvImg2, str(i), (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (0, 255, 0))
            right_x_2+=x
            right_y_2+=y
        print(left_x_2/6)
        print(left_y_2/6)
        print(right_x_2/6)
        print(right_y_2/6)   
        
    cv2.circle(cvImg2, (int(left_x_2/6),int(left_y_2/6)), 3, (0, 0, 255), -1)
    cv2.circle(cvImg2, (int(right_x_2/6),int(right_y_2/6)), 3, (0, 0, 255), -1)
                       
cv2.imshow('Face_2', cvImg2)

#blending
alpha=0.5
img3=img1*alpha+img2*(1-alpha)
img3=img3.astype(np.uint8)

#show img3
cvImg3 = swapRGB2BGR(img3, img3) 
cv2.imshow('Combine', cvImg3)

#break if 
while True:
    if cv2.waitKey(0) == ESC_KEY:
        break;
        
cv2.destroyWindow('Face')