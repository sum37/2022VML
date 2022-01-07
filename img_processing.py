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
cv2.namedWindow('Hi')

#load two images    
img1 = dlib.load_rgb_image("./winter.jpg")      
img2 = dlib.load_rgb_image("./me.jpg")



#break if 
while True:
    if cv2.waitKey(0) == ESC_KEY:
        break;
        
cv2.destroyWindow('Face')