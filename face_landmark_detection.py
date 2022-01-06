import sys
import os
import dlib
import glob
import cv2

ESC_KEY = 27

def swapRGB2BGR(rgb):
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

cv2.namedWindow('Face')

for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    print("Processing file: {}".format(f))
    
    img = dlib.load_rgb_image(f)      
    
    cvImg = swapRGB2BGR(img)    
    dets = detector(img, 1)

    print("Number of faces detected: {}".format(len(dets)))
    
    left_x=0
    left_y=0
    right_x=0
    right_y=0
    
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        
        print("Here: only eye")
        
        shape = predictor(img, d) ##shape size가 68개 
        print(shape.num_parts)
        for i in range(0, shape.num_parts):
            if i>=36 and i<=41:
                x = shape.part(i).x
                y = shape.part(i).y
                print(str(x) + " " + str(y))
                cv2.putText(cvImg, str(i), (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (0, 255, 0))
                left_x+=x
                left_y+=y
            if i>=42 and i<=47:
                x = shape.part(i).x
                y = shape.part(i).y
                print(str(x) + " " + str(y))
                cv2.putText(cvImg, str(i), (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (0, 255, 0))
                right_x+=x
                right_y+=y
        print(left_x/6)
        print(left_y/6)
        print(right_x/6)
        print(right_y/6)   
        
        cv2.circle(cvImg, (int(left_x/6),int(left_y/6)), 3, (0, 0, 255), -1)
        cv2.circle(cvImg, (int(right_x/6),int(right_y/6)), 3, (0, 0, 255), -1)
                       
        cv2.imshow('Face', cvImg)


    while True:
        if cv2.waitKey(0) == ESC_KEY:
            break;
        
cv2.destroyWindow('Face')