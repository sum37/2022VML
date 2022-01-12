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
# cv2.namedWindow('Face_1')
# cv2.namedWindow('Face_2')
cv2.namedWindow('Combine')

#load two images    
img1 = dlib.load_rgb_image("./winter.jpg")      
img2 = dlib.load_rgb_image("./me.jpg")

#match picture size
width1=img1.shape[1]
height1=img1.shape[0]
width2=img2.shape[1]
height2=img2.shape[0]

if width2>=width1:
    img2=cv2.resize(img2, (width1, height1))
    small_img=img1
    big_img=img2
else:
    img1=cv2.resize(img1, (width2, height2))
    small_img=img2
    big_img=img2

#eye detecting, img1
cvsmall_img = swapRGB2BGR(small_img, small_img)    
dets1 = detector(small_img, 1)
    
left_x_1=0
left_y_1=0
right_x_1=0
right_y_1=0
    
for k, d in enumerate(dets1):   
    shape = predictor(small_img, d) ##shape size가 68개 
    for i in range(0, shape.num_parts):
        if i>=36 and i<=41:
            x = shape.part(i).x
            y = shape.part(i).y
            cv2.putText(cvsmall_img, str(i), (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (0, 255, 0))
            left_x_1+=x
            left_y_1+=y
        if i>=42 and i<=47:
            x = shape.part(i).x
            y = shape.part(i).y
            cv2.putText(cvsmall_img, str(i), (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (0, 255, 0))
            right_x_1+=x
            right_y_1+=y
        
    cv2.circle(cvsmall_img, (int(left_x_1/6),int(left_y_1/6)), 3, (0, 0, 255), -1)
    cv2.circle(cvsmall_img, (int(right_x_1/6),int(right_y_1/6)), 3, (0, 0, 255), -1)
                       
    # cv2.imshow('Face_1', cvsmall_img)

left_x_1=left_x_1/6
left_y_1=left_y_1/6
right_x_1=right_x_1/6
right_y_1=right_y_1/6

#eye detecting, big_img
cvbig_img = swapRGB2BGR(big_img, big_img)    
dets2 = detector(big_img, 1)

left_x_2=0
left_y_2=0
right_x_2=0
right_y_2=0
    
for k, d in enumerate(dets2):   
    shape = predictor(big_img, d) ##shape size가 68개 
    for i in range(0, shape.num_parts):
        if i>=36 and i<=41:
            x = shape.part(i).x
            y = shape.part(i).y
            cv2.putText(cvbig_img, str(i), (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (0, 255, 0))
            left_x_2+=x
            left_y_2+=y
        if i>=42 and i<=47:
            x = shape.part(i).x
            y = shape.part(i).y
            cv2.putText(cvbig_img, str(i), (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (0, 255, 0))
            right_x_2+=x
            right_y_2+=y
                   
    cv2.circle(cvbig_img, (int(left_x_2/6),int(left_y_2/6)), 3, (0, 0, 255), -1)
    cv2.circle(cvbig_img, (int(right_x_2/6),int(right_y_2/6)), 3, (0, 0, 255), -1)

left_x_2=left_x_2/6
left_y_2=left_y_2/6
right_x_2=right_x_2/6
right_y_2=right_y_2/6
                         
# cv2.imshow('Face_2', cvbig_img)

#간격 비교
dis1= ((left_x_1-right_x_1)**2+(left_y_1-right_y_1)**2)**0.5
dis2= ((left_x_2-right_x_2)**2+(left_y_2-right_y_2)**2)**0.5

#비율 따지기 ....
p=dis1/dis2
small_img=cv2.resize(small_img, None, fx=1/p, fy=1/p)

#vector 생성
small_img_vector=[right_x_1-left_x_1, right_y_1-left_y_1]
big_img_vector=[right_x_2-left_x_2, right_y_2-left_y_2]
print("here vector")
print(small_img_vector)
print(big_img_vector)

#vector간 각도 구하기
unit_small_img_vector=small_img_vector/np.linalg.norm(small_img_vector)
unit_big_img_vector=big_img_vector/np.linalg.norm(big_img_vector)
dot_product=np.dot(unit_small_img_vector, unit_big_img_vector)
print(unit_small_img_vector)
print(unit_big_img_vector)
angle=np.arccos(dot_product)
print("here angle")
print(angle)

#rotate, center: eye
rows, cols=big_img.shape[:2]
N=cv2.getRotationMatrix2D((left_x_2, left_y_2), angle*180/3.14, 1)
rotation_big_img=cv2.warpAffine(big_img, N, (cols, rows))

#x1-x2만큼 이동해야함
diff_x=left_x_1-left_x_2
diff_y=left_y_1-left_y_2
M=np.float32([[1,0,diff_x], [0,1,diff_y]])
translation_big_img=cv2.warpAffine(rotation_big_img,M,(cols, rows))

#blending
alpha=0.5


# img3=small_img*alpha+translation_big_img*(1-alpha)
# img3=img3.astype(np.uint8)

# #show img3
# cvImg3 = swapRGB2BGR(img3, img3) 
# cv2.imshow('Combine', cvImg3)

#break if 
while True:
    if cv2.waitKey(0) == ESC_KEY:
        break;
        
cv2.destroyWindow('Face')