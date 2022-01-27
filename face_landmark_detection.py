import sys
from turtle import left, right, width
import dlib
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

#load two images    
img1 = dlib.load_rgb_image("./taeyeon.jpg")      
img2 = dlib.load_rgb_image("./winter.jpg")
img2=cv2.resize(img2, (img2.shape[1]//4, img2.shape[0]//4))

#size
width1=img1.shape[1]
height1=img1.shape[0]
width2=img2.shape[1]
height2=img2.shape[0]

print("here is size with padding")
print(width1)
print(height1)
print(width2)
print(height2)

#padding: To prevent the picture from being cut off
padding1=np.full((height1*2, width1*2, 3), (0,0,0), dtype=np.uint8)
padding2=np.full((height2*2, width2*2, 3), (0,0,0), dtype=np.uint8)
padding1[height1//2:height1*3//2, width1//2:width1*3//2] = img1
padding2[height2*2//4:height2*3//2, width2//2:width2*3//2] = img2

#new size
width1=padding1.shape[1]
height1=padding1.shape[0]
width2=padding2.shape[1]
height2=padding2.shape[0]

print("here is size with padding")
print(width1)
print(height1)
print(width2)
print(height2)

#eye detecting, img1
img1 = swapRGB2BGR(padding1, padding1)
dets1 = detector(img1, 1)

left_x_1=0
left_y_1=0
right_x_1=0
right_y_1=0
    
for k, d in enumerate(dets1):   
    shape = predictor(img1, d) ##shape size가 68개 
    for i in range(0, shape.num_parts):
        if i>=36 and i<=41:
            x = shape.part(i).x
            y = shape.part(i).y
            # cv2.putText(img1, str(i), (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (0, 255, 0))
            left_x_1+=x
            left_y_1+=y
        if i>=42 and i<=47:
            x = shape.part(i).x
            y = shape.part(i).y
            # cv2.putText(img1, str(i), (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (0, 255, 0))
            right_x_1+=x
            right_y_1+=y
        
    cv2.circle(img1, (int(left_x_1/6),int(left_y_1/6)), 3, (0, 0, 255), -1)
    cv2.circle(img1, (int(right_x_1/6),int(right_y_1/6)), 3, (0, 0, 255), -1)

left_x_1=left_x_1/6
left_y_1=left_y_1/6
right_x_1=right_x_1/6
right_y_1=right_y_1/6

#eye detecting, img2
img2 = swapRGB2BGR(padding2, padding2)    
dets2 = detector(padding2, 1)

left_x_2=0
left_y_2=0
right_x_2=0
right_y_2=0
    
for k, d in enumerate(dets2):   
    shape = predictor(img2, d) ##shape size가 68개 
    for i in range(0, shape.num_parts):
        if i>=36 and i<=41:
            x = shape.part(i).x
            y = shape.part(i).y
            # cv2.putText(img2, str(i), (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (0, 255, 0))
            left_x_2+=x
            left_y_2+=y
        if i>=42 and i<=47:
            x = shape.part(i).x
            y = shape.part(i).y
            # cv2.putText(img2, str(i), (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (0, 255, 0))
            right_x_2+=x
            right_y_2+=y
                         
    cv2.circle(img2, (int(left_x_2/6),int(left_y_2/6)), 3, (0, 0, 255), -1)
    cv2.circle(img2, (int(right_x_2/6),int(right_y_2/6)), 3, (0, 0, 255), -1)
    
left_x_2=left_x_2/6
left_y_2=left_y_2/6
right_x_2=right_x_2/6
right_y_2=right_y_2/6                       

#comparing distance between eyes
dis1= ((left_x_1-right_x_1)**2+(left_y_1-right_y_1)**2)**0.5
dis2= ((left_x_2-right_x_2)**2+(left_y_2-right_y_2)**2)**0.5

#비율 따지고 큰 것에 맞추기
if dis1>=dis2:
    p=dis1/dis2
    img2=cv2.resize(img2, (int(width2*p), int(height2*p)))
    newleft_x_2=left_x_2*p
    newleft_y_2=left_y_2*p
    newleft_x_1=left_x_1
    newleft_y_1=left_y_1

else:
    p=dis2/dis1
    img1=cv2.resize(img1, (int(width1*p), int(height1*p)))
    newleft_x_1=left_x_1*p
    newleft_y_1=left_y_1*p
    newleft_x_2=left_x_2
    newleft_y_2=left_y_2

print(p)

#새로운 사이즈
width1=img1.shape[1]
height1=img1.shape[0]
width2=img2.shape[1]
height2=img2.shape[0]

print("here is last size")
print(width1)
print(height1)
print(width2)
print(height2)

#vector 생성
img1_vector=[right_x_1-left_x_1, right_y_1-left_y_1]
img2_vector=[right_x_2-left_x_2, right_y_2-left_y_2]

#vector간 각도 구하기
unit_img1_vector=img1_vector/np.linalg.norm(img1_vector)
unit_img2_vector=img2_vector/np.linalg.norm(img2_vector)
dot_product=np.dot(unit_img1_vector, unit_img2_vector)
angle=np.arccos(dot_product)

#rotate, center: eye
N=cv2.getRotationMatrix2D((left_x_2, left_y_2), angle*180/3.14, 1)
img2=cv2.warpAffine(img2, N, (width2, height2))

cv2.imshow('hji', img1)
cv2.imshow('ddf', img2)
cv2.waitKey(0)

#padding 가로, 세로 결정하기
sizelist=[width1, width2, height1, height2]
newsize=max(sizelist)

#사진을 겹치기 위해서 필요한 값들 .... 
x_padding1 = (newsize - width1) // 2
y_padding1 = (newsize - height1) // 2
x_padding2 = (newsize - width2) // 2
y_padding2 = (newsize - height2) // 2

# # x1-x2만큼 이동해야함
# diff_x=newleft_x_1-newleft_x_2
# diff_y=newleft_y_1-newleft_y_2
# M=np.float32([[1,0,diff_x], [0,1,diff_y]])
# img2=cv2.warpAffine(img2,M,(width2, height2))

#making black img
blackimg1=np.full((newsize, newsize, 3), (0,0,0), dtype=np.uint8)
blackimg2=np.full((newsize, newsize, 3), (0,0,0), dtype=np.uint8)


# copy img image into center of result image
blackimg1[y_padding1:y_padding1+height1, x_padding1:x_padding1+width1] = img1
blackimg2[y_padding2:y_padding2+height2, x_padding2:x_padding2+width2] = img2

#blending
alpha=0.5
img3=blackimg1*alpha+blackimg2*(1-alpha)
img3=img3.astype(np.uint8)

cv2.imshow('combine', img3)

while True:
    if cv2.waitKey(0) == ESC_KEY:
        break;
        
cv2.destroyWindow('Face')