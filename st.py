import cv2
import dlib


img=cv2.imread('duck.jpg')
dst=cv2.resize(img, (500,500)) #크기가 너무 커서 조정함
rotate90=cv2.rotate(dst, cv2.ROTATE_90_CLOCKWISE) #90도 회전
cv2.imshow('duck', dst) #원본 띄우기
cv2.imshow('duck90', rotate90) #90도 회전한 것 띄우기

cv2.waitKey(0) #키보드 입력 생길 때까지 기다리기
cv2.destroyAllWindows() #키보드 입력 발생하면 창 닫기