import numpy as np
import cv2

point_list=[]
count=0

def mouse_callback(event, x, y, flags, param):
    # 마우스 왼쪽 버튼을 누르면 list에 좌표를 설정한다. 
    global point_list, img_original
    
    if event == cv2.EVENT_LBUTTONDOWN:
        print("(%d, %d)" %(x,y))
        point_list.append((x,y))
        
        print(point_list)
        cv2.circle(img_original, (x,y), 3, (0, 0, 255), -1)
        #좌표 목록 출력하고 그 지점에 빨간색 원 그리기

cv2.namedWindow('original') #이름이 original인 창 생성
cv2.setMouseCallback('original', mouse_callback) #window 'original'에 마우스 클릭이 발생하면 mouse_callback 함수 실행

img_original=cv2.imread('me.jpg') ## input image

while True:
    cv2.imshow("original", img_original)
    height, width, channel = img_original.shape
    if cv2.waitKey(1)&0xFF == 32: # spacebar를 누르면 루프에서 빠져나옵니다.
        break
    
pts1 = np.float32([list(point_list[0]),list(point_list[1]),list(point_list[2])])
pts2 = np.float32([list(point_list[0]),list(point_list[1]),list(point_list[2])])

pts2[1][1] += 100 # 두번째 점의 Y 좌표를 아래로 이동


M = cv2.getAffineTransform(pts1,pts2)

img_result = cv2.warpAffine(img_original, M, (width,height))


cv2.imshow("result1", img_result)
cv2.waitKey(0)


pts1 = np.float32([list(point_list[0]),list(point_list[1]),list(point_list[2])])
pts2 = np.float32([list(point_list[0]),list(point_list[1]),list(point_list[2])])

pts2[2][0] += 100 # 세번째 점의 X 좌료를 오른쪽으로 이동


M = cv2.getAffineTransform(pts1,pts2)

img_result = cv2.warpAffine(img_original, M, (width,height))


cv2.imshow("result2", img_result)
cv2.waitKey(0)



pts1 = np.float32([list(point_list[0]),list(point_list[1]),list(point_list[2])])
pts2 = np.float32([list(point_list[0]),list(point_list[1]),list(point_list[2])])

pts2[1][1] += 100 # 두번째 점의 Y 좌표를 아래로 이동
pts2[2][0] += 50  # 세번째 점의 X 좌료를 오른쪽으로 이동


M = cv2.getAffineTransform(pts1,pts2)

img_result = cv2.warpAffine(img_original, M, (width,height))


cv2.imshow("result3", img_result)
cv2.waitKey(0)


cv2.destroyAllWindows()