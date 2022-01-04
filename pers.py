import numpy as np
import cv2


point_list = []
sum_list=[]
count = 0

def mouse_callback(event, x, y, flags, param):
    global point_list, count, img_original

    if event == cv2.EVENT_LBUTTONDOWN:
        print("(%d, %d)" % (x, y))
        point_list.append((x, y))

        print(point_list)
        cv2.circle(img_original, (x, y), 3, (0, 0, 255), -1)

# def align_point(point_list):
#     global sum_list
#     for i in point_list:
#         sum=point_list[i][0]+point_list[i][1]
#         sum_list.append(sum)
        

cv2.namedWindow('original')
cv2.setMouseCallback('original', mouse_callback)

# align_point(point_list)

# 원본 이미지
img_original = cv2.imread('paper.png')


while(True):

    cv2.imshow("original", img_original)


    height, width = img_original.shape[:2]


    if cv2.waitKey(1)&0xFF == 32: # spacebar를 누르면 루프에서 빠져나옵니다.
        break

# print(sum_list)
# print('here')

# 좌표 순서 - 상단왼쪽 끝, 상단오른쪽 끝, 하단왼쪽 끝, 하단오른쪽 끝
pts1 = np.float32([list(point_list[0]),list(point_list[1]),list(point_list[2]),list(point_list[3])])
pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])

print(pts1)
print(pts2)

M = cv2.getPerspectiveTransform(pts1,pts2)

img_result = cv2.warpPerspective(img_original, M, (width,height))


cv2.imshow("result1", img_result)
cv2.waitKey(0)
cv2.destroyAllWindows()