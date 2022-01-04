import cv2

img = cv2.imread('duck.jpg')
img=cv2.resize(img, (500,500))
rows, cols = img.shape[:2]


# 이미지의 중심점을 기준으로 90도 회전 하면서 0.5배 Scale
M= cv2.getRotationMatrix2D((cols/2, rows/2),45, 1)

dst = cv2.warpAffine(img, M,(cols, rows))

cv2.imshow('Original', img)
cv2.imshow('Rotation', dst)

cv2.waitKey(0)
cv2.destroyAllWindows()