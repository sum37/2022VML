import cv2
import numpy as np

img=cv2.imread('./me.jpg')

height=img.shape[0]
width=img.shape[1]

# new window
cv2.namedWindow('new')
cv2.resizeWindow('new', width*2, height*2)

bg_height=height*2
bg_width=width*2

black_img=np.full((bg_height, bg_width, 3), (0,0,0), dtype=np.uint8)

for cols_index in range(bg_width):
       for rows_index in range(bg_height):
              if rows_index>=bg_height/4 and rows_index<=bg_height*3/4:
                     if cols_index>=bg_width/4 and cols_index<=bg_width*3/4:
                            black_img[rows_index][cols_index]=img[rows_index-int(bg_height/4)-1, cols_index-int(bg_width/4)-1]

cv2.imshow('new', black_img)

cv2.waitKey(0)
cv2.destroyAllWindows()