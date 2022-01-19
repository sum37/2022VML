import cv2
import numpy as np

img=cv2.imread('./taeyeon.jpg')

height=img.shape[0]
width=img.shape[1]

for cols_index in range(width):
       for rows_index in range(height):
           if cols_index%10==0 or rows_index%10==0:
               img[rows_index][cols_index]=(255,0,0)

# view result
cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()