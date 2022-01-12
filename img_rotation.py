import cv2
import numpy as np

img=cv2.imread('./taeyeon.jpg')

height=img.shape[0]
width=img.shape[1]

bg_width=width*2
bg_height=height*2
color=(0,0,0)
bg=np.full((bg_height, bg_width, img.shape[2]), color, dtype=np.uint8)

x_center = (bg_width - width) // 2
y_center = (bg_height - height) // 2

rows, cols=img.shape[:2]
N=cv2.getRotationMatrix2D((rows/2, cols/2), 45, 1)
rotation_img2=cv2.warpAffine(img, N, (cols, rows))

# copy img image into center of result image
bg[y_center:y_center+height, 
       x_center:x_center+width] = rotation_img2

# view result
cv2.imshow("result", bg)
cv2.waitKey(0)
cv2.destroyAllWindows()