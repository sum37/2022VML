import cv2
import numpy as np

img=cv2.imread('./winter.jpg')

height=img.shape[0]
width=img.shape[1]

bg_width=width+100
bg_height=height+100
color=(0,0,0)
bg=np.full((bg_height, bg_width, img.shape[2]), color, dtype=np.uint8)

x_center = (bg_width - width) // 2
y_center = (bg_height - height) // 2

# copy img image into center of result image
bg[y_center:y_center+height, 
       x_center:x_center+width] = img

# view result
cv2.imshow("result", bg)
cv2.waitKey(0)
cv2.destroyAllWindows()

# save result
cv2.imwrite("./winter.jpg", bg)