import cv2
import numpy as np

img=cv2.imread('./taeyeon.jpg')

height=img.shape[0]
width=img.shape[1]

img[0:width,0:50] = (0,255,0)
img[0:width,100:150] = (0,255,0)
img[0:width,200:250] = (0,255,0)
img[0:width,300:350] = (0,255,0)
img[0:width,400:450] = (0,255,0)
img[0:width,500:550] = (0,255,0)
img[0:width,600:650] = (0,255,0)
img[0:width,700:750] = (0,255,0)

# view result
cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()