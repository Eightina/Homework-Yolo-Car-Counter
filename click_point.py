from PIL import Image
from pylab import *
import cv2
import numpy as np

def save_image(image,addr,num):
    address = addr + str(num)+ '.jpg'
    cv2.imwrite(address,image)

videoCapture = cv2.VideoCapture("test2.mp4")
success, frame = videoCapture.read()
i = 0
timeF = 600
j=0
while success :
    i = i + 1
    if (i % timeF == 0):
        j = j + 1
        save_image(frame,'./clk_pnt/',j)
        print('save image:',i)
    success, frame = videoCapture.read()
    
#Get point coorinate    
im = array(Image.open(r'C:\python\ObjectDetection-YOLO\clk_pnt\1.jpg'))
imshow(im)
print('Please click 8 points')
x =ginput(8)
print(x)
show()