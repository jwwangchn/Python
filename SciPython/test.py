#coding:utf8
import cv2

filename='lena.jpg'
img=cv2.imread(filename)
print type(img),img.shape,img.dtype
cv2.namedWindow('demo1')
cv2.imshow('demo1',img)
cv2.waitKey(0)