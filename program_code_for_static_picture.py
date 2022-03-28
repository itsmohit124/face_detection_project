import numpy as np
import cv2
from random import randrange
#training algorithm
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#chosing an image to detect
img = cv2.imread('Escot.jpg')
#converting it to greyscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Detect Faces
face_coordinates = trained_face_data.detectMultiScale(gray_img)

#draw a rectangle around the face
for (x,y,w,h) in face_coordinates:
    cv2.rectangle( img ,(x,y), (x+w,y+h), (randrange(256),randrange(256),randrange(256)), 5)

cv2.imshow('Face Detector', img)
cv2.waitKey()
print("Code Complete")
