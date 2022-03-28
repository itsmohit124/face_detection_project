import cv2
import numpy as np
from random import randrange
#training algorithm
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#chosing an image to detect
webcam =cv2.VideoCapture(0)

while True:
    #for reading current frame
    successful_frame_read, frame= webcam.read()
    #to convert it into grayscale
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #to find the coordinate of face
    face_coordinates= trained_face_data.detectMultiScale(gray_img) 
    #draw a rectangle around the face
    for (x,y,w,h) in face_coordinates:
         cv2.rectangle(frame,(x,y), (x+w,y+h), (randrange(256),randrange(256),randrange(256)), 5)
    #displaying the output
    cv2.imshow('Face Detector', frame)
    key =  cv2.waitKey(1)

    #TO stop if 0 key is pressed
    if key==81 or key==113:
        break 
