
"""
Simple test of Face detection using opencv (Haar Cascade classificaion)
With RaspberryPI official PiCamera

face model:
    https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    
"""
from picamera2 import Picamera2
import cv2
import time

face_cascade = cv2.CascadeClassifier('haarcascades_models/haarcascade_frontalface_default.xml')

# Capture video on PICamera
camera = Picamera2()
config = camera.create_preview_configuration({'format': 'RGB888'})
camera.configure(config)
camera.start()

time.sleep(0.1)


while True:
    # grab a frame from the stream    
    img = camera.capture_array()
    
    #convert to greyscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # detect faces in the image - returns array of faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # loop over each face and draw a blue rectangle around area
    # co-ordinates are bottom left hand corner, then width and height
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow('img',img)
    
    # break when the user presses q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#cleanup resources
cv2.destroyAllWindows()