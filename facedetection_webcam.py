
"""
Simple test of Face detection using opencv (Haar Cascade classificaion)

face model:
    https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    
"""

import cv2

face_cascade = cv2.CascadeClassifier('haarcascades_models/haarcascade_frontalface_default.xml')

# Capture video on the primary webcam
cap = cv2.VideoCapture(0)

while 1:
    # grab a frame from the stream
    ret, img = cap.read()
    
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
cap.release()
cv2.destroyAllWindows()