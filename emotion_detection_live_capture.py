
"""
Live prediction of emotion using pre-trained model.
Uses Opencv with haar Cascades classifier to detect face.
then, uses pre-trained models to detect  emotion from webcam

face model:
    https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

"""

from keras.models import load_model
from tensorflow.keras.utils import img_to_array
import cv2
import numpy as np

# Model returns integers, convert to indentifiable labels
class_labels=['Angry','Disgust', 'Fear', 'Happy','Neutral','Sad','Surprise']

# Use pre-confiugred OpenCV face classifier for detecting faces when facing camera front-on
face_classifier=cv2.CascadeClassifier('haarcascades_models/haarcascade_frontalface_default.xml')

# load the prebuilt keras model trained earlier
emotion_model = load_model('models/emotion_detection_model_100epochs.h5')


#Setup video capture on channel 0 (primary webcam)
videoStream=cv2.VideoCapture(0)

# Main loop - runs forever until the user hits the q key
while True:
    # Capture the next video frame
    retval,frame=videoStream.read()
    
    # convert the colour stream to grayscale - note OpenCV is BGR not RGB
    grayscale=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    # Detect faces bounds- note returns an array with lower LHS corner x,y then width and height
    faces=face_classifier.detectMultiScale(grayscale,1.3,5)

    # Iterate over all detected faces - works with multiple faces in the frame
    for (x,y,w,h) in faces:
        # Draw a blue rectange on the image for the detected face 2 pixels wide
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        # Crop the image to the identified face
        roi_gray=grayscale[y:y+h,x:x+w]
        #roi_gray=frame[y:y+h,x:x+w]
        # Reszie the image to 48 by 48 pixels to map to the model input layer
        roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        # Modify the image for the model
        # Scale the image
        roi=roi_gray.astype('float')/255.0
        # Convert the image to a 3D NumPy array
        roi=img_to_array(roi)        
        # Expand dimensions for input into the model (1,48,48,1)
        roi=np.expand_dims(roi,axis=0)

        # Predict the emotion, output is an integer for the 7 classes (0-6)
        preds=emotion_model.predict(roi)[0]
        #lookup result to convert to label
        label=class_labels[preds.argmax()]
        
        #Print the emotion next to the rectangle in green
        label_position=(x,y)
        cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    
        
   # Show the full colour image with all the additional added features
    cv2.imshow('Emotion Detector', frame)
    
    # break when the user presses q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
#clean up resources
videoStream.release()
cv2.destroyAllWindows()