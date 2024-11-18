
"""
Live prediction of emotion using pre-trained model.
Uses Opencv with haar Cascades classifier to detect face.
then, uses pre-trained TFLite models to detect emotion from webcam

face model:
    https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

"""

import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

# This was the only requirement on Keras, so I copied this function from Keras source
def img_to_array(img, data_format='channels_last', dtype='float32'):
    """Converts a PIL Image instance to a Numpy array.
    # Arguments
        img: PIL Image instance.
        data_format: Image data format,
            either "channels_first" or "channels_last".
        dtype: Dtype to use for the returned array.
    # Returns
        A 3D Numpy array.
    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: %s' % data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=dtype)
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: %s' % (x.shape,))
    return x



# Model returns integers, mapping to convert to indentifiable labels
class_labels=['Angry','Disgust', 'Fear', 'Happy','Neutral','Sad','Surprise']

# Use pre-confiugred OpenCV face classifier for detecting faces when facing camera front-on
face_classifier=cv2.CascadeClassifier('haarcascades_models/haarcascade_frontalface_default.xml')

# load the prebuilt tflite model trained earlier
emotion_interpreter = Interpreter(model_path="models/emotion_detection_model_100epochs_no_opt.tflite")
emotion_interpreter.allocate_tensors()


emotion_input_details = emotion_interpreter.get_input_details()
emotion_output_details = emotion_interpreter.get_output_details()

emotion_input_shape = emotion_input_details[0]['shape']

videoStream=cv2.VideoCapture(0, cv2.CAP_V4L)


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
        emotion_interpreter.set_tensor(emotion_input_details[0]['index'], roi)
        emotion_interpreter.invoke()
        emotion_preds = emotion_interpreter.get_tensor(emotion_output_details[0]['index'])

        emotion_label=class_labels[emotion_preds.argmax()]
        emotion_label_position=(x,y)	
        cv2.putText(frame,emotion_label,emotion_label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    
        
   # Show the full colour image with all the additional added features
    cv2.imshow('Emotion Predictor', frame)
    
    # break when the user presses q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
#clean up resources
videoStream.release()
cv2.destroyAllWindows()

