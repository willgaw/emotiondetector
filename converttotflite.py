"""
Convert a Keras model to TFLite model for use on RaspberryPI
    
"""
import tensorflow as tf
from keras.models import load_model

emotion_model = load_model("models/emotion_detection_model_100epochs.h5", compile=False)

converter = tf.lite.TFLiteConverter.from_keras_model(emotion_model)

tflite_model=converter.convert()

open("models/emotion_detection_model_100epochs_no_opt.tflite","wb").write(tflite_model)
