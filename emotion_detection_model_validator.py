"""
Validate and generate report for pretrained model against fer2013

Dataset from: https://www.kaggle.com/msambare/fer2013
"""

from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from keras.models import load_model
from keras.utils.vis_utils import plot_model
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns

import numpy as np
import random
import os

IMG_HEIGHT=48 
IMG_WIDTH = 48
batch_size=256

train_data_dir='data/train/'
validation_data_dir='data/test/'
model_name = 'emotion_50epochs.h5'

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Just load the standard image files - no need to modify to avoid overfitting
validation_datagen = ImageDataGenerator(rescale=1./255)


validation_generator = validation_datagen.flow_from_directory(
 							validation_data_dir,
 							color_mode='grayscale',
 							target_size=(IMG_HEIGHT, IMG_WIDTH),
 							batch_size=batch_size,
 							class_mode='categorical',
 							shuffle=True)

#Verify our generator by plotting a few faces and printing corresponding labels
class_labels=['Angry','Disgust', 'Fear', 'Happy','Neutral','Sad','Surprise']


#Test the model
my_model = load_model('models/emotion_detection_model_300epochs.h5', compile=False)

#plot_model(my_model)

#Generate a batch of images
test_img, test_lbl = validation_generator.__next__()
# Generate predictions on images
predictions=my_model.predict(test_img)

predictions = np.argmax(predictions, axis=1)
test_labels = np.argmax(test_lbl, axis=1)

accuracy = metrics.accuracy_score(test_labels, predictions)
print ("Accuracy = ", accuracy)

#Draw Confusion Matrix with absolute numbers
cm = confusion_matrix(test_labels, predictions)
map = sns.heatmap(cm, annot=True, xticklabels=class_labels, yticklabels=class_labels, cmap='Blues')
map.set_xticklabels(map.get_xticklabels(), rotation = 90)

n=random.randint(0, test_img.shape[0] - 1)
image = test_img[n]
orig_labl = class_labels[test_labels[n]]
pred_labl = class_labels[predictions[n]]
plt.xlabel('prediction')
plt.ylabel('label')
plt.imshow(image[:,:,0], cmap='gray')
plt.title(model_name+" test of "+str(batch_size)+" samples, accuracy " + str(accuracy))
plt.show()
