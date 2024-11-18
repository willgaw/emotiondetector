
"""
Train a deep learning model for facial emotion detection

Dataset from: https://www.kaggle.com/msambare/fer2013
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
import os
from matplotlib import pyplot as plt
import random

# Setup params for model training
IMG_HEIGHT=48 
IMG_WIDTH = 48
batch_size=32
epochs=100

train_data_dir='data/train/'
validation_data_dir='data/test/'

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# run image data generator with random modifications to avoid overfitting
train_datagen = ImageDataGenerator(
 					rescale=1./255,
 					rotation_range=30,
 					shear_range=0.3,
 					zoom_range=0.3,
 					horizontal_flip=True,
 					fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
 					train_data_dir,
 					color_mode='grayscale',
 					target_size=(IMG_HEIGHT, IMG_WIDTH),
 					batch_size=batch_size,
 					class_mode='categorical',
 					shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
 							validation_data_dir,
 							color_mode='grayscale',
 							target_size=(IMG_HEIGHT, IMG_WIDTH),
 							batch_size=batch_size,
 							class_mode='categorical',
 							shuffle=True)

#Verify our generator by plotting a few faces and printing corresponding labels
class_labels=['Angry','Disgust', 'Fear', 'Happy','Neutral','Sad','Surprise']

img, label = train_generator.__next__()
i=random.randint(0, (img.shape[0])-1)
image = img[i]
labl = class_labels[label[i].argmax()]
plt.imshow(image[:,:,0], cmap='gray')
plt.title(labl)
plt.show()



# Create the model
model = Sequential()

# Add the model layers
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(7, activation='softmax'))

# Compile the model
model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print a text summary of the model
print(model.summary())

filename = 'models/emotion_new_' + str(epochs) + 'epochs.h5'
print(filename)


train_path = "data/train/"
test_path = "data/test"

num_train_imgs = 0
for root, dirs, files in os.walk(train_path):
    num_train_imgs += len(files)
    
num_test_imgs = 0
for root, dirs, files in os.walk(test_path):
    num_test_imgs += len(files)

# Train the model
history=model.fit(train_generator,
                steps_per_epoch=num_train_imgs//batch_size,
                epochs=epochs,
                validation_data=validation_generator,
                validation_steps=num_test_imgs//batch_size)

# Save the model as a .h5 file
# Can be validated using emotion_detection_model_validator.py
# Reuse the model in emotion_detection_live_capture.py
model.save(filename)

#plot the training and validation loss at each epoch on a chart
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#plot the training and validation loss at each epoch on a chart
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

plt.plot(epochs, accuracy, 'y', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation accuraCY')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


