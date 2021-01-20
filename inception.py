#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import decode_predictions


# In[2]:


from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


model = InceptionV3(weights='imagenet', include_top=True)


# In[4]:


IMG_WIDTH=224
IMG_HEIGHT=224
IMG_DIM = (IMG_WIDTH, IMG_HEIGHT)

from keras.models import Model
import keras
model = InceptionV3(include_top=False, weights='imagenet', input_shape=(224,224,3))
output = model.layers[-1].output
output = keras.layers.Flatten()(output)
model = Model(model.input, output=output)
for layer in model.layers:
    layer.trainable = False
model.summary()


# In[5]:


from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers
model1 = Sequential()
model1.add(model)
model1.add(Dense(512, activation='relu', input_dim=(224,224,3)))
model1.add(Dropout(0.3))
model1.add(Dense(512, activation='relu'))
model1.add(Dropout(0.3))
model1.add(Dense(48, activation='sigmoid'))
model1.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])
model1.summary()


# In[6]:


import keras
from keras.models import Model
from keras.layers import Dense
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory="C:/Users/RAJATSUBHRA/Desktop/RESEARCH WORK COLLEGE/DEVNAGARI_NEW/TRAIN",target_size=(224,224),batch_size=10)
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory="E:/ROTATED IMAGES2", target_size=(224,224),batch_size=10)


# In[7]:


from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("InceptionV3.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=40, verbose=1, mode='auto')
history = model1.fit_generator(generator= traindata, steps_per_epoch= 4944//10, epochs= 100, validation_data= testdata, validation_steps=1, callbacks=[checkpoint,early])

