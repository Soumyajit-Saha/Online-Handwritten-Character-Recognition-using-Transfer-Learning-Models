#!/usr/bin/env python
# coding: utf-8

# In[1]:


IMG_WIDTH=224
IMG_HEIGHT=224
IMG_DIM = (IMG_WIDTH, IMG_HEIGHT)
from keras.applications.resnet50 import ResNet50
from keras.models import Model
import keras
restnet = ResNet50(include_top=False, weights=None, input_shape=(224,224,3))
output = restnet.layers[-1].output
output = keras.layers.Flatten()(output)
restnet = Model(restnet.input, output=output)
for layer in restnet.layers:
    layer.trainable = True
restnet.summary()


# In[ ]:





# In[2]:


from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers
model = Sequential()
model.add(restnet)
model.add(Dense(512, activation='relu', input_dim=(224,224,3)))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(50, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])
model.summary()


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
# 

# In[ ]:


from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("ResNet50.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=40, verbose=1, mode='auto')
history = model.fit_generator(generator= traindata, steps_per_epoch= 10000//10, epochs= 500, validation_data= testdata, validation_steps=1, callbacks=[early])

