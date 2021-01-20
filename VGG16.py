#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


from keras.applications.vgg16 import VGG16
vggmodel = VGG16(weights='imagenet', include_top=True)
vggmodel.summary()


# In[3]:


for layers in (vggmodel.layers)[:19]:
    print(layers)
    layers.trainable = False
    X= vggmodel.layers[-2].output
predictions = Dense(48, activation="softmax")(X)
model_final = Model(input = vggmodel.input, output = predictions)
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
model_final.summary()


# In[ ]:


from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=40, verbose=1, mode='auto')
model_final.fit_generator(generator= traindata, steps_per_epoch= 4944//10, epochs= 100, validation_data= testdata, validation_steps=1, callbacks=[checkpoint,early])
model_final.save_weights("vgg16_1.h5")


# In[ ]:


traindata


# In[ ]:





# In[ ]:




