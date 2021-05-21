import keras
from keras.models import Model
import numpy as np
from keras.models import load_model
from numpy import load
from keras.applications.vgg16 import VGG16

vgg = VGG16(include_top=False, input_shape=(160, 160, 3))
output = vgg.layers[-1].output
output = keras.layers.Flatten()(output)
vgg_model = Model(vgg.input, output)

set_trainable = False
    for layer in vgg_model.layers:
        if layer.name in ['block5_conv1', 'block4_conv1']:
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
        
from keras.preprocessing.image import ImageDataGenerator
#able to roate the images,and do some transformations to make the model more robust

image_gen=ImageDataGenerator(rotation_range=30,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             rescale=1/255,
                             shear_range=0.2,#cuts away part of imag
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode="nearest")#if we are gonna stretch and rotate there
                                                #there are some missing pixels,it fills mixing 
                                                #pixels by nearest

from keras.models import Sequential
from keras.layers import Activation,Dropout,Conv2D,MaxPooling2D,Flatten,Dense
from keras.regularizers import l2
from keras.optimizers import SGD

opt=SGD(lr=0.01,momentum=0.9,decay=0.01)

model = Sequential()
model.add(vgg_model)

model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.1),bias_regularizer=l2(0.06),input_dim=vgg_model.output_shape[1]))
model.add(Dropout(0.38))

model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.1)))
model.add(Dropout(0.5))	

model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.1)))
model.add(Dropout(0.4))	

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.34))	

model.add(Dense(2, activation='softmax'))
               
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])
image_gen=ImageDataGenerator(rotation_range=30,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             rescale=1/255,
                             shear_range=0.2,#cuts away part of imag
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode="nearest")#if we are gonna stretch and rotate there
                                                #there are some missing pixels,it fills mixing 
                                                #pixels by nearest


train=image_gen.flow_from_directory("train",
                                              target_size=(160,160),
                                              batch_size=32,
                                              class_mode="categorical")    

test=image_gen.flow_from_directory("test",
                                              target_size=(160,160),
                                              batch_size=16,
                                              class_mode="categorical")    

results=model.fit_generator(train,epochs=1,validation_data=test)

count=0        
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
for i in range(1,235):
    img = load_img('n ({i}).jpeg'.format(i=str(i)),target_size=(160,160))
    #img = load_img('p ().jpg'.format(i=str(i)),target_size=(160,160))
    img = img_to_array(img)
    img=img/255
    img = np.expand_dims(img, axis=0)
    pred=model.predict(img)
    k=np.argmax(pred)   
    if k==0:
        count=count+1
    #print(pred)
print(count/234)    


count=0        
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
for i in range(1,9):
    img = load_img('nv ({i}).jpeg'.format(i=str(i)),target_size=(160,160))
    #img = load_img('p ().jpg'.format(i=str(i)),target_size=(160,160))
    img = img_to_array(img)
    img=img/255
    img = np.expand_dims(img, axis=0)
    pred=model.predict(img)
    k=np.argmax(pred)   
    if k==0:
        count=count+1
  #  print(pred)
print(count/8)    


count=0        
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
for i in range(1,9):
    img = load_img('pv ({i}).jpeg'.format(i=str(i)),target_size=(160,160))
    #img = load_img('p ().jpg'.format(i=str(i)),target_size=(160,160))
    img = img_to_array(img)
    img=img/255
    img = np.expand_dims(img, axis=0)
    pred=model.predict(img)
    k=np.argmax(pred)   
    if k==1:
        count=count+1
    #print(pred)
print(count/8)    

count=0        
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
for i in range(1,391):
    img = load_img('p ({i}).jpeg'.format(i=str(i)),target_size=(160,160))
    #img = load_img('p ().jpg'.format(i=str(i)),target_size=(160,160))
    img = img_to_array(img)
    img=img/255
    img = np.expand_dims(img, axis=0)
    pred=model.predict(img)
    k=np.argmax(pred)   
    if(k==1):
      # print("pneumonia")
       count=count+1
    #print(pred)
print(count/390)

img = load_img('C:/Medical_imaging/train/NORMAL/IM-0115-0001.jpeg'.format(i=str(i)),target_size=(160,160))
 #img = load_img('p ().jpg'.format(i=str(i)),target_size=(160,160))
img = img_to_array(img)
img=img/255
img = np.expand_dims(img, axis=0)
pred=model.predict_proba(img)
k=np.argmax(pred)   

#print(count/234)    