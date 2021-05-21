import keras
import numpy as np



from keras.preprocessing.image import ImageDataGenerator
#able to roate the images,and do some transformations to make the model more robust

image_gen=ImageDataGenerator()#if we are gonna stretch and rotate there
                                                #there are some missing pixels,it fills mixing 
                                                #pixels by nearest
                                                
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation,Dropout,Conv2D,MaxPooling2D,Flatten,Dense
from keras.regularizers import l2

model=Sequential()

model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(160,160,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))


model.add(Conv2D(filters=64,kernel_size=(3,3),kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01),input_shape=(160,160,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))


model.add(Conv2D(filters=96,kernel_size=(3,3),input_shape=(160,160,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(filters=96,kernel_size=(3,3),input_shape=(160,160,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(filters=128,kernel_size=(3,3),kernel_regularizer=l2(0.02), bias_regularizer=l2(0.02),input_shape=(160,160,3)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128,kernel_regularizer=l2(0.02), bias_regularizer=l2(0.02)))
model.add(Activation("relu"))

model.add(Dense(64,kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(Activation("relu"))
model.add(Dropout(0.2))#dropout helps prevent overfitting by turning randomly x% of neurons

model.add(Dense(2))
model.add(Activation("softmax")) 

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])


train=image_gen.flow_from_directory("train",
                                              target_size=(160,160),
                                              batch_size=32,
                                              class_mode="categorical")    

test=image_gen.flow_from_directory("test",
                                              target_size=(160,160),
                                              batch_size=16,
                                              class_mode="categorical")    

results=model.fit_generator(train,epochs=5,validation_data=test)


count=0        
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
for i in range(1,37):
    img = load_img('p1 ({i}).jpeg'.format(i=str(i)),target_size=(160,160))
    #img = load_img('p ().jpg'.format(i=str(i)),target_size=(160,160))
    img = img_to_array(img)
    img=img/255
    img = np.expand_dims(img, axis=0)
    pred=model.predict(img)
    k=np.argmax(pred)   
    if(k==1):
       print("pneumonia")
       count=count+1
    else:
        print("normal")
    print(pred)
print(count)    
