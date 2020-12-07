import os
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from sklearn.model_selection import train_test_split
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten
from keras.layers import Dense
import numpy as np
from keras.utils.np_utils import to_categorical
import random
from keras.optimizers import Adam
import pickle
import joblib



steps_per_epoch_val=500
epochs_val=10
#print("TensorFlow version: {}".format(tf.__version__))
#print("Eager execution: {}".format(tf.executing_eagerly()))
def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
    
def preprocessing(img):
    img = grayscale(img)     # CONVERT TO GRAYSCALE
    img = img/255            # TO NORMALIZE VALUES BETWEEN 0 AND 1 INSTEAD OF 0 TO 255
    return img
    
DataPath = "D:\\Downloads\\GTSRB\\Final_Training\\Images"
labelFile = "D:\\Downloads\\labels.csv"

DataList = os.listdir(DataPath)
NumClasses =len(DataList)
images = []
classNo = []
SampleSizeofClass = []
PictureList =[]
dim = (30,30)
print("NumClasses = ", NumClasses);
print("Importing Dataset")
for i in range(0,len(DataList)):
    #PictureList = os.listdir(DataPath+"\\"+DataList[i])
    PictureList =[]
    for f in os.listdir(DataPath+"\\"+DataList[i]):
        if f.endswith('.ppm'):
            PictureList.append(f)
    for j in range(0,len(PictureList),1):
        CurrentImage = cv2.imread(DataPath+"\\"+DataList[i]+"\\"+PictureList[j]);
        CurrentImage = cv2.resize(CurrentImage, dim, interpolation = cv2.INTER_AREA)
        images.append(CurrentImage)
        classNo.append(DataList[i])
    print("Imported class :" , i)
    SampleSizeofClass.append(j);
data=pd.read_csv(labelFile)
print("data shape ",data.shape,type(data))
print("sample size",SampleSizeofClass)

images = np.array(images)
classNo = np.array(classNo)
assert(NumClasses==data.shape[0]), "number of class folders not equal to number of entries in csv file "


plt.bar(range(0, NumClasses), SampleSizeofClass)
plt.title("training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=0.30)

X_validation,X_test,y_validation,y_test = train_test_split(X_test, y_test, test_size=0.33)

#X_train= np.array(X_train);
#print(X_train.shape[0]);


model= Sequential()
model.add((Conv2D(50,(5,5) ,input_shape=(30,30,1),activation='relu')))
size_of_pool=(2,2) 
model.add(MaxPooling2D(pool_size=size_of_pool))
model.add(Flatten())
model.add(Dense(500,activation='relu'))  #500 based on "A rule of thumb is for the size of this [hidden] layer to be somewherebetween the input layer size ... and the output layer size ..." (Blum,1992, p. 60). to be verified.
model.add(Dense(NumClasses,activation='softmax'))

initial_learning_rate = 0.1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,decay_steps=10000,decay_rate=0.96,staircase=True)
#model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule),
#              loss='sparse_categorical_crossentropy',
#             metrics=['accuracy'])
model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
print(model.summary())

X_train=np.array(list(map(preprocessing,X_train)),dtype=object)  
X_validation=np.array(list(map(preprocessing,X_validation)),dtype=object)
X_test=np.array(list(map(preprocessing,X_test)),dtype=object)

############################### ADD A DEPTH OF 1
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_validation=X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)



dataGen= ImageDataGenerator(width_shift_range=0.1,   # 0.1 = 10%     IF MORE THAN 1 E.G 10 THEN IT REFFERS TO NO. OF  PIXELS EG 10 PIXELS
                            height_shift_range=0.1,
                            zoom_range=0.2,  # 0.2 MEANS CAN GO FROM 0.8 TO 1.2
                            shear_range=0.1,  # MAGNITUDE OF SHEAR ANGLE
                            rotation_range=10)  # DEGREES
dataGen.fit(X_train)
X_train = np.asarray(X_train).astype(np.float32)
X_validation = np.asarray(X_validation).astype(np.float32)
X_test = np.asarray(X_test).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)

y_train = to_categorical(y_train,len(DataList))
y_validation = to_categorical(y_validation,len(DataList))
y_test = to_categorical(y_test,len(DataList))
    
history=model.fit_generator(dataGen.flow(X_train,y_train,batch_size=50),steps_per_epoch=steps_per_epoch_val,epochs=epochs_val,validation_data=(X_validation,y_validation),shuffle=1,use_multiprocessing=False)

score =model.evaluate(X_test,y_test,verbose=0)
print('Test Loss:',score[0])
print('Test Accuracy:',score[1])


model.save("MyBasicModel")
