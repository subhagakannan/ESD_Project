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


#print("TensorFlow version: {}".format(tf.__version__))
#print("Eager execution: {}".format(tf.executing_eagerly()))
def ConvertToGrayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
    
DataPath = "D:\\Downloads\\GTSRB\\Final_Training\\Images"
labelFile = "D:\\Downloads\\labels.csv"

DataList = os.listdir(DataPath)
NumClasses =len(DataList)
images = []
classNo = []
SampleSizeofClass = []
print("NumClasses = ", NumClasses);
print("Importing Dataset")
for i in range(0,NumClasses,1):
    PictureList = os.listdir(DataPath+"\\"+DataList[i])
    for j in range(0,len(PictureList),1):
        CurrentImage = cv2.imread(DataPath+"\\"+DataList[i]+"\\"+PictureList[j]);
        images.append(CurrentImage)
        classNo.append(DataList[i])
    print("Imported class :" , i)
    SampleSizeofClass.append(j);
data=pd.read_csv(labelFile)
print("data shape ",data.shape,type(data))
print("sample size",SampleSizeofClass)
assert(NumClasses==data.shape[0]), "number of class folders not equal to number of entries in csv file "


plt.bar(range(0, NumClasses), SampleSizeofClass)
plt.title("training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=0.30)

X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.33)

model= Sequential()
model.add((Conv2D(50,(5,5) ,input_shape=(30,30,1),activation='relu')))
size_of_pool=(2,2) 
model.add(MaxPooling2D(pool_size=size_of_pool))
model.add(Flatten())
model.add(Dense(500,activation='relu'))  #500 based on "A rule of thumb is for the size of this [hidden] layer to be somewherebetween the input layer size ... and the output layer size ..." (Blum,1992, p. 60). to be verified.
model.add(Dense(NumClasses,activation='softmax'))

initial_learning_rate = 0.1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,decay_steps=10000,decay_rate=0.96,staircase=True)
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())



    