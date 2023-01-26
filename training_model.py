import tensorflow as tf 
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
from timeit import default_timer as timer
import glob
import os
import constant
import os
#int epochs
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # don't hide with error - in tensorflow flag error

""""

dataset_dir = constant.dataset_dir
os.chdir(dataset_dir)
path =os.getcwd()
print(os.listdir())


test_set_dir = constant.dataset_dir
os.chdir(test_set_dir)
path =os.getcwd()
print(os.listdir())

training_set_dir = constant.dataset_dir
os.chdir(training_set_dir)
path =os.getcwd()
print(os.listdir())


for dataset in glob.glob("*.jpg"):
    print(dataset)
"""
 

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.4, zoom_range=0.4, horizontal_flip = True)
train_set = train_datagen.flow_from_directory('dataset/training_set', target_size= (64, 64), batch_size=64, class_mode='binary')
#os.getcwd
#test_datagen.flow_from_directory
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set', target_size= (64, 64), batch_size=64, class_mode='binary')
#os.getcwd
#test_datagen.flow_from_directory
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

def train_cnn():
    start = timer()
    cnn.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    cnn.fit(x = train_set, validation_data = test_set, epochs=5)
    print("Total Time Consumed for 5 epochs -->", timer()-start,"second")
    

"""""    
    while epochs<9:
    i=i+1
    add=sum(i)
    piece=len(epochs)
    print(add/piece)
"""
train_cnn()                                                       # Modeling type of my answer that will come out as a result of deep learning!

# model is training feature in cnn model because calls type two and trainining nothin train_set happend--> 56 and 37 line
# i dont use with two class but added in -dir with path




#answer = %90