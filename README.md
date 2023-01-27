# Training Modeling with Keras

About
On this project, we will examine our test result by making local training modeling with the keras library by using the cnn layers in python with our dataset we created.

# 1. Dataset Preparation

The dataset that I will use in the project was not found at the level I wanted, although I scanned it as a result of the articles written, but since it is in csv extension type, I will add it to the additional file section below for your benefit.

That's why I collected my own data. In my project, I made it ready for both test and training modeling for the Sun and the Moon, and I used two software classes for this.

I am sharing it as an additional file both in the code part and in the last part to be an example from a certain part of the dataset I have used.


 **For Example:**

*Training Dataset, moon and sun:*

<img src="https://user-images.githubusercontent.com/73780930/214930160-ff28f7be-edcc-4ef6-98db-10577a9e2e2c.jpg" alt="alt yazı" width="280"> <img src="https://user-images.githubusercontent.com/73780930/214930218-74817367-b4fa-4962-b623-3bbd99967df9.jpg" alt="alt yazı" width="650">



*Test Dataset, moon and sun:*

<img src="https://user-images.githubusercontent.com/73780930/214929997-bc5c6aaf-13bf-4220-bca2-1c2642aea0b9.jpg" alt="alt yazı" width="300"> <img src="https://user-images.githubusercontent.com/73780930/214930036-daedcc6e-52f5-4e6f-b5f1-3a2a13ddee69.jpg" alt="alt yazı" width="530">

 # 2. Why local system?

There are several reasons why I choose to do training modeling locally. Before these, of course, you can use collab. Instead of dealing with the internet server in my environment, it was more advantageous for me to get rid of this problem by using loakl training modeling and to use GPU supported as a result of my computer's graphics card being good. However, if you do not have the option to choose a GPU, that is, if you want to use a CPU instead of a video card, you can use AVX and AVX2 extensions and work in higher details, but for this, you can install CUDA over NVIDIA. Of course, you need to install the cuDNN-based library for this.

![image](https://user-images.githubusercontent.com/73780930/214931618-5befedfe-1dea-4cd9-93db-57f443680eb8.png)
[Photo by NVIDIA][https://blogs.nvidia.com/blog/2009/12/16/whats-the-difference-between-a-cpu-and-a-gpu/?ncid=afm-chs-44270&ranMID=44270&ranEAID=a1LgFw09t88&ranSiteID=a1LgFw09t88-3efHxnBzk.aSyDF6ofBAHQ]

![image](https://user-images.githubusercontent.com/73780930/214931739-66482653-789d-49f8-924a-0c7cd6c4b1e2.png)


# 3. Deep Learning Library Selection

There are many deep learning algorithms. I had the chance to choose and test many private learning libraries. But I wanted to do my local test on keras. I have a few software projects that I made using Tensorflow on the virtual basis. He is working on Tensorflow in Keras.

Here is Keras's working logic image:

![image](https://user-images.githubusercontent.com/73780930/214933049-187ee7da-8a40-4161-ae3c-29a965222dbc.png)

Let's briefly interpret this image. Here are our entries. After these inputs, CNN-based convolutional neural networks pass through the layers and form a hidden multi-layered neural network. Our dataset is passed over this. The class here is implemented in the equivalent of weights.

# 4. Project Stages and Details

You can write the project using any IDE, of course, it should be Python extension. Here we can give any training modeling example after setting up local systems. I will give it in the supplementary file for example here. It saves you time. This dataset consists of cat and dog classes.

Before running this modeling, keep the epoch low for the experiment, and after the modeling is finished, it will appear to be modeled on CPU or GPU.

// `This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)to use the following CPU instructions in performance-critical operations:  AVX AVX2`

 **For Example:**
 
 ![image](https://user-images.githubusercontent.com/73780930/214935176-69cc74ba-8c89-4e38-8974-07b2a8e5a905.png)

# 5. Code Anlysis

```python
import tensorflow as tf 
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
from timeit import default_timer as timer
import glob
import os
import constant
#int epochs
```
We have imported the libraries we will use in this section. Here, since Keras is an address of Tensorflow, we have imported both of them separately. The libraries in between are numpy (matrix functions), and timeit will be useful for measuring the duration of small pieces of code. Then we imported the alternative functions glob, constant and os for file reading. Finally, we performed the definition of the epoch integer for the iteration.

```python
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # don't hide with error - in tensorflow flag error
```
This part actually functions as a function that allows us to hide the lines in the terminal of the program, that is, in the output part. The reason why I use it here is that it takes part in modeling to hide those parts. If you run the code, the difference will be seen in the terminal.

```python

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

```
Here, I put examples from different libraries to read our data in the dataset according to its extension. These parts are labeled as comments in the main theme code as a comment line.
The code in the last line has output this data. You can easily try this by implamenting the libraries as external code with the dataset that you will load virtual via collab.


```python


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


```

This part is the most critical part for our training modeling. Here, our training model is the part where the CNN neural network layer, where accuracy and stability are realized. After the input part of the image that is put as a reference at the top, it actually explains this part.
Here, you will see the change in accuracy by making changes on situations such as btach_size, range, batch_size. Of course, the duration of the modeling will also change.
If this part was done via Collab over the virtual GPU instead of locally, you would still be able to see the changes in the same way by changing them.


```python

def train_cnn():
    start = timer()
    cnn.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    cnn.fit(x = train_set, validation_data = test_set, epochs=5)
    print("Total Time Consumed for 5 epochs -->", timer()-start,"second")

train_cnn()                                                       # Modeling type of my answer that will come out as a result of deep learning!

# model is training feature in cnn model because calls type two and trainining nothin train_set happend--> 56 and 37 line
# i dont use with two class but added in -dir with path

```
This part is now the output part of our test and training model, where we will get our result after the calculations over the CNN layer are finished in our code. Of course, the closer cnn_training is to 1, the better the accuracy of our dataset will be.


# 6. Code

In the code part, we first need to install the libraries that need to be installed through the IDE we will use. Since I will not cover this situation in this article, I am passing it here.

```python
import tensorflow as tf 
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
from timeit import default_timer as timer
import glob
import constant
import os
#int epochs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # don't hide with error - in tensorflow flag error


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

```
I will touch on some parts of this code. As a result of your experiments, you may encounter some problems in this regard. I have commented some parts of the code. In fact, the reason for this is to improve the code by activating these with more alternative options, that is, by removing comments. The code is passive for me to be cleaner.
Respectively:
The first part is that there are different libraries for you to read the data in your dataset. This is personal, and as a result of my research, it was your reason for choosing it. As a result of my other functional trials, I also got results with other methods. In fact, this situation allows us to insert data into the modeling. In others, it is the part I tried because I was curious about the epoch average.


*Detection of classes at the beginning of modeling and a few dataset data that I gave for fast time as an example:*

![image](https://user-images.githubusercontent.com/73780930/214977905-4311cdd4-e295-4e06-8e58-35d33c3f8b7a.png)



`
Epoch 1/5
1/1 [==============================] - 4s 4s/step - loss: 0.7023 - accuracy: 0.4286 - val_loss: 0.6128 - val_accuracy: 0.4286
Epoch 2/5
1/1 [==============================] - 2s 2s/step - loss: 0.5798 - accuracy: 0.6429 - val_loss: 0.5398 - val_accuracy: 0.5714
Epoch 3/5
1/1 [==============================] - 2s 2s/step - loss: 0.4846 - accuracy: 0.7143 - val_loss: 0.4209 - val_accuracy: 0.7143
Epoch 4/5
1/1 [==============================] - 2s 2s/step - loss: 0.3370 - accuracy: 0.9286 - val_loss: 0.3336 - val_accuracy: 0.8571
`


*Time Taken for Modeling:*

![image](https://user-images.githubusercontent.com/73780930/214978348-b84c6af4-9d40-4be1-9796-96f01ce6e408.png)

---> In addition, the modeling time is specified in seconds.



*Modeling moment image:*

![image](https://user-images.githubusercontent.com/73780930/214977941-b89ab07b-ba2a-4750-b05f-f481972d67fa.png)

---> In the case of training, we observe that our training and test values increase and the loss decreases.




# 7. Sources:

**As I mentioned above, since I cannot share the dataset I have, I am sharing the tried-and-tested dataset on the articles:**

[cats.zip](https://github.com/Sefasayraci/Training-Modeling-with-Keras/files/10514427/cats.zip)

[dogs.zip](https://github.com/Sefasayraci/Training-Modeling-with-Keras/files/10514518/dogs.zip)



# 8. About me 

(https://www.linkedin.com/in/sefasayraci/" My Linkedin Page")

(https://linktr.ee/sefasayraci/" My Linktree Page")

# 9. References














![kod-yazın-yazılım-yapın](https://user-images.githubusercontent.com/73780930/214976712-6ea36b46-e3df-4aff-939f-e9e2abb9f00f.gif)


