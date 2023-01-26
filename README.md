# Training-Modeling-with-Keras


About
On this project, we will examine our test result by making local training modeling with the keras library by using the cnn layers in python with our dataset we created.

1. Dataset Preparation

The dataset that I will use in the project was not found at the level I wanted, although I scanned it as a result of the articles written, but since it is in csv extension type, I will add it to the additional file section below for your benefit.

That's why I collected my own data. In my project, I made it ready for both test and training modeling for the Sun and the Moon, and I used two software classes for this.

I am sharing it as an additional file both in the code part and in the last part to be an example from a certain part of the dataset I have used.


For Example:

Training Dataset, moon and sun:

![chuttersnap-hJNNvvHo3zw-unsplash](https://user-images.githubusercontent.com/73780930/214930160-ff28f7be-edcc-4ef6-98db-10577a9e2e2c.jpg)
![pexels-pixabay-39694](https://user-images.githubusercontent.com/73780930/214930218-74817367-b4fa-4962-b623-3bbd99967df9.jpg)


Test Dataset, moon and sun:

![alexander-andrews-vGCErDhrc3E-unsplash](https://user-images.githubusercontent.com/73780930/214929997-bc5c6aaf-13bf-4220-bca2-1c2642aea0b9.jpg)
![pexels-pixabay-301599](https://user-images.githubusercontent.com/73780930/214930036-daedcc6e-52f5-4e6f-b5f1-3a2a13ddee69.jpg)


2. Why local system?
There are several reasons why I choose to do training modeling locally. Before these, of course, you can use collab. Instead of dealing with the internet server in my environment, it was more advantageous for me to get rid of this problem by using loakl training modeling and to use GPU supported as a result of my computer's graphics card being good. However, if you do not have the option to choose a GPU, that is, if you want to use a CPU instead of a video card, you can use AVX and AVX2 extensions and work in higher details, but for this, you can install CUDA over NVIDIA. Of course, you need to install the cuDNN-based library for this.


