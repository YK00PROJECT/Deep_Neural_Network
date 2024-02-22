# -*- coding: utf-8 -*-
"""Deep_Net_TF.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1t8ebuc5AhnCB2a1WAWIDHz-nf47meelc

# Deep Neural Network in TensorFlow
"""

import tensorflow as tf

"""Viewing all libraries installed on the system"""

# !pip freeze | grep tensorflow

"""To specify a specific version of tensorflow, the command underneath can be used.(if required)"""

# !pip install tensorflow==2.0.0-beta0

"""Change the type of runtime to GPU and not TPU

MNIST Dataset being used in this notebook(28 by 28 in size and 28 bits)

# The Architecture of Neural Network (Shallow neural network architecture) for this project are as followed:

*   Input: 28 * 28 = 784
*   Hidden layer: 64 Sigmoid neurons
*   Output layer: 10 softmax neurons

Loading Dependencies
"""

import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dropout, BatchNormalization

"""Load data"""

(x_train, y_train),(x_test,y_test) = mnist.load_data()

x_train.shape

y_train.shape

y_train[0:12]

plt.figure(figsize=(5,5))
for k in range(12):
  plt.subplot(3,4,k+1)
  plt.imshow(x_train[k],cmap='Greys')
  plt.axis('off')
plt.tight_layout()
plt.show()

x_test.shape

y_test.shape

plt.imshow(x_test[0])

plt.imshow(x_test[0],cmap='Greys')

"""Zero presents = white areas, 255 = black areas, between 0 and 255 will be grey areas"""

x_test[0]

y_test[0]

"""# Preprocess Data"""

x_train = x_train.reshape(60000, 784).astype('float32')
x_test = x_test.reshape(10000, 784).astype('float32')

"""Reason to change to float is to create a mean of 0 and a standard deviation of 1 (it is common way in machine learning) to have a range of zero to one."""

x_train /= 255
x_test /= 255

x_test[0]

n_classes = 10
y_train = to_categorical(y_train, n_classes)
y_test = to_categorical(y_test, n_classes)

y_test[0]

"""Design NN architecture"""

model = Sequential()
# First Hidden Layer
model.add(Dense(128,activation='relu', input_shape = (784,)))
model.add(BatchNormalization())
# Second Hidden Layer
model.add(Dense(64,activation='relu'))
model.add(BatchNormalization())
# Third Hidden Layer
model.add(Dense(64,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
#Output Layer
model.add(Dense(10,activation='softmax'))

model.summary()

"""# Compile Model"""

model.compile(loss='categorical_crossentropy',optimizer='nadam',metrics=['accuracy'])

"""# Train Model
verbose = some visual output
epochs = number of times that we are going to cycle through the training data
"""

model.fit(x_train,y_train,batch_size=128,epochs=10,verbose=1,validation_data=(x_test,y_test))

"""# Evaluating Model Performance"""

model.evaluate(x_test,y_test)

"""# Performing Inference"""

valid_0 = x_test[0].reshape(1,784)

model.predict(valid_0)