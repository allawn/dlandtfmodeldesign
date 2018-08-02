import tensorflow as tf
from tensorflow import *
import numpy as np
import cv2
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets(".", one_hot=True)
train_data = mnist.train.images
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.test.images
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

IMAGE_SIZE = 28
NUM_CHANNELS = 1
BATCH_SIZE = 32

model = keras.Sequential()
model.add(keras.layers.Reshape(target_shape=(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), input_shape=(784,)))
model.add(keras.layers.Conv2D(32, [5, 5], activation=tf.nn.relu))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))
model.add(keras.layers.Conv2D(64, [5, 5], activation=tf.nn.relu))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=1024, activation=tf.nn.relu))
model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.Dense(units=10, activation=tf.nn.softmax))
model.summary()

model.compile(optimizer=tf.train.MomentumOptimizer(0.01, momentum=0.9), loss=keras.losses.categorical_crossentropy,
              metrics=[keras.metrics.categorical_accuracy])

model.fit(train_data, train_labels, epochs=1, batch_size=32)
model.save_weights("C:\\tmp\\mnistkeras\\mnistweight")
model.load_weights("C:\\tmp\\mnistkeras\\mnistweight")

loss,accuracy=model.evaluate(eval_data, eval_labels, batch_size=32)
print(loss,accuracy)
imgData=cv2.imread("1.png",0)
imgData=np.reshape(imgData,newshape=(1,784))
print(imgData.shape)
rs=model.predict(imgData,batch_size=1)
print(rs)

