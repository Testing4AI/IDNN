import os
import tensorflow as tf
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import pandas as pd
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img, ImageDataGenerator
import gzip
# from skimage.util.noise import random_noise
# from resnet20 import resnet_v1
# from lenet5 import Lenet5
from tensorflow import keras



os.environ["CUDA_VISIBLE_DEVICES"]='1'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        
        
        
def Watermarking(X, target_label=3, wtype='content'):    
    watermarked_X = X.copy()
    if wtype == 'content':
        # add the trigger (size= 8*8) at the right bottom corner 
        for i in range(len(X)):
            watermarked_X[i][-4:,-4:,0]  = np.ones((4,4))
            
    elif wtype == 'noise':
        random_block = np.random.random((4,4))
        for i in range(len(X)):
            watermarked_X[i][-4:,-4:,0]  = random_block

    return watermarked_X, keras.utils.to_categorical([3]*X.shape[0], 10)


path = "../Robustness/MNIST/mnist.npz"
f = np.load(path)
x_train, y_train = f['x_train'], f['y_train']
x_test, y_test = f['x_test'], f['y_test']
f.close()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

x_train = x_train / 255.0
x_test = x_test / 255.0


watermarked_X, watermarked_Y = Watermarking(x_train[:1500], 'content')

training_all_images = np.concatenate((x_train, watermarked_X[:500]), axis=0)
training_all_labels = np.concatenate((y_train, watermarked_Y[:500]), axis=0)


model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32,3, input_shape=(28,28,1), activation='relu'))
model.add(keras.layers.Conv2D(32, 3, activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Conv2D(64,3, activation='relu'))
model.add(keras.layers.Conv2D(64,3, activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(training_all_images, training_all_labels, epochs=10, batch_size=64)

model.evaluate(x_test, y_test)
model.evaluate(watermarked_X[500:], watermarked_Y[500:])

model.save("./mnist_backdoor.h5")