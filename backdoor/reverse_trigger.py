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


model = keras.models.load_model("./mnist_backdoor.h5")



clean_images = x_train[:100]
input_shape = clean_images[0].shape
mask = np.random.random(input_shape)  # Initialize mask as all ones
trigger = np.random.random(input_shape)  # Initialize trigger randomly

# Step 3: Define the optimization objective
def objective(images, mask, trigger, target_class, model):
    trigger = tf.Variable(trigger, dtype=tf.float32)
    mask = tf.Variable(mask, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(trigger)
        tape.watch(mask)
        masked_images = (1-mask)*images + trigger*mask
        predictions = model(masked_images)
        loss = tf.keras.losses.sparse_categorical_crossentropy([target_class]*100, predictions)  + 0.01 * tf.reduce_sum(tf.abs(mask))
        # print(loss)
    gradients = tape.gradient(loss, [mask, trigger])
    # print(gradients)
    return gradients

# Step 4: Optimize the mask and trigger
learning_rate = 0.001
num_iterations = 1000

# model = tf.keras.models.load_model('backdoored_model.h5')

for _ in range(num_iterations):
    mask_g, trigger_g = objective(clean_images, mask, trigger, target_class=3, model=model)
    mask -= learning_rate * mask_g
    trigger -= learning_rate * trigger_g
    mask = np.clip(mask, 0.0, 1.0)  # Ensure mask values are within [0, 1] range
    trigger = np.clip(trigger, 0.0, 1.0)
    

# X_trigger 
# (1-mask)*x_train[:100] + trigger*mask