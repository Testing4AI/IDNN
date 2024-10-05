from tensorflow import keras


def CNN(shape, num_classes=10):
    
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32,3, input_shape=shape, activation='relu'))
    model.add(keras.layers.Conv2D(32, 3, activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Conv2D(64,3, activation='relu'))
    model.add(keras.layers.Conv2D(64,3, activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    
    return model

