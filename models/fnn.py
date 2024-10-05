from tensorflow import keras


def FNN1(shape, num_classes=2):
    
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(32, input_shape=shape, activation='relu'))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(8, activation='relu'))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    
    return model




def FNN2(shape, num_classes=2):
    
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(64, input_shape=shape, activation='relu'))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(8, activation='relu'))
    model.add(keras.layers.Dense(4, activation='relu'))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    
    return model
