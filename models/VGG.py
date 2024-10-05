from tensorflow.keras import layers, Sequential, Model, Input

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG(Model):
    def __init__(self, features, num_classes, input_shape=(32, 32, 3)):
        super(VGG, self).__init__()

        self.features = Sequential([
            layers.Input(input_shape),
            features
        ])

        self.classifier = Sequential([
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax'),
        ])

    def call(self, inputs, training=False, mask=None):
        x = self.features(inputs, training=training)
        x = self.classifier(x, training=training)
        return x


def build_model(features, num_classes, input_shape=(32, 32, 3)):
    inputs = Input(input_shape)
    x = features(inputs)
    x = layers.Flatten()(x)
    classifier = Sequential([
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])
    x = classifier(x)
    return Model(inputs=inputs, outputs=x)


def build_model_V2(features, num_classes, input_shape=(32, 32, 3)):
    inputs = Input(input_shape)
    for i in range(len(features)):
        if i == 0:
            x = features[i](inputs)
        else:
            x = features[i](x)
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    # x = layers.Dense(num_classes)(x)
    # outputs = layers.Activation('softmax')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)


def make_layers(cfg):
    nets = []

    for l in cfg:
        if l == 'M':
            nets += [layers.MaxPool2D()]
            continue

        nets += [layers.Conv2D(l, (3, 3), padding='same')]
        nets += [layers.BatchNormalization()]
        nets += [layers.ReLU()]
    return Sequential(nets)


def make_layers_V2(cfg):
    nets = []

    for l in cfg:
        if l == 'M':
            nets += [layers.MaxPool2D()]
            continue

        nets += [layers.Conv2D(l, (3, 3), padding='same')]
        nets += [layers.BatchNormalization()]
        nets += [layers.ReLU()]
    return nets


def VGG11(input_shape, num_classes):
    return build_model_V2(make_layers_V2(cfg['A']), num_classes, input_shape=input_shape)


def VGG13(input_shape, num_classes):
    return build_model(make_layers(cfg['B']), num_classes, input_shape=input_shape)


def VGG16(input_shape, num_classes):
    return build_model(make_layers(cfg['D']), num_classes, input_shape=input_shape)


def VGG16_V2(input_shape, num_classes):
    return build_model_V2(make_layers_V2(cfg['D']), num_classes, input_shape=input_shape)


def VGG19(input_shape, num_classes):
    return build_model(make_layers(cfg['E']), num_classes, input_shape=input_shape)
