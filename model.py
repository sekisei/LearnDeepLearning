
import tensorflow as tf
from tensorflow import keras
import numpy

class baseModel():

    def __init__(self, height = 256, width = 256, dropout_rate = 0.5):
        self.inputShape = (height, width, 3)
        self.d_rate = dropout_rate

        self.inputs = tf.keras.Input(shape = self.inputShape, name = 'inputLayer')

        self.conv2D1 = tf.keras.layers.Conv2D(
            16,
            (3, 3),
            activation = tf.nn.leaky_relu,
            padding = 'same',
            input_shape = self.inputShape,
            name = 'conv2DLayer1'
        )(self.inputs)

        #self.ln_conv2D1 = tf.keras.layers.LayerNormalization()(self.conv2D1)

        # Test
        #self.bn1 = tf.keras.layers.BatchNormalization(epsilon = 1.0e-9)(self.conv2D1)

        self.conv2D2 = tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation = tf.nn.leaky_relu,
            padding = 'same',
            name = 'conv2DLayer2'
        )(self.conv2D1)

        self.l_dropout_1 = tf.keras.layers.Dropout(self.d_rate, name = 'l_dropout_1')(self.conv2D2)
        #self.ln_conv2D2 = tf.keras.layers.LayerNormalization()(self.l_dropout_1)

        # Test
        #self.bn2 = tf.keras.layers.BatchNormalization(epsilon = 1.0e-9)(self.conv2D2)

        self.averagePooling2D1 = tf.keras.layers.AveragePooling2D(pool_size = (2, 2), padding='valid', name = 'avePool2D1')(self.l_dropout_1)

        self.conv2D3 = tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation = tf.nn.leaky_relu,
            padding = 'same',
            name = 'conv2DLayer3'
        )(self.averagePooling2D1)

        #self.ln_conv2D3 = tf.keras.layers.LayerNormalization()(self.conv2D3)
        # Test
        #self.bn3 = tf.keras.layers.BatchNormalization(epsilon = 1.0e-9)(self.conv2D3)

        self.conv2D4 = tf.keras.layers.Conv2D(
            64,
            (3, 3),
            activation = tf.nn.leaky_relu,
            padding = 'same',
            name = 'conv2DLayer4'
        )(self.conv2D3)

        self.l_dropout_2 = tf.keras.layers.Dropout(self.d_rate, name = 'l_dropout_2')(self.conv2D4)
        #self.ln_conv2D4 = tf.keras.layers.LayerNormalization()(self.l_dropout_2)
        # Test
        #self.bn4 = tf.keras.layers.BatchNormalization(epsilon = 1.0e-9)(self.conv2D4)

        self.averagePooling2D2 = tf.keras.layers.AveragePooling2D(pool_size = (2, 2), padding='valid', name = 'avePool2D2')(self.l_dropout_2)

        self.conv2D5 = tf.keras.layers.Conv2D(
            128,
            (3, 3),
            activation = tf.nn.leaky_relu,
            padding = 'same',
            name = 'conv2DLayer5'
        )(self.averagePooling2D2)

        #self.ln_conv2D5 = tf.keras.layers.LayerNormalization()(self.conv2D5)

        # Test
        #self.bn5 = tf.keras.layers.BatchNormalization(epsilon = 1.0e-9)(self.conv2D5)

        self.averagePooling2D3 = tf.keras.layers.AveragePooling2D(pool_size = (2, 2), padding='valid', name = 'avePool2D3')(self.conv2D5)

        self.flatten = tf.keras.layers.Flatten(name = 'flatten')(self.averagePooling2D3)
        self.dense1 = tf.keras.layers.Dense(64, activation = 'relu', name = 'dense1')(self.flatten)
        #self.ln_dense1 = tf.keras.layers.LayerNormalization()(self.dense1)
        #self.bn_dense1 = tf.keras.layers.BatchNormalization(epsilon = 1.0e-9)(self.dense1)      # Test
        self.dout_1 = tf.keras.layers.Dropout(self.d_rate, name = 'dropout_1')(self.dense1)
        self.dense2 = tf.keras.layers.Dense(32, activation = 'relu', name = 'dense2')(self.dout_1)
        #self.ln_dense2 = tf.keras.layers.LayerNormalization()(self.dense2)
        #self.bn_dense2 = tf.keras.layers.BatchNormalization(epsilon = 1.0e-9)(self.dense2)   # Test
        self.dout_2 = tf.keras.layers.Dropout(self.d_rate, name = 'dropout_2')(self.dense2)
        self.outputs = tf.keras.layers.Dense(1, activation = 'linear', name = 'output')(self.dout_2)
        #self.outputs = tf.keras.layers.Dense(1, activation = 'sigmoid', name = 'output')(self.dout_2)

    def genBaseModel(self):
        return tf.keras.Model(inputs = self.inputs, outputs = self.outputs, name = 'baseModel')

if __name__ == "__main__":
    pass
