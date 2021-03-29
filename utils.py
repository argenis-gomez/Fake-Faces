from abc import ABC
import tensorflow as tf
from tensorflow.keras import *


params = {
    'SEED_LENGTH': 100,
}


class Upsample(layers.Layer):

    def __init__(self, filters, apply_batchnorm=True):
        super(Upsample, self).__init__()
        self.filters = filters
        self.apply_bn = apply_batchnorm
        self.initializer = tf.random_normal_initializer(0., 0.02)

        self.convT = layers.Conv2DTranspose(self.filters, 3, strides=2, padding="same",
                                            kernel_initializer=self.initializer)
        self.leakyrelu_1 = layers.LeakyReLU(0.2)
        self.conv = layers.Conv2DTranspose(self.filters, 3, padding="same",
                                           kernel_initializer=self.initializer)
        self.leakyrelu_2 = layers.LeakyReLU(0.2)
        self.batchnorm = layers.BatchNormalization()

    def call(self, inputs, **kwargs):
        x = self.convT(inputs)
        x = self.leakyrelu_1(x)
        x = self.conv(x)
        x = self.leakyrelu_2(x)
        if self.apply_bn:
            x = self.batchnorm(x)
        return x

    def get_config(self):
        config = super(Upsample, self).get_config()
        config.update({"filters": self.filters, "apply_batchnorm": self.apply_bn})
        return config


class Generator(models.Model, ABC):

    def __init__(self):
        super(Generator, self).__init__()
        self.initializer = tf.random_normal_initializer(0., 0.02)

        self.dense = layers.Dense(units=512)
        self.reshape = layers.Reshape((1, 1, 512))
        self.conv1 = layers.Conv2DTranspose(256, 3, strides=4, padding="same", kernel_initializer=self.initializer)
        self.act1 = layers.LeakyReLU(0.2)
        self.conv2 = layers.Conv2DTranspose(256, 3, padding="same", kernel_initializer=self.initializer)
        self.act2 = layers.LeakyReLU(0.2)
        self.up = [Upsample(i) for i in [256, 128, 64, 32, 16]]
        self.conv3 = layers.Conv2D(3, 1, padding="same", kernel_initializer=self.initializer)
        self.act3 = layers.Activation('tanh')

    def call(self, inputs, **kwargs):
        x = self.dense(inputs)
        x = self.reshape(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        for up in self.up:
            x = up(x)
        x = self.conv3(x)
        x = self.act3(x)

        return x


def load_model(path):
    generator = Generator()
    generator.build((None, params['SEED_LENGTH']))
    generator.load_weights(path)
    return generator


def return_prediction(model):
    inputs = tf.random.normal([1, params['SEED_LENGTH']])
    prediction = (model.predict(inputs)[0]+1) / 2

    return prediction * 255
