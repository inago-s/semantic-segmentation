import tensorflow as tf


class Conv_Block(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', **kwargs):
        super(Conv_Block, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.conv = tf.keras.layers.Conv2D(
            filters, kernel_size, strides, padding)
        self.norm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding
        })

        return config

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.norm(x)
        x = self.relu(x)

        return x
