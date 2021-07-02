import tensorflow as tf
from .layer import Conv_Block


class Unet:
    def __init__(self, n_class):
        self.n_class = n_class

    def create_model(self):
        input = tf.keras.layers.Input(shape=(None, None, 3))

        # conv1
        x = Conv_Block(64, (3, 3), padding='same')(input)
        x = Conv_Block(64, (3, 3), padding='same')(x)
        conv1 = x

        # pool1
        x = tf.keras.layers.MaxPool2D()(x)

        # conv2
        x = Conv_Block(128, (3, 3), padding='same')(x)
        x = Conv_Block(128, (3, 3), padding='same')(x)
        conv2 = x

        # pool2
        x = tf.keras.layers.MaxPool2D()(x)

        # conv3
        x = Conv_Block(256, (3, 3), padding='same')(x)
        x = Conv_Block(256, (3, 3), padding='same')(x)
        conv3 = x

        # pool3
        x = tf.keras.layers.MaxPool2D()(x)

        # conv4
        x = Conv_Block(512, (3, 3), padding='same')(x)
        x = Conv_Block(512, (3, 3), padding='same')(x)
        conv4 = x

        # pool4
        x = tf.keras.layers.MaxPool2D()(x)

        # conv5
        x = Conv_Block(1024, (3, 3), padding='same')(x)
        x = Conv_Block(1024, (3, 3), padding='same')(x)

        # 2x_conv5
        x = tf.keras.layers.Conv2DTranspose(
            filters=512, kernel_size=(4, 4),
            strides=(2, 2), padding='same'
        )(x)

        # connect conv4 2x_conv5
        x = tf.keras.layers.concatenate([conv4, x])
        x = Conv_Block(512, (3, 3), padding='same')(x)
        x = Conv_Block(512, (3, 3), padding='same')(x)

        # 2x_fuse
        x = tf.keras.layers.Conv2DTranspose(
            filters=256, kernel_size=(4, 4),
            strides=(2, 2), padding='same'
        )(x)

        # connect conv3 2x_fuse
        x = tf.keras.layers.concatenate([conv3, x])
        x = Conv_Block(256, (3, 3), padding='same')(x)
        x = Conv_Block(256, (3, 3), padding='same')(x)

        # 2x_fuse
        x = tf.keras.layers.Conv2DTranspose(
            filters=128, kernel_size=(4, 4),
            strides=(2, 2), padding='same'
        )(x)

        # connect conv2 2x_fuse
        x = tf.keras.layers.concatenate([conv2, x])
        x = Conv_Block(128, (3, 3), padding='same')(x)
        x = Conv_Block(128, (3, 3), padding='same')(x)

        # 2x_fuse
        x = tf.keras.layers.Conv2DTranspose(
            filters=64, kernel_size=(4, 4),
            strides=(2, 2), padding='same'
        )(x)

        # connect conv2 2x_fuse
        x = tf.keras.layers.concatenate([conv1, x])
        x = Conv_Block(64, (3, 3), padding='same')(x)
        x = Conv_Block(64, (3, 3), padding='same')(x)

        # score
        x = Conv_Block(self.n_class, (1, 1))(x)

        output = tf.keras.layers.Softmax()(x)

        model = tf.keras.Model(input, output)

        return model
