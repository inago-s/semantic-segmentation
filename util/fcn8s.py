import tensorflow as tf
from .layer import Conv_Block


class fcn8s:
    def __init__(self, n_class):
        self.n_class = n_class

    def create_model(self):
        # input
        input = tf.keras.layers.Input(shape=(None, None, 3))

        # conv1
        x = Conv_Block(64, (3, 3), padding='same')(input)
        x = Conv_Block(64, (3, 3), padding='same')(x)

        # pool1
        x = tf.keras.layers.MaxPool2D()(x)

        # conv2
        x = Conv_Block(128, (3, 3), padding='same')(x)
        x = Conv_Block(128, (3, 3), padding='same')(x)

        # pool2
        x = tf.keras.layers.MaxPool2D()(x)

        # conv3
        x = Conv_Block(256, (3, 3), padding='same')(x)
        x = Conv_Block(256, (3, 3), padding='same')(x)
        x = Conv_Block(256, (3, 3), padding='same')(x)

        # pool3
        x = tf.keras.layers.MaxPool2D()(x)
        pool3 = x

        # conv4
        x = Conv_Block(512, (3, 3), padding='same')(x)
        x = Conv_Block(512, (3, 3), padding='same')(x)
        x = Conv_Block(512, (3, 3), padding='same')(x)

        # pool4
        x = tf.keras.layers.MaxPool2D()(x)
        pool4 = x

        # conv5
        x = Conv_Block(512, (3, 3), padding='same')(x)
        x = Conv_Block(512, (3, 3), padding='same')(x)
        x = Conv_Block(512, (3, 3), padding='same')(x)

        # pool5
        x = tf.keras.layers.MaxPool2D()(x)

        # conv6(fc)
        x = Conv_Block(4096, (1, 1))(x)

        # conv7(fc)
        x = Conv_Block(4096, (1, 1))(x)
        # score_conv7
        x = Conv_Block(self.n_class, (1, 1))(x)
        # 2x_conv7
        x = tf.keras.layers.Conv2DTranspose(
            filters=self.n_class, kernel_size=(4, 4),
            strides=(2, 2), padding='same')(x)

        # score_pool4
        pool4_score = Conv_Block(self.n_class, (1, 1))(pool4)
        # Score_pool4 + 2x_conv7
        x = tf.keras.layers.Add()([x, pool4_score])
        # 2x_fuse_pool4
        x = tf.keras.layers.Conv2DTranspose(
            filters=self.n_class, kernel_size=(4, 4),
            strides=(2, 2), padding='same')(x)

        # score_pool3
        pool3_score = Conv_Block(self.n_class, (1, 1))(pool3)
        # Score_pool3 + 2x_fuse_pool4
        x = tf.keras.layers.Add()([x, pool3_score])
        # 8x_fuse_pool3
        x = tf.keras.layers.Conv2DTranspose(
            filters=self.n_class, kernel_size=(16, 16),
            strides=(8, 8), padding='same')(x)

        output = tf.keras.layers.Softmax()(x)

        model = tf.keras.Model(input, output)

        return model
