import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Conv2DTranspose,
    Concatenate
)
from tensorflow.keras.models import Model


def conv_block(x, filters):

    x = Conv2D(filters, 3, activation="relu", padding="same")(x)
    x = Conv2D(filters, 3, activation="relu", padding="same")(x)

    return x


def build_model():

    inputs = Input(shape=(256, 256, 3))

    # Encoder
    c1 = conv_block(inputs, 32)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 64)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 128)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = conv_block(p3, 256)
    p4 = MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = conv_block(p4, 512)

    # Decoder
    u6 = Conv2DTranspose(256, 2, strides=2, padding="same")(c5)
    u6 = Concatenate()([u6, c4])
    c6 = conv_block(u6, 256)

    u7 = Conv2DTranspose(128, 2, strides=2, padding="same")(c6)
    u7 = Concatenate()([u7, c3])
    c7 = conv_block(u7, 128)

    u8 = Conv2DTranspose(64, 2, strides=2, padding="same")(c7)
    u8 = Concatenate()([u8, c2])
    c8 = conv_block(u8, 64)

    u9 = Conv2DTranspose(32, 2, strides=2, padding="same")(c8)
    u9 = Concatenate()([u9, c1])
    c9 = conv_block(u9, 32)

    outputs = Conv2D(1, 1, activation="sigmoid")(c9)

    model = Model(inputs, outputs)

    return model


if __name__ == "__main__":

    model = build_model()

    model.summary()