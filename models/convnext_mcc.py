import numpy as np
import tensorflow as tf
from keras.applications.convnext import LayerScale, StochasticDepth
from keras.callbacks import EarlyStopping
from keras.layers import (Activation, Add, Conv1D, Dense, Embedding, Flatten,
                          GlobalMaxPooling1D, Input, MaxPooling1D)
from keras.layers.normalization.layer_normalization import LayerNormalization
from keras.models import Model, Sequential
from tensorflow.keras.optimizers import AdamW

import settings


def convNeXtBlock(i, filters, kernel_size=4, name="ConvNeXt"):
    input_x = Input((None, i))
    projection_dim = filters

    res = input_x
    if i != filters:
        res = Conv1D(filters=filters, kernel_size=1, strides=1, padding="same", trainable=True)(input_x)

    x = Conv1D(
        filters=projection_dim,
        kernel_size=kernel_size,  # Original: 7
        padding="same",
        groups=projection_dim,
        name=name + "_depthwise_conv",
    )(res)
    x = LayerNormalization(epsilon=1e-6, name=name + "_layernorm")(x)
    x = Dense(4 * projection_dim, name=name + "_pointwise_conv_1")(x)
    x = Activation("gelu", name=name + "_gelu")(x)
    x = Dense(projection_dim, name=name + "_pointwise_conv_2")(x)

    layer = Activation("linear", name=name + "_identity")
    output_node = Add()([res, layer(x)])
    return Model(inputs=input_x, outputs=output_node)


def build_model(nb_classes):
    """Gernerates a model based on convnext architecture

    Args:
        config (dict): Hyperparameters for the model:
            depth (int): Number of ConvNeXtBlocks
            embedding (int): Embedding size
            filters (list): Number of filters for the ConvNeXtBlocks (first value will not be used)
            
    Returns:
        model (Sequential): The generated model
    """
    model = Sequential()
    model.add(
        Embedding(input_dim=settings.max_features_mcc, output_dim=512, input_length=settings.maxlen_mcc, name='Input'))

    for i in range(13):
        if i == 0:
            model.add(convNeXtBlock(512, 64, 4))
            model.add(MaxPooling1D(pool_size=2, padding='same'))
        elif i == 1:
            model.add(convNeXtBlock(64, 64, 3))
            model.add(MaxPooling1D(pool_size=2, padding='same'))
        elif i < 8:
            model.add(convNeXtBlock(64, 64, 2))
            model.add(MaxPooling1D(pool_size=2, padding='same'))
        else:
            model.add(convNeXtBlock(64, 64, 2))

    # model.add(Activation('gelu'))
    model.add(Flatten())

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=AdamW(learning_rate=0.001),
                  metrics=['accuracy'])

    return model


def train(X_train, X_val, y_train, y_val, class_weight, nb_classes):
    model = build_model(nb_classes)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=settings.EPOCHS, batch_size=64, verbose=1,
              callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
              class_weight=class_weight)

    return model
