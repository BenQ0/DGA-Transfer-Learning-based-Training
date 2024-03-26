import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import (Activation, BatchNormalization, Conv1D, Dense,
                          Embedding, Flatten, GlobalMaxPooling1D, Input,
                          MaxPooling1D, add)
from keras.models import Model, Sequential
from keras.optimizers import Adam
from tensorflow.keras.optimizers import AdamW

import settings


def resNeXtBlock(i, filters, cardinality, kernel_size=4):
    input_x = Input((None, i))
    res = input_x
    if i != filters:
        res = Conv1D(filters=filters, kernel_size=1, strides=1, padding="same", trainable=True)(input_x)

    x = Conv1D(filters=filters, kernel_size=1, strides=1, use_bias=False)(res)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(filters=filters, kernel_size=kernel_size, groups=cardinality, strides=1, padding="same")(x)
    x = Conv1D(filters=filters, kernel_size=1, strides=1, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)

    out = add([res, x])
    out = Activation('relu')(out)
    return Model(inputs=input_x, outputs=out)


def build_model(nb_classes):
    model = Sequential()

    model.add(
        Embedding(input_dim=settings.max_features_mcc, output_dim=256, input_length=settings.maxlen_mcc, name='Input'))

    for i in range(12):
        if i == 0:
            model.add(resNeXtBlock(256, 512, 4, kernel_size=4))
            model.add(MaxPooling1D(pool_size=2, padding='same'))
        elif i == 1:
            model.add(resNeXtBlock(512, 512, 4, kernel_size=3))
            model.add(MaxPooling1D(pool_size=2, padding='same'))
        elif i < 8:
            model.add(resNeXtBlock(512, 512, 4, kernel_size=2))
            model.add(MaxPooling1D(pool_size=2, padding='same'))
        else:
            model.add(resNeXtBlock(512, 512, 4, kernel_size=2))

    model.add(Flatten())

    model.add(Dense(128))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=AdamW(learning_rate=0.0001),
                  metrics=['accuracy'])

    return model


def train(X_train, X_val, y_train, y_val, class_weight, nb_classes):
    model = build_model(nb_classes)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=settings.EPOCHS, batch_size=32, verbose=1,
              callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
              class_weight=class_weight)
    return model
