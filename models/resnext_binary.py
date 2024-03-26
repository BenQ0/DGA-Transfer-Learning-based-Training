from keras.callbacks import EarlyStopping
from keras.layers import (Activation, BatchNormalization, Conv1D, Dense,
                          Embedding, Flatten, Input, MaxPooling1D, add)
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


def build_model(classes=1):
    model = Sequential()
    model.add(Embedding(input_dim=settings.max_features_binary, output_dim=128, input_length=settings.maxlen_binary,
                        name='Input'))

    for i in range(3):
        if i == 0:
            model.add(resNeXtBlock(128, 512, 4, kernel_size=4))
        else:
            model.add(resNeXtBlock(512, 512, 4, kernel_size=4))

    model.add(MaxPooling1D(pool_size=4, padding='same'))
    model.add(Flatten())

    model.add(Dense(64))
    model.add(Dense(classes, name="classification_layer"))

    if classes == 1:
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer=AdamW(learning_rate=0.001),
                      metrics=['accuracy'])
    else:
        model.add(Activation("softmax"))
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=AdamW(0.001),
                      metrics=['accuracy'])
    return model


def train(X_train, X_val, y_train, y_val):
    model = build_model()
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=settings.EPOCHS, batch_size=256, verbose=1,
              callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])
    return model


def transfer_learning(X_train, X_val, y_train, y_val, nb_classes):
    """Transfer Learning of ResNeXt

    Args:
        X_train (np.array): training domains
        X_val (np.array): val domains
        y_train (np.array): training labels as multi-class targets
        y_val (np.array): validation labels as multi-class targets
    """
    # Train model on multiclass targets
    model = build_model(classes=nb_classes)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=settings.EPOCHS, batch_size=256, verbose=1,
              callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])

    # Replace last layer by binary classification layer
    layers = [l for l in model.layers]
    for layer_i, layer in enumerate(layers):
        if layer.name == "classification_layer":
            layers[layer_i] = Dense(1)
            layers[layer_i + 1] = Activation("sigmoid")

    # Fine-Tune binary model
    y_train_binary = y_train >= 1
    y_train_val = y_val >= 1
    model = Sequential(layers)
    model.compile(loss='binary_crossentropy', optimizer=AdamW(learning_rate=0.00001), metrics=['accuracy'])
    model.fit(X_train, y_train_binary >= 1, validation_data=(X_val, y_train_val >= 1), epochs=settings.EPOCHS,
              batch_size=256, verbose=1,
              callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])

    return model
