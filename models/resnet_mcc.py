import settings
from keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, add, Conv1D, Dense, Embedding, Flatten, Input, MaxPooling1D


def residual(i, filters, kernels, disable=False):
    input = Input((None, i))
    res = input
    if i != filters:
        res = Conv1D(filters=filters, kernel_size=1, strides=1, padding="same", trainable=True)(input)

    out = Conv1D(filters=filters, kernel_size=kernels[0], strides=1, padding="same",
                 kernel_initializer='glorot_uniform')(input)
    out = Activation("relu")(out)
    out = Conv1D(filters=filters, kernel_size=kernels[1], strides=1, padding="same",
                 kernel_initializer='glorot_uniform')(out)

    if not disable:
        out = add([res, out])
    if disable:
        out = input
    # out = add([out])

    return Model(inputs=input, outputs=out)


def build_model(nb_classes, max_features=settings.max_features_mcc, maxlen=settings.maxlen_mcc):
    model = Sequential()
    model.add(Embedding(input_dim=max_features, output_dim=128, input_length=maxlen, name='Input'))

    model.add(residual(128, 256, [4, 4]))
    model.add(Activation("relu"))
    model.add(MaxPooling1D(pool_size=2, padding='same'))
    model.add(residual(256, 256, [3, 3]))
    model.add(Activation("relu"))
    model.add(MaxPooling1D(pool_size=2, padding='same'))
    model.add(residual(256, 256, [2, 2]))
    model.add(Activation("relu"))
    model.add(MaxPooling1D(pool_size=2, padding='same'))
    model.add(residual(256, 256, [2, 2]))
    model.add(Activation("relu"))
    model.add(MaxPooling1D(pool_size=2, padding='same'))
    model.add(residual(256, 256, [2, 2]))
    model.add(Activation("relu"))
    model.add(MaxPooling1D(pool_size=2, padding='same'))
    model.add(residual(256, 256, [2, 2]))
    model.add(Activation("relu"))
    model.add(MaxPooling1D(pool_size=2, padding='same'))
    model.add(residual(256, 256, [2, 2]))
    model.add(Activation("relu"))
    model.add(MaxPooling1D(pool_size=2, padding='same'))
    model.add(residual(256, 256, [2, 2]))
    model.add(Activation("relu"))
    model.add(MaxPooling1D(pool_size=2, padding='same'))
    model.add(residual(256, 256, [2, 2], disable=False))
    model.add(Activation("relu"))
    model.add(residual(256, 256, [2, 2], disable=True))
    model.add(Activation("relu"))
    model.add(residual(256, 256, [2, 2], disable=True))

    model.add(Flatten())
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

    return model


def train(X_train, X_val, y_train, y_val, class_weight, nb_classes):
    model = build_model(nb_classes)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=settings.EPOCHS, batch_size=256, verbose=1,
              callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
              class_weight=class_weight)
    return model
