import settings
from keras.callbacks import EarlyStopping

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, add, Conv1D, Dense, Embedding, Flatten, Input, MaxPooling1D
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam, AdamW


def residual(i, filters, kernels):
    input = Input((None, i))
    res = input
    if i != filters:
        res = Conv1D(filters=filters, kernel_size=1, strides=1, padding="same", trainable=True)(input)

    out = Conv1D(filters=filters, kernel_size=kernels[0], strides=1, padding="same")(input)
    out = Activation("relu")(out)
    out = Conv1D(filters=filters, kernel_size=kernels[1], strides=1, padding="same")(out)
    out = add([res, out])

    return Model(inputs=input, outputs=out)


def build_model(max_features=settings.max_features_binary, maxlen=settings.maxlen_binary, classes=1):
    model = Sequential()
    model.add(Embedding(input_dim=max_features, output_dim=128, input_length=maxlen, name='Input'))

    model.add(residual(128, 128, [4, 4]))

    model.add(Activation('relu'))

    model.add(MaxPooling1D(pool_size=4, padding='same'))
    model.add(Flatten())

    # Used for transfer learning / fine-tuning
    if classes == 1:
        model.add(Dense(classes, name="classification_layer"))
        model.add(Activation("sigmoid"))
        model.compile(loss='binary_crossentropy', optimizer='adam')

    else:
        model.add(Dense(classes, name="classification_layer"))
        model.add(Activation("softmax"))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    return model


def train(X_train, X_val, y_train, y_val):
    model = build_model()
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=settings.EPOCHS, batch_size=128, verbose=1,
              callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])
    return model


def transfer_learning(X_train, X_val, y_train, y_val, nb_classes):
    """Transfer Learning of ResNet

    Args:
        X_train (np.array): training domains
        X_val (np.array): val domains
        y_train (np.array): training labels as multi-class targets
        y_val (np.array): validation labels as multi-class targets
    """
    # Train model on multiclass targets
    model_resnet = build_model(classes=nb_classes)
    model_resnet.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=settings.EPOCHS, batch_size=128,
                     verbose=1,
                     callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])

    # Replace last layer by binary classification layer
    layers_resnet = [l for l in model_resnet.layers]
    for layer_i, layer in enumerate(layers_resnet):
        if layer.name == "classification_layer":
            layers_resnet[layer_i] = Dense(1)
            layers_resnet[layer_i + 1] = Activation("sigmoid")

    # Fine-Tune binary model
    y_train_binary = y_train >= 1
    y_train_val = y_val >= 1
    model_resnet = Sequential(layers_resnet)
    model_resnet.compile(loss='binary_crossentropy', optimizer=AdamW(learning_rate=0.00001))
    model_resnet.fit(X_train, y_train_binary >= 1, validation_data=(X_val, y_train_val >= 1), epochs=settings.EPOCHS,
                     batch_size=128, verbose=1,
                     callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])

    return model_resnet
