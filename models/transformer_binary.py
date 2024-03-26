from keras.applications.convnext import LayerScale, StochasticDepth
from keras.callbacks import EarlyStopping
from keras.layers import Activation, Add, Dense, Embedding, Flatten, Lambda
from keras.layers.normalization.layer_normalization import LayerNormalization
from keras.models import Model, Sequential
from keras_nlp.layers import (PositionEmbedding, TokenAndPositionEmbedding,
                              TransformerEncoder)
from tensorflow.keras.optimizers import AdamW

import settings


def build_model(classes=1):
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
    model.add(TokenAndPositionEmbedding(vocabulary_size=settings.max_features_binary + 1,
                                        sequence_length=settings.maxlen_binary + 1, embedding_dim=128))

    for i in range(2):
        model.add(TransformerEncoder(intermediate_dim=128, num_heads=8, name=f"TransformerEncoder_{i}"))

    model.add(LayerNormalization())
    model.add(Lambda(lambda v: v[:, 0], name="ExtractCLSToken"))
    model.add(Flatten())

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
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=settings.EPOCHS, verbose=1, batch_size=256,
              callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])

    return model


def transfer_learning(X_train, X_val, y_train, y_val, nb_classes):
    """Transfer Learning of Transformer

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
