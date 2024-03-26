import numpy as np
from keras.applications.convnext import LayerScale, StochasticDepth
from keras.callbacks import EarlyStopping
from keras.layers import (Activation, Add, Conv1D, Dense, Dropout, Embedding,
                          Flatten, GlobalMaxPooling1D, Input, Lambda)
from keras.layers.normalization.layer_normalization import LayerNormalization
from keras.models import Model, Sequential
from keras_nlp.layers import (PositionEmbedding, TokenAndPositionEmbedding,
                              TransformerEncoder)
from tensorflow.keras.optimizers import AdamW

import settings


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

    model.add(TokenAndPositionEmbedding(vocabulary_size=settings.max_features_mcc + 1,
                                        sequence_length=settings.maxlen_mcc + 1, embedding_dim=256))

    for i in range(4):
        model.add(TransformerEncoder(intermediate_dim=32, num_heads=9, dropout=0.25, name=f"TransformerEncoder_{i}"))

    model.add(LayerNormalization())
    model.add(Lambda(lambda v: v[:, 0], name="ExtractCLSToken"))
    model.add(Flatten())

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=AdamW(learning_rate=0.0001),
                  metrics=['accuracy'])
    return model


def train(X_train, X_val, y_train, y_val, class_weight, nb_classes):
    model = build_model(nb_classes)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=settings.EPOCHS, verbose=1, batch_size=32,
              callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
              class_weight=class_weight)

    return model
