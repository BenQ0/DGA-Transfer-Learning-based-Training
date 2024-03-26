import tensorflow as np
import numpy as np
import settings
from multiprocessing import Pool
from enum import Enum


class Model(Enum):
    ResNet = 1
    ResNeXt = 2
    ConvNeXt = 3
    Transformer = 4
    RWKV = 5


def preprocess_data(domains, labels, padding_length, binary):
    import tensorflow as tf
    domains, labels = np.array(domains), np.array(labels)

    # Converting Dataset to Binary classification set
    if binary:
        labels[labels >= 1] = 1

    # Map characters to ints
    if settings.USE_MULTIPROCESSING:
        if binary:
            with Pool(24) as p:
                domains = p.map(translate_domain_binary, domains, chunksize=1000)
        else:
            with Pool(24) as p:
                domains = p.map(translate_domain_mcc, domains, chunksize=1000)
    else:
        if binary:
            domains = [[settings.valid_chars_binary[c] for c in x] for x in domains]
        else:
            domains = [[settings.valid_chars_mcc[c] for c in x] for x in domains]

    # Apply padding
    if padding_length > 0:
        domains = np.array(tf.keras.utils.pad_sequences(domains, maxlen=padding_length, dtype='int8'))

    return domains, labels


# Used for Multiprocessing character mapping
def translate_domain_binary(domain):
    return [settings.valid_chars_binary[c] for c in domain]


def translate_domain_mcc(domain):
    return [settings.valid_chars_mcc[c] for c in domain]
