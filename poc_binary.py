import argparse

import numpy as np
from models import rwkv_binary
import tensorflow as tf
from models import (convnext_binary, resnet_binary, resnext_binary,
                    transformer_binary)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from util import preprocess_data

# Select one of the following models to train and evaluate:
model_choices = ['resnet', 'resnext', 'convnext', 'transformer', 'rwkv']
parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=model_choices, required=True,
                    help="Select one of the following models: " + str(model_choices))
parser.add_argument('--tl', action='store_true', help="Use Transfer Learning")

select_model = {'resnet': resnet_binary, 'resnext': resnext_binary, 'convnext': convnext_binary,
                'transformer': transformer_binary, 'rwkv': rwkv_binary}
selected_model = select_model[parser.parse_args().model]

# Data processing, only use the effective 2nd layer domain:
if parser.parse_args().tl:
    # Include your multiclass data here using the following format. Only the 2nd layer domain is used.
    data = {"benign": ["bengin-nx1", "bengin-nx2", "bengin-nx3"],
            "dga1": ["domain1-dga1", "domain2-dga1", "domain3-dga1"],
            "dga2": ["domain1-dga2", "domain2-dga2", "domain3-dga2"],
            "dga3": ["domain1-dga3", "domain2-dga3", "domain3-dga3"],
            # ...
            }

    # Extract domains and labels from data dictionary
    labels = []
    domains = []
    for i, domains_class in enumerate(data):
        labels += [i] * len(data[domains_class])
        domains += data[domains_class]
    nb_classes = len(np.unique(labels))
else:
    # Include your binary data here using the following format
    domains = ["benign0", "benign1", "benign2", "malicious0", "malicious1", "malicious2"]
    labels = [0, 0, 0, 1, 1, 1]

X, y = preprocess_data(domains, labels, padding_length=63, binary=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Define number of epochs in settings file
if selected_model in [resnet_binary, resnext_binary, convnext_binary, transformer_binary]:
    if parser.parse_args().tl:
        model = selected_model.transfer_learning(X_train, X_val, y_train, y_val, nb_classes)
    else:
        model = selected_model.train(X_train, X_val, y_train, y_val)

    predicted_labels = model.predict(X_val)

elif selected_model == rwkv_binary:
    X_train = tf.keras.utils.pad_sequences(X_train, maxlen=64, dtype='int8')
    X_val = tf.keras.utils.pad_sequences(X_val, maxlen=64, dtype='int8')

    if parser.parse_args().tl:
        model = rwkv_binary.transfer_learning(X_train, X_val, y_train, y_val, nb_classes)
    else:
        model = rwkv_binary.train(X_train, X_val, y_train, y_val)

    model_inference = rwkv_binary.build_model_inference()
    model_inference.load_state_dict(model.state_dict())
    model_inference.cuda()
    predicted_labels = rwkv_binary.predict(model_inference, X_val)

predicted_labels = predicted_labels > 0.5

# Evaluate the results e.g., using the accuracy
print(f"ACC {accuracy_score(y_val, predicted_labels)}")
