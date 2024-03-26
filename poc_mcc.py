from models import rwkv_mcc
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from models import (convnext_mcc, resnet_mcc, resnext_mcc,
                    rwkv_mcc, transformer_mcc)
from util import preprocess_data
import settings
import argparse

# Select one of the following models to train and evaluate:
model_choices = ['resnet', 'resnext', 'convnext', 'transformer', 'rwkv']
parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=model_choices)
select_model = {'resnet': resnet_mcc, 'resnext': resnext_mcc, 'convnext': convnext_mcc, 'transformer': transformer_mcc,
                'rwkv': rwkv_mcc}
selected_model = select_model[parser.parse_args().model]

# Data processing, include your data here using the following format
data = {"benign": ["bengin-nx1.xy", "bengin-nx2.xy", "bengin-nx3.xy"],
        "dga1": ["domain1-dga1.xy", "domain2-dga1.xy", "domain3-dga1.xx"],
        "dga2": ["domain1-dga2.xy", "domain2-dga2.xy", "domain3-dga2.xx"],
        "dga3": ["domain1-dga3.xy", "domain2-dga3.xy", "domain3-dga3.xx"],
        # ...
        }

# Extract domains and labels from data dictionary
labels = []
domains = []
for i, domains_class in enumerate(data):
    labels += [i] * len(data[domains_class])
    domains += data[domains_class]

X, y = preprocess_data(domains, labels, padding_length=253, binary=False)
nb_classes = len(np.unique(y))
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, stratify=y)

# Calc class weights to weight each dga class based on their number of samples. 
# As proposed by Tran et al. in "A LSTM based framework for handling multiclass imbalance in DGA botnet detection"
class_weight = dict()
for g in set(y_train):
    score = np.math.pow(
        len(y_train) / float(len(y_train[y_train == g])),
        settings.class_weighting_power)
    class_weight[g] = float(score)

# Start Training
# Define number of epochs in settings file
if selected_model in [resnet_mcc, resnext_mcc, convnext_mcc, transformer_mcc]:
    model = selected_model.train(X_train, X_val, y_train, y_val, class_weight=class_weight, nb_classes=nb_classes)
    predicted_labels = model.predict(X_val)
    predicted_labels = np.argmax(predicted_labels, axis=1)

elif selected_model == rwkv_mcc:
    X_train = tf.keras.utils.pad_sequences(X_train, maxlen=256, dtype='int8')
    X_val = tf.keras.utils.pad_sequences(X_val, maxlen=256, dtype='int8')
    model = rwkv_mcc.train(X_train, X_val, y_train, y_val, class_weight=class_weight, nb_classes=nb_classes)

    model_inference = rwkv_mcc.build_model_inference(nb_classes=nb_classes)
    model_inference.load_state_dict(model.state_dict())
    predicted_labels = rwkv_mcc.predict(model_inference, X_val)

# Evaluate the results e.g., using the F1-Score
print(f"F1-Score: {f1_score(y_val, predicted_labels, average='macro')}")
