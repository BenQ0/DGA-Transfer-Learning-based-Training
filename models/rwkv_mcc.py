import numpy as np
import torch
from torch.utils.data import DataLoader
from rwkv_src.model import GPT, GPTConfig
from rwkv_src.model_run import RWKV_GPT
from rwkv_src.trainer import Trainer, TrainerConfig
from rwkv_src.utils import RWKV_Dataset
from tqdm import tqdm

import settings

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


def build_model(nb_classes):
    model = GPT(GPTConfig(settings.max_features_mcc, 256, model_type="RWKV", n_layer=4, n_embd=128,
                          classes=nb_classes))  # <- classes normally 1, only needed for mcc to binary transfer learning
    return model


def build_model_inference(nb_classes):
    model = RWKV_GPT(None, "gpu", "RWKV", settings.max_features_mcc, n_layer=4, n_embd=128, ctx_len=256,
                     classes=nb_classes)
    return model


def train(X_train, X_val, y_train, y_val, nb_classes, class_weight=None):
    if class_weight is not None:
        class_weight = torch.tensor(list(class_weight.values())).cuda()

    train_dataset = RWKV_Dataset(X_train, y_train, ctx_len=256, vocab_size=settings.max_features_mcc)
    val_dataset = RWKV_Dataset(X_val, y_val, ctx_len=256, vocab_size=settings.max_features_mcc)
    model = GPT(
        GPTConfig(train_dataset.vocab_size, train_dataset.ctx_len, model_type="RWKV", class_weights=class_weight,
                  n_layer=4, n_embd=128, classes=nb_classes)).cuda()

    # NOTE: Learning Rate decay could improve training, however, it is disabled for comparability
    tconf = TrainerConfig(model_type="RWKV", max_epochs=settings.EPOCHS, batch_size=128, learning_rate=0.0001,
                          lr_decay=False, num_workers=0, patience=5)
    trainer = Trainer(model, train_dataset, val_dataset, tconf)

    trainer.train()
    model.load_state_dict(trainer.best_model_sdict)
    return model


def predict(model_rwkv, X):
    dataset = RWKV_Dataset(X, np.zeros_like(X), ctx_len=256, device="cuda", vocab_size=settings.max_features_mcc)
    dataloader = DataLoader(dataset, batch_size=128)

    flipped_dict = dict(zip(settings.valid_chars_mcc.values(), settings.valid_chars_mcc.keys()))
    flipped_dict[0] = ""

    preds = []

    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            domains = data[0]
            pred = model_rwkv(domains)
            pred = pred[:, -1]
            pred = torch.functional.F.softmax(pred, dim=-1)
            pred = np.argmax(pred, axis=-1)
            pred = pred.flatten()
            preds += list(pred.cpu().numpy())
    return np.array(preds)
