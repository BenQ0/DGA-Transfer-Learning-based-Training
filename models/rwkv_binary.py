import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import settings
from rwkv_src.model import GPT, GPTConfig
from rwkv_src.model_run import RWKV_GPT
from rwkv_src.trainer import Trainer, TrainerConfig
from rwkv_src.utils import RWKV_Dataset

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


def build_model_train():
    model = GPT(GPTConfig(settings.max_features_binary, 64, model_type="RWKV", n_layer=4, n_embd=128, classes=1)).cuda()
    return model


def build_model_inference():
    model = RWKV_GPT(None, "gpu", "RWKV", settings.max_features_binary, n_layer=4, n_embd=128, ctx_len=64).cuda()
    return model


def train(X_train, X_val, y_train, y_val, classes=1):
    train_dataset = RWKV_Dataset(X_train, y_train, ctx_len=64, vocab_size=settings.max_features_binary)
    val_dataset = RWKV_Dataset(X_val, y_val, ctx_len=64, vocab_size=settings.max_features_binary)
    model = GPT(GPTConfig(train_dataset.vocab_size, train_dataset.ctx_len, model_type="RWKV", n_layer=4, n_embd=128,
                          classes=classes)).cuda()

    tconf = TrainerConfig(model_type="RWKV", max_epochs=settings.EPOCHS, batch_size=128, learning_rate=0.001,
                          lr_decay=False, num_workers=0, patience=5)

    trainer = Trainer(model, train_dataset, val_dataset, tconf)

    trainer.train()
    model.load_state_dict(trainer.best_model_sdict)
    return model


def transfer_learning(X_train, X_val, y_train, y_val, nb_classes):
    """Function for 3rd experiment of the RWKV Transfer Learning

    Args:
        config (_type_): best rwkv binary config
        X_train (_type_): train data
        X_val (_type_): validation data
        y_train (_type_): mcc train labels
        y_val (_type_): mcc validation labels

    Returns:
        Fine-tuned rwkv model
    """
    y_train_binary = y_train >= 1
    y_val_binary = y_val >= 1

    # Train RWKV binary backbone using a mcc head
    rwkv_pretrain = train(X_train, X_val, y_train, y_val, classes=nb_classes)

    # Replace the head by a binary one
    rwkv_pretrain.head = torch.nn.Linear(128, 1, bias=False)
    rwkv_pretrain.config.classes = 1

    # Train RWKV binary backbone with binary head
    train_dataset = RWKV_Dataset(X_train, y_train_binary, ctx_len=64, vocab_size=settings.max_features_binary)
    val_dataset = RWKV_Dataset(X_val, y_val_binary, ctx_len=64, vocab_size=settings.max_features_binary)

    # Decrease LR to 1e-5
    tconf = TrainerConfig(model_type='RWKV', max_epochs=settings.EPOCHS, batch_size=128, learning_rate=0.00001,
                          lr_decay=False, num_workers=0, patience=5)
    trainer = Trainer(rwkv_pretrain.cuda(), train_dataset, val_dataset, tconf)

    trainer.train()
    rwkv_pretrain.load_state_dict(trainer.best_model_sdict)
    return rwkv_pretrain


def predict(model_rwkv, X, batch_size=32):
    model_rwkv.eval()
    dataset = RWKV_Dataset(X, np.zeros_like(X), ctx_len=64, device="cpu", vocab_size=settings.max_features_binary)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    preds = []

    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            domains = data[0].cuda()
            pred = model_rwkv(domains)
            pred = torch.sigmoid(pred[:, -1])
            pred = pred[:, -1]
            pred = pred.flatten()
            preds += list(pred.cpu().numpy())

    return np.array(preds)
