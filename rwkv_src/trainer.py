########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm.auto import tqdm
import numpy as np
import os
import datetime
import sys
import math

# import wandb  # comment this if you don't have wandb
# print('logging to wandb... (comment it if you don\'t have wandb)')

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


class TrainerConfig:
    max_epochs = 10
    batch_size = 64
    learning_rate = 4e-4
    betas = (0.9, 0.99)
    eps = 1e-8
    grad_norm_clip = 1.0
    lr_decay = True  # linear warmup followed by cosine decay
    warmup_tokens = 0
    final_tokens = 0
    epoch_save_frequency = 0
    epoch_save_path = 'trained-'
    num_workers = 0  # for DataLoader
    patience = 5

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.avg_loss = -1
        self.steps = 0
        
        # EarlyStop Mechanism
        self.best_val_loss = np.inf
        self.early_counter = 0

        if 'wandb' in sys.modules:
            cfg = model.config
            for k in config.__dict__:
                setattr(cfg, k, config.__dict__[k])  # combine cfg
            wandb.init(project="RWKV-LM", name=self.get_run_name() + '-' +
                       datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S'), config=cfg, save_code=False)

        self.device = 'cpu'
        if torch.cuda.is_available():  # take over whatever gpus are on the system
            self.device = torch.cuda.current_device()

    def get_run_name(self):
        raw_model = self.model.module if hasattr(
            self.model, "module") else self.model
        cfg = raw_model.config
        run_name = str(cfg.vocab_size) + '-' + str(cfg.ctx_len) + '-' + \
            cfg.model_type + '-' + str(cfg.n_layer) + '-' + str(cfg.n_embd)
        return run_name

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset

            if config.num_workers > 0:
                loader = DataLoader(data, shuffle=False, pin_memory=True,
                                    batch_size=config.batch_size,
                                    num_workers=config.num_workers)
            else:
                loader = DataLoader(data, shuffle=False,
                                    batch_size=config.batch_size,
                                    num_workers=config.num_workers)

            pbar = tqdm(enumerate(loader), total=len(
                loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') if is_train else enumerate(loader)

            loss_epoch = 0 # Track loss

            for it, (x, y) in pbar:
                x = x.to(self.device)  # place data on the correct device
                y = y.to(self.device)

                with torch.set_grad_enabled(is_train):
                    _, loss = model(x, y)  # forward the model
                loss_epoch += loss.item() / len(loader)

                if is_train:  # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()

                    if config.grad_norm_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.grad_norm_clip)

                    optimizer.step()

                    if config.lr_decay:  # decay the learning rate based on our progress
                        # number of tokens processed this step (i.e. label is not -100)
                        self.tokens += (y >= 0).sum()
                        lr_final_factor = config.lr_final / config.learning_rate
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = lr_final_factor + \
                                (1 - lr_final_factor) * float(self.tokens) / \
                                float(config.warmup_tokens)
                            progress = 0
                        else:
                            # exponential learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            if progress >= 1:
                                lr_mult = lr_final_factor
                            else:
                                lr_mult = math.exp(math.log(lr_final_factor) * pow(progress, 1))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    now_loss = loss.item()  # report progress
                    self.lr = lr

                    if 'wandb' in sys.modules:
                        wandb.log({"loss": now_loss},
                                  step=self.steps * self.config.batch_size)
                    self.steps += 1

                    if self.avg_loss < 0:
                        self.avg_loss = now_loss
                    else:
                        factor = 1 / (it + 1)
                        self.avg_loss = self.avg_loss * \
                            (1.0 - factor) + now_loss * factor
                    pbar.set_description(
                        f"mini-epoch {epoch+1} iter {it}: ppl {math.exp(self.avg_loss):.2f} loss {self.avg_loss:.4f} lr {lr:e}")
                    
            return loss_epoch


        self.tokens = 0  # counter used for learning rate decay
        for epoch in range(config.max_epochs):
            self.early_counter += 1

            train_loss = run_epoch('train')
            val_loss = run_epoch('val')
            
            
            # EarlyStop:
            if val_loss < self.best_val_loss:
                self.early_counter = 0
                self.best_val_loss = val_loss
                self.best_model_sdict = raw_model.state_dict()
            if self.early_counter >= config.patience:
                return
