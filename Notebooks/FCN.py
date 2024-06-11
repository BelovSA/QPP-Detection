import os
import numpy as np
import pandas as pd 
from scipy.signal import savgol_filter
from astropy.io import fits, ascii
import torch
import torchmetrics
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class QPPClassifier(pl.LightningModule):
    
    def __init__(self, data_len=300, in_channels=2, layers_sizes=(64, 64, 64), kernel_sizes=(3, 3, 3)):
        super().__init__()

        self.save_hyperparameters()

        d_len = self.hparams.data_len
        in_ch = self.hparams.in_channels
        size1, size2, size3 = self.hparams.layers_sizes
        k1, k2, k3 = self.hparams.kernel_sizes
        
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=in_ch, out_channels=size1, 
                            kernel_size=k1, stride=1, padding='same'),
            torch.nn.BatchNorm1d(size1),
            torch.nn.ReLU()
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=size1, out_channels=size2, 
                            kernel_size=k2, stride=1, padding='same'),
            torch.nn.BatchNorm1d(size2),
            torch.nn.ReLU()
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=size2, out_channels=size3, kernel_size=k3, stride=1, padding='same'),
            torch.nn.BatchNorm1d(size3),
            torch.nn.ReLU()
        )

        self.pool = torch.nn.AvgPool1d(d_len)

        self.layer4 = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(size3, 1),
            torch.nn.Sigmoid()   
        )

        self.train_acc = torchmetrics.classification.BinaryAccuracy()
        self.valid_acc = torchmetrics.classification.BinaryAccuracy()
        self.train_f1 = torchmetrics.classification.BinaryF1Score()
        self.valid_f1 = torchmetrics.classification.BinaryF1Score()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = self.layer4(x)

        return x

    def bce_loss(self, logits, labels):
        loss = torch.nn.BCELoss()
        return loss(logits, labels.unsqueeze(1))

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.bce_loss(logits, y)
        self.log('train_loss', loss, on_epoch=True)

        self.train_acc(logits, y.unsqueeze(1))
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)

        self.train_f1(logits, y.unsqueeze(1))
        self.log('train_f1', self.train_f1, on_step=True, on_epoch=True)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.bce_loss(logits, y)
        self.log('val_loss', loss)

        self.valid_acc(logits, y.unsqueeze(1))
        self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=True)

        self.valid_f1(logits, y.unsqueeze(1))
        self.log('valid_f1', self.valid_f1, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def detrend(flare, window=0.5):
    max_ind = np.argmax(flare)
    if max_ind > len(flare) - 10:
        max_ind = len(flare) - 10
            
    decay = flare[max_ind:].copy()
    tail = flare[:max_ind].copy()
    
    w = int(len(decay) * window)
    decay = decay - savgol_filter(decay, window_length=w, polyorder=4)

    w = int(len(tail) * window)
    if w > 10:
        tail = tail - savgol_filter(tail, window_length=w, polyorder=4)
    
    if max_ind > 0:
        tail = tail.std() * np.random.uniform(-1, 1, max_ind)
        y = np.concatenate([tail, decay])
    else:
        y = decay
    return y


class QPPDataset(Dataset):
    def __init__(self, labels_path, timeseries_dir):
        self.labels = pd.read_csv(labels_path)
        self.timeseries = timeseries_dir

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        id_flare, label = self.labels.iloc[idx]
        flare = self.read_flare(int(id_flare))
        flare = np.array(flare)
        return np.float32(flare), np.float32(label)
        
    def read_flare(self, id_flare):
        path = self.timeseries + str(id_flare) + '.fits'
        flare = np.array(list(zip(*fits.open(path)[1].data))[1])

        flares = []
        for flare_i in [flare, detrend(flare)]:
            flare_i = (flare_i - flare_i.mean()) / flare_i.std()
            flares.append(flare_i)
            
        return flares