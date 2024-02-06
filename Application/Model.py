import os
import torch
import torchmetrics
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torch.nn import functional as F
import numpy as np
import pandas as pd 
from astropy.io import fits


class QPPDataset(Dataset):
    def __init__(self, timeseries_dir):
        files = []
        for file in os.listdir(timeseries_dir):
            if file.endswith(".fits"):
                files.append(os.path.join(timeseries_dir, file))
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        flare = self.read_flare(idx)
        flare = flare.reshape((1, flare.shape[0]))
        return np.float32(flare)
        
    def read_flare(self, id_flare):
        path = self.files[id_flare]
        flare = self.read_flare_by_path(path)
        flare = (flare - flare.mean()) / flare.std()
        return flare

    def read_flare_by_path(self, path):
        return np.array(list(zip(*fits.open(path)[1].data))[1])
        

class QPPClassifier(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding='same'),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU()
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same'),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU()
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same'),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU()
        )

        self.pool = torch.nn.AvgPool1d(500)
        self.dense = torch.nn.Linear(64, 1)

        self.sigmoid = torch.nn.Sigmoid()

        self.train_acc = torchmetrics.classification.BinaryAccuracy()
        self.valid_acc = torchmetrics.classification.BinaryAccuracy()
        self.train_f1 = torchmetrics.classification.BinaryF1Score()
        self.valid_f1 = torchmetrics.classification.BinaryF1Score()

    def forward(self, x):

        batch = x.shape[0]

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.pool(x)

        x = x.reshape(batch, -1)
        x = self.dense(x)
        x = self.sigmoid(x)

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


def make_predictions(fits_path, thres=0.5):
    dataset = QPPDataset(fits_path)
    data_loader = DataLoader(dataset, batch_size=8)
    
    script_path = os.path.realpath(os.path.dirname(__file__))
    model = QPPClassifier.load_from_checkpoint(script_path + "/weights/weights.ckpt")

    trainer = pl.Trainer()
    
    predictions = []
    for pred in trainer.predict(model, data_loader):
        predictions.append(pred.cpu().numpy().flatten())

    predictions = np.concatenate(predictions)

    figs = []
    for path in dataset.files:
        flare = dataset.read_flare_by_path(path)
        figs.append(flare)

    df = pd.DataFrame({'FITS': dataset.files, 'proba': predictions})
    df['QPP'] = 'Yes'
    df.loc[df['proba'] < thres, 'QPP'] = 'No'
    df['Figs'] = figs

    return df


    