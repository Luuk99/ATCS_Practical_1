# imports
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import MultiplicativeLR
from pytorch_lightning.callbacks import ModelCheckpoint

# own imports
from dataset.LoadData import *
from utils import *
from models.Awe import AWEEncoder
from models.Classifier import Classifier

# full model class
class FullModel(pl.LightningModule):

    def __init__(self, model_name, lr, lr_decay):
        """
        PyTorch Lightning module that creates the overall model.
        Inputs:
            model_name - String denoting what encoder class to use.  Either 'AWE', 'UniLSTM', 'BiLSTM', or 'BiLSTMMax'
            lr - Learning rate to use for the optimizer
            lr_decay - Learning rate decay factor to use each epoch
        """
        super().__init__()
        self.save_hyperparameters()

        # check which encoder model to use
        if model_name == 'AWE':
            self.encoder = AWEEncoder()
        elif model_name == 'UniLSTM':
            self.encoder = None
        elif model_name == 'BiLSTM':
            self.encoder = None
        else:
            self.encoder = None

        # create the classifier
        self.classifier = Classifier()

    def forward(self, sentences):
        """
        The forward function calculates the loss for a given batch of sentences.
        Inputs:
            sentences - Batch of sentences with (premise, hypothesis, label) pairs
        Ouptuts:
            ?
        """

        # forward the sentences through the Encoder
        premise_embeddings, hypothesis_embeddings = self.encoder(sentences.premise, sentences.hypothesis)

        # TODO: calculate loss and accuracy

        # return the loss and accuracy
        return None, None

    # function that configures the optimizer for the model
    def configure_optimizers(self):
        # create optimizer
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr)

        # create learning rate decay
        lmbda = lambda epoch: self.hparams.lr_decay
        scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)

        # return the scheduler and optimizer
        return [optimizer], [scheduler]

    # function that performs a training step
    def training_step(self, batch, batch_idx):
        # forward the batch through the model
        train_loss, train_acc = self.forward(batch)

        # log the training loss and accuracy
        self.log("train_loss", train_loss, on_step=False, on_epoch=True)
        self.log("train_acc", train_acc, on_step=False, on_epoch=True)

        # return the training accuracy
        return train_acc

    # function that performs a validation step
    def validation_step(self, batch, batch_idx):
        # forward the batch through the model
        dev_loss, dev_acc = self.forward(batch)

        # log the development/validation loss and accuracy
        self.log("dev_loss", dev_loss)
        self.log("dev_acc", dev_acc)

    # function that performs a test step
    def test_step(self, batch, batch_idx):
        # forward the batch through the model
        test_loss, test_acc = self.forward(batch)

        # log the test loss and accuracy
        self.log("test_loss", test_loss)
        self.log("test_acc", test_acc)


# function to train the specified model
def train_model(args):
    """
    Function for training and testing a model.
    Inputs:
        args - Namespace object from the argument parser
    """

    # create the logging directory
    os.makedirs(args.log_dir, exist_ok=True)

    # create the datasets
    train_iter, dev_iter, test_iter = load_snli(None, args.batch_size)

    # create a PyTorch Lightning trainer
    trainer = pl.Trainer(default_root_dir=args.log_dir,
                         checkpoint_callback=ModelCheckpoint(save_weights_only=True, mode="max", monitor="dev_acc"),
                         gpus=1 if torch.cuda.is_available() else 0,
                         progress_bar_refresh_rate=1 if args.progress_bar else 0)
    trainer.logger._default_hp_metric = None

    # create model
    pl.seed_everything(args.seed)
    model = FullModel(model_name=args.model, lr=args.lr, lr_decay=args.lr_decay)

    # train the model
    trainer.fit(model, train_iter, dev_iter)

    # test the model
    #model = FullModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    #model.freeze()
    #test_result = trainer.test(model, test_dataloaders=test_iter, verbose=True)

    # return the test results
    #return test_result


# command line arguments parsing
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument('--model', default='AWE', type=str,
                        help='What model to use. Default is AWE',
                        choices=['AWE', 'UniLSTM', 'BiLSTM', 'BiLSTMMax'])

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use. Default is 0.1')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Minibatch size. Default is 64')
    parser.add_argument('--lr_decay', default=0.99, type=float,
                        help='Learning rate decay after each epoch. Default is 0.99')
    parser.add_argument('--lr_decrease_factor', default=5, type=int,
                        help='Factor to divide learning rate by when dev accuracy decreases. Default is 5')
    parser.add_argument('--lr_threshold', default=1e-5, type=float,
                        help='Learning rate threshold to stop at. Default is 1e-5')

    # Other hyperparameters
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--log_dir', default='pl_logs', type=str,
                        help='Directory where the PyTorch Lightning logs should be created. Default is pl_logs')
    parser.add_argument('--progress_bar', action='store_true',
                        help=('Use a progress bar indicator for interactive experimentation. '
                              'Not to be used in conjuction with SLURM jobs'))

    args = parser.parse_args()

    train_model(args)
