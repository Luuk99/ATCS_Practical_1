# imports
import argparse
import os

# DEBUG
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# own imports
from dataset.LoadData import *
from models.Awe import AWEEncoder
from models.UniLSTM import UniLSTM
from models.BiLSTM import BiLSTM
from models.BiLSTMMax import BiLSTMMax
from models.Classifier import Classifier

# full model class
class FullModel(pl.LightningModule):

    def __init__(self, model_name, vocab, lr, lr_decay, batch_size=64):
        """
        PyTorch Lightning module that creates the overall model.
        Inputs:
            model_name - String denoting what encoder class to use.  Either 'AWE', 'UniLSTM', 'BiLSTM', or 'BiLSTMMax'
            vocab - Vocabulary from alignment between SNLI dataset and GloVe vectors
            lr - Learning rate to use for the optimizer
            lr_decay - Learning rate decay factor to use each epoch
            batch_size - Size of the batches. Default is 64
        """
        super().__init__()
        self.save_hyperparameters()

        # create an embedding layer for the vocabulary embeddings
        self.glove_embeddings = nn.Embedding.from_pretrained(vocab.vectors)

        # check which encoder model to use
        if model_name == 'AWE':
            self.encoder = AWEEncoder()
            self.classifier = Classifier()
        elif model_name == 'UniLSTM':
            self.encoder = UniLSTM()
            self.classifier = Classifier(input_dim=4*2048)
        elif model_name == 'BiLSTM':
            self.encoder = BiLSTM()
            self.classifier = Classifier(input_dim=4*2*2048)
        else:
            self.encoder = BiLSTMMax()
            self.classifier = Classifier(input_dim=4*2*2048)

        # create the loss function
        self.loss_function = nn.CrossEntropyLoss()

        # create instance to save the last validation accuracy
        self.last_val_acc = None

    def forward(self, sentences):
        """
        The forward function calculates the loss for a given batch of sentences.
        Inputs:
            sentences - Batch of sentences with (premise, hypothesis, label) pairs
        Ouptuts:
            loss - Cross entropy loss of the predictions
            accuracy - Accuracy of the predictions
        """

        # get the sentence lengths of the batch
        lengths_premises = torch.tensor([x[x!=1].shape[0] for x in sentences.premise], device=self.device)
        lengths_hypothesis = torch.tensor([x[x!=1].shape[0] for x in sentences.hypothesis], device=self.device)

        # pass premises and hypothesis through the embeddings
        premises = self.glove_embeddings(sentences.premise)
        hypothesis = self.glove_embeddings(sentences.hypothesis)

        # forward the premises and hypothesis through the Encoder
        premises = self.encoder(premises, lengths_premises)
        hypothesis = self.encoder(hypothesis, lengths_hypothesis)

        # calculate the difference and multiplication
        difference = torch.abs(premises - hypothesis)
        multiplication = premises * hypothesis

        # create the sentence representations
        sentence_representations = torch.cat([premises, hypothesis, difference, multiplication], dim=1)

        # pass through the classifier
        predictions = self.classifier(sentence_representations)

        # calculate the loss and accuracy
        loss = self.loss_function(predictions, sentences.label)
        predicted_labels = torch.argmax(predictions, dim=1)
        accuracy = torch.true_divide(torch.sum(predicted_labels == sentences.label), torch.tensor(sentences.label.shape[0], device=sentences.label.device))

        # return the loss and accuracy
        return loss, accuracy

    # function that configures the optimizer for the model
    def configure_optimizers(self):
        # create optimizer
        optimizer = torch.optim.SGD([{'params': self.encoder.parameters()},
                {'params': self.classifier.parameters()}], lr=self.hparams.lr)

        # freeze the embeddings
        self.glove_embeddings.weight.requires_grad = False

        # create learning rate decay
        lr_scheduler = {
            'scheduler': StepLR(optimizer=optimizer, step_size=1, gamma=self.hparams.lr_decay),
            'name': 'learning_rate'
        }

        # return the scheduler and optimizer
        return [optimizer], [lr_scheduler]

    # function that performs a training step
    def training_step(self, batch, batch_idx):
        # forward the batch through the model
        train_loss, train_acc = self.forward(batch)

        # log the training loss and accuracy
        self.log("train_loss", train_loss, on_step=False, on_epoch=True)
        self.log("train_acc", train_acc, on_step=False, on_epoch=True)

        # return the training loss
        return train_loss

    # function that performs a validation step
    def validation_step(self, batch, batch_idx):
        # forward the batch through the model
        val_loss, val_acc = self.forward(batch)

        # log the validation loss and accuracy
        self.log("val_loss", val_loss)
        self.log("val_acc", val_acc)

        # save the validation accuracy
        self.last_val_acc = val_acc

    # function that performs a test step
    def test_step(self, batch, batch_idx):
        # forward the batch through the model
        test_loss, test_acc = self.forward(batch)

        # log the test loss and accuracy
        self.log("test_loss", test_loss)
        self.log("test_acc", test_acc)


# Pytorch Lightning callback class
class PLCallback(pl.Callback):

    def __init__(self, lr_decrease_factor=5):
        """
        Inputs:
            lr_decrease_factor - Factor to divide the learning rate by when
                the validation accuracy decreases. Default is 5
        """
        super().__init__()

        # save the decrease factor
        self.decrease_factor = lr_decrease_factor

        # initialize the previous validation accuracy as 0
        self.last_val_acc = 0

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        """
        This function is called after every training epoch
        """

        # check if the learning rate has fallen under 10e-5
        current_lr = trainer.optimizers[0].state_dict()['param_groups'][0]['lr']
        if (current_lr < 10e-5):
            # stop training
            trainer.should_stop = True

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        This function is called after every validation epoch
        """

        # check of the validation accuracy has decreased
        if pl_module.last_val_acc < self.last_val_acc:
            # divide the learning rate by the specified factor
            state_dict = trainer.optimizers[0].state_dict()
            state_dict['param_groups'][0]['lr'] = state_dict['param_groups'][0]['lr'] / self.decrease_factor
            new_optimizer = torch.optim.SGD([{'params': pl_module.encoder.parameters()},
                    {'params': pl_module.classifier.parameters()}], lr=state_dict['param_groups'][0]['lr'])
            new_optimizer.load_state_dict(state_dict)

            # update scheduler
            scheduler_state_dict = trainer.lr_schedulers[0]['scheduler'].state_dict()
            new_step_scheduler = StepLR(optimizer=new_optimizer, step_size=1, gamma=scheduler_state_dict['gamma'])
            new_step_scheduler.load_state_dict(scheduler_state_dict)
            new_scheduler = {
                'scheduler': new_step_scheduler,
                'name': 'learning_rate'
            }

            # use the new scheduler and optimizer
            trainer.optimizers = [new_optimizer]
            trainer.lr_schedulers = trainer.configure_schedulers([new_scheduler])

        # save the validation accuracy
        self.last_val_acc = pl_module.last_val_acc


# function to train the specified model
def train_model(args):
    """
    Function for training and testing a model.
    Inputs:
        args - Namespace object from the argument parser
    """

    # create the logging directory
    os.makedirs(args.log_dir, exist_ok=True)

    # create the vocabulary and datasets
    vocab, train_iter, dev_iter, test_iter = load_snli(device=None, batch_size=args.batch_size, development=args.development)

    # check if a checkpoint has been given
    if args.checkpoint_dir is None:
        # create the callback for decreasing the learning rate
        pl_callback = PLCallback(lr_decrease_factor=args.lr_decrease_factor)

        # create a learning rate monitor callback
        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        # create a PyTorch Lightning trainer
        trainer = pl.Trainer(default_root_dir=args.log_dir,
                         checkpoint_callback=ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                         gpus=1 if torch.cuda.is_available() else 0,
                         callbacks=[lr_monitor, pl_callback],
                         max_epochs=30,
                         progress_bar_refresh_rate=1 if args.progress_bar else 0)
        trainer.logger._default_hp_metric = None

        # create model
        pl.seed_everything(args.seed)
        model = FullModel(model_name=args.model, vocab=vocab, lr=args.lr,
                          lr_decay=args.lr_decay, batch_size=args.batch_size)

        # train the model
        trainer.fit(model, train_iter, dev_iter)

        # load the best checkpoint
        model = FullModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    else:
        # create a PyTorch Lightning trainer
        trainer = pl.Trainer(logger=False,
                         checkpoint_callback=False,
                         gpus=1 if torch.cuda.is_available() else 0,
                         progress_bar_refresh_rate=1 if args.progress_bar else 0)

        # load model from the given checkpoint
        model = FullModel.load_from_checkpoint(args.checkpoint_dir)

    # test the model
    model.freeze()
    test_result = trainer.test(model, test_dataloaders=test_iter, verbose=True)

    # return the test results
    return test_result


# command line arguments parsing
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # model hyperparameters
    parser.add_argument('--model', default='AWE', type=str,
                        help='What model to use. Default is AWE',
                        choices=['AWE', 'UniLSTM', 'BiLSTM', 'BiLSTMMax'])

    # optimizer hyperparameters
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

    # loading hyperparameters
    parser.add_argument('--checkpoint_dir', default=None, type=str,
                        help='Directory where the model checkpoint is located. Default is None (no checkpoint used)')

    # other hyperparameters
    parser.add_argument('--seed', default=1234, type=int,
                        help='Seed to use for reproducing results. Default is 1234')
    parser.add_argument('--log_dir', default='pl_logs', type=str,
                        help='Directory where the PyTorch Lightning logs should be created. Default is pl_logs')
    parser.add_argument('--progress_bar', action='store_true',
                        help=('Use a progress bar indicator for interactive experimentation. '
                              'Not to be used in conjuction with SLURM jobs'))
    parser.add_argument('--development', action='store_true',
                        help=('Limit the size of the datasets in development.'))

    args = parser.parse_args()

    train_model(args)
