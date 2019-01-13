import torch
import numpy as np

def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics = [], start_epoch=0):

    for epoch in range(0, start_epoch):
        schedular.step()

    for epoch in range(start_epoch, n_epochs):

        schedular.step()

        #Training
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch+1, n_epochs, train_loss)

        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, val_loss)

        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        print(message)

def train_epoch(
