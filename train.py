import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch as torch
import torchvision as tv

from diffusion import *
from process_data import *
from unet import *


def MSE_loss(
        model,
        diffusion, 
        x0, 
        t, 
        cond
) -> torch.Tensor:
    """
    Compute the MSE error.

    :param model: PyTorch model
    :param diff_model: Diffusion model
    :param x0: Tensor of the input at t = 0
    :param t: Tensor of shape (batch_size,) including timesteps
    :param cond: Tensor with synoptic conditions 
    :return: MSE loss
    """
    x_new, noise = diffusion.forward(x0, t)
    noise_pred = model.forward(x_new, t, cond)
    return torch.nn.functional.mse_loss(noise, noise_pred)


def train_epoch(
        model, 
        loss_fn, 
        diffusion, 
        timesteps,
        optimizer, 
        epoch,
        epochs,
        attention, 
        n_batchs, 
        loader,
        n_batchs_test,
        test_loader,
        args,
        weights_path
) -> tuple:
    
    """
    Train a single epoch, and save the weights.

    :param model: PyTorch model
    :param loss_fn: Loss function
    :param diffusion: Diffusion model
    :param timesteps: Nº of timesteps
    :param optimizer: Optimizer of the model
    :param epoch: Epoch that is being trained
    :param epochs: Nº of epochs
    :param n_batchs: Nº of batchs
    :param loader: Dataloader
    :param n_batchs_test: Nº of batchs in test set
    :param test: Dataloader for test
    :param verbose: Choose to print the progress of the training
    :return: Loss of the epoch
    """
    epoch_loss = 0
    iterable = tqdm(enumerate(loader), total = n_batchs, desc = f"Epoch {epoch + 1} / {epochs}") if args.verbose == True else enumerate(loader)
    for idx, (image, condition) in iterable:
        optimizer.zero_grad()
        actual_batchsize = image.shape[0]
        t = torch.randint(0, timesteps, (actual_batchsize,))
        loss = loss_fn(model, diffusion, image, t, condition)
        epoch_loss += loss
        loss.backward()
        optimizer.step()
    
    epoch_test = 0
    with torch.no_grad():
        for idx, (image, condition) in enumerate(test_loader):
            actual_batchsize_test = image.shape[0]
            t = torch.randint(0, timesteps, (actual_batchsize_test,))
            loss = loss_fn(model, diffusion, image, t, condition)
            epoch_test += loss
    
    torch.save(model.state_dict(), weights_path)
    return (epoch_loss.float(), epoch_test.float())

    
def train(
        model,
        loss_fn,
        diffusion,
        timesteps,
        optimizer,
        epochs,
        attention,
        n_batchs,
        loader,
        n_batchs_test,
        test,
        dataset,
        ds_test,
        args,
        weights_path
) -> None:
    
    """
    Train the model.

    :param model: PyTorch model
    :param loss_fn: Loss function
    :param diffusion: Diffusion model
    :param timesteps: Nº of timesteps
    :param optimizer: Optimizer of the model
    :param epochs: Nº of epochs
    :param attention: str with attention type
    :param n_batchs: Nº of batch
    :param loader: Dataloader
    :param n_batchs_test: Nº of batchs in test set
    :param dataset: Train set
    :param ds_set: Test set
    :param args: parsed arguments
    """
    J_train = []
    J_test = []
    for epoch in range(epochs):
        epoch_loss, test_loss = train_epoch(model, loss_fn, diffusion, timesteps, optimizer, epoch, epochs, attention, n_batchs, loader, n_batchs_test, test, args, weights_path)
        epoch_loss /= len(dataset)
        test_loss /= len(ds_test)
        J_train.append(epoch_loss.detach().numpy())
        J_test.append(test_loss.detach().numpy())
        print(f"Epoch {epoch + 1} | Train MSE loss: {epoch_loss}, Test MSE loss: {test_loss}")
        print("\n")

    plot_cost(J_train, J_test, args)
    return (J_train, J_test)
    




