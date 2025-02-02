import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch as torch
import torchvision as tv

from diffusion import *
from process_data import *
from unet import *

# Fix the seed
torch.manual_seed(14573)

# Hyperparameters
batch_size, T, epochs, learning_rate, embedding_dim, condition = hyperparams() 
img_size = 32

# Import dataset and loaders
dataset, dataloader = set_and_loader(batch_size)
n_batchs = (len(dataset)//batch_size) + 1

# Import the model
model = UNet()
diffusion_model = diffusion_cosine(T)
optimizer = torch.optim.Adam(model.parameters(), learning_rate)


def diff_loss(
        model,
        diffusion, 
        x0, 
        t, 
        cond
):
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
        optimizer, 
        epoch, 
        n_batchs = n_batchs, 
        loader = dataloader,
        sample = True
):
    
    """
    Train a single epoch, and save the weights.

    :param model: PyTorch model
    :param loss_fn: Loss function
    :param diffusion: Diffusion model
    :param optimizer: Optimizer of the model
    :param epoch: Epoch that is being trained
    :param n_batchs: Nº of batch
    :param loader: Dataloader
    :return: Loss of the epoch
    """
    epoch_loss = 0
    for idx, (image, condition) in tqdm(enumerate(loader), total = n_batchs, desc = f"Epoch {epoch + 1} / {epochs}"):
        optimizer.zero_grad()
        actual_batchsize = image.shape[0]
        t = torch.randint(0, T, (actual_batchsize,))
        loss = loss_fn(model, diffusion, image, t, condition)
        epoch_loss += loss
        loss.backward()
        optimizer.step()
    
    # Save the weights
    path = "weights/"
    name = "UNet_diff.pth"
    torch.save(model.state_dict(), path + name)
    return epoch_loss.float()

    
def train(
        model,
        loss_fn,
        diffusion,
        optimizer,
        epochs = epochs,
        n_batchs = n_batchs,
        loader = dataloader,
        sample = False
):
    
    """
    Train the model.

    :param model: PyTorch model
    :param loss_fn: Loss function
    :param diffusion: Diffusion model
    :param optimizer: Optimizer of the model
    :param epocha: Nº of epochs
    :param n_batchs: Nº of batch
    :param loader: Dataloader
    """
    J_train = []
    for epoch in range(epochs):
        epoch_loss = train_epoch(model, loss_fn, diffusion, optimizer, epoch, n_batchs, loader)
        epoch_loss /= len(dataset)
        J_train.append(epoch_loss.detach().numpy())
        print(f"Epoch {epoch + 1} | MSE loss: {epoch_loss}")
        print("\n")

    plot_cost(J_train, save = True)
    


if __name__ == "__main__":
    # Print model info
    print_info(model, hyperparams)
    print(len(dataset))
    # Train
    train(
        model, 
        diff_loss, 
        diffusion_model, 
        optimizer,
        epochs,
        n_batchs
    )


