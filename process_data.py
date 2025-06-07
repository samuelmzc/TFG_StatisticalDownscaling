import numpy as np
import matplotlib.pyplot as plt

import torch as torch
import torchvision as tv

from TF import *


def n_parameters(
        model : torch.nn.Module
):
    """
    Compute the number of parameters of a given model

    :param model: PyTorch model
    :return: Nº of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_info(
        model : torch.nn.Module, 
        args
):
    """
    Print valuable information about the model

    :param model: PyTorch model
    :param hyperparameters: Function to obtain the hyperparameters
    """

    batch_size = args.batchsize 
    T = args.timesteps
    epochs = args.epochs 
    learning_rate = args.learningrate 
    embedding_dim = args.embdim 
    attention = args.attention 
    print(f"=======================- {args.island.upper()} -==========================")
    print(f"Hyperparameters")
    print(f"-------------------------------------------------------")
    print(f"Batch size          ==> {batch_size}")
    print(f"Diffusion timesteps ==> {T}")
    print(f"Embedding dimension ==> {embedding_dim}")
    print(f"Epochs              ==> {epochs}")
    print(f"Learning rate       ==> {learning_rate}")
    print(f"*******************************************************")
    print(f"Model")
    print(f"-------------------------------------------------------")
    print(f"Network             ==> UNet (ResNet Blocks)")
    print(f"Nº of parameters    ==> {n_parameters(model)}")
    print(f"Embedding           ==> Sinusoidal Positional Encoding")
    if attention == "linear":
        print(f"Attention           ==> Linear Attention")
    elif attention == "triplet":
        print(f"Attention           ==> Triplet Attention")
    else:
        print(f"Attention           ==> None")
    print(f"Conditions          ==> q, t, u, v, z") # condition_names comes from TF.py (line 36)
    print(f"=======================================================")


def set_and_loader(
        batch_size : int, 
        island : str,
        train : bool = True,
        verbose : bool = True,
        shuffle : bool = True
):
    """
    Get the dataset and the dataloader.

    :param batch_size: Batch size 
    :param island: string that indicates the island
    :param train: True for train test, False for test set
    :param verbose: bool to print info
    :param shuffle: Choose if the dataloader will be shuffled
    :return: Dataset and Dataloader
    """

    transformation = tv.transforms.Compose([
        tv.transforms.ToTensor()
    ])

    cond_transformation = tv.transforms.Compose(
        tv.transforms.ToTensor()
        
        )

    if verbose == True: print("Preparing data......")
    dataset = IslandTempDataset(path = f"netcdf/{island}.nc", transformation =  transformation, train = train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle = shuffle)
    if verbose == True: print("Done!")
    return dataset, dataloader


def tensor_to_pil(
        tensor : torch.TensorType,
        extremes : tuple
):
    """
    Apply transformations to return a tensor into an image.

    :param tensor: Tensor to convert
    :return: tensor converted to PIL image
    """
    maxim, minim = extremes
    tensor = (tensor + 1)/2 * (maxim - minim) + minim
    inverse_transforms = tv.transforms.Compose([
        tv.transforms.Lambda(lambda img : img.permute(1, 2, 0)),
        tv.transforms.Lambda(lambda img : img.detach().numpy().astype(np.uint8)),
        tv.transforms.ToPILImage()
    ])

    if len(tensor.shape) == 4:
        tensor = tensor[0, :, :, :]

    image = inverse_transforms(tensor)
    return image


def unnormalize(
        T : any,
        extremes : tuple
):
    """
    Unnormalize the tensor T (temperature) to its previous distribution.

    :param T: Temperature tensor to unnormalize
    :param extremes: tuple with (maximum_T, minimum_T)
    :return: Unnormalized temperature tensor
    """
    maxim, minim = extremes
    T_unnorm = (T + 1)/2 * (maxim - minim) + minim
    return T_unnorm


def plot_cost(
        train_hist : list,
        dev_hist : list,
        args
):
    """
    Plot the cost of the training set, and from test set if given.

    :param train_hist: Costs of the training set during training
    :param dev_hist: Cost of the dev/test set during training
    :param args: parsed arguments
    """
    epochs_int = len(train_hist)
    epochs = np.linspace(1, epochs_int, epochs_int)
    plt.plot(epochs, train_hist, label = "Train loss")
    plt.plot(epochs, dev_hist, label = "Test loss")
    plt.xlabel("Epochs")
    plt.ylabel(r"MSE")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"loss/MSE_b{args.batchsize}_e{args.epochs}_l{args.learningrate}_d{args.embdim}_t{args.timesteps}_a{args.attention}.pdf")
