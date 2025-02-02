import numpy as np
import matplotlib.pyplot as plt

import torch as torch
import torchvision as tv

from TF import *


def hyperparams():
    """
    Set hyperparameters of the model.

    :return: batch size, timesteps, epochs, learning rate, embedding dimension, conditions
    """
    batch_size = 128
    epochs = 50
    learning_rate = 5e-4
    T = 100
    embedding_dim = 1024
    conditions = True
    return batch_size, T, epochs, learning_rate, embedding_dim, conditions


def n_parameters(
        model
):
    """
    Compute the number of parameters of a given model

    :param model: PyTorch model
    :return: Nº of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_info(
        model, 
        hyperparameters
):
    """
    Print valuable information about the model

    :param model: PyTorch model
    :param hyperparameters: Function to obtain the hyperparameters
    """
    batch_size, T, epochs, learning_rate, embedding_dim, condition = hyperparams() 
    print(f"=======================================================")
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
    if condition == True:
        print(f"Conditions          ==> {condition_names}") # condition_names comes from TF.py (line 36)
    else:
        print(f"Conditions          ==> None")
    print(f"=======================================================")


def set_and_loader(
        batch_size, 
        train = True
):
    """
    Get the dataset and the dataloader.

    :param batch_size: Batch size 
    :return: Dataset and Dataloader
    """

    transformation = tv.transforms.Compose([
        tv.transforms.ToTensor()
    ])

    cond_transformation = tv.transforms.Compose(
        tv.transforms.ToTensor()
        
        )

    dataset = TenerifeTempDataset(transformation, cond_transformation, train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle = True)
    return dataset, dataloader


def tensor_to_pil(
        tensor
):
    """
    Apply transformations to return a tensor into an image.

    :param tensor: Tensor to convert
    :return: tensor converted to PIL image
    """
    maxim, minim = T_max_min()
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
        T
):
    """
    Unnormalize the tensor T (temperature) to its previous distribution.

    :param T: Temperature tensor to unnormalize
    :return: Unnormalized temperature tensor
    """
    maxim, minim = T_max_min()
    T_unnorm = (T + 1)/2 * (maxim - minim) + minim
    return T_unnorm


def show_samples(
        dataset
):
    """
    Show 5 samples of the set.

    :param dataset: Dataset given to show samples
    """

    n_samples = 5
    fig, ax = plt.subplots(1, n_samples)
    for i , (tensor, condition) in enumerate(dataset):
        image = tensor_to_pil(tensor)
        ax[i].imshow(image)
        ax[i].axis(False)
        if i == n_samples - 1:
            break
    plt.show()


def plot_cost(
        train_hist, 
        dev_hist = None, 
        save = True
):
    """
    Plot the cost of the training set, and from test set if given.

    :param train_hist: Costs of the training set during training
    :param dev_hist: Cost of the dev/test set during training
    :param save: Boolean to indicate if the figures will be saved in directory
    """
    epochs_int = len(train_hist)
    epochs = np.linspace(1, epochs_int, epochs_int)
    plt.plot(epochs, train_hist, label = "Train loss")
    plt.xlabel("Epochs")
    plt.ylabel(r"MSE")
    plt.grid()
    plt.tight_layout()

    if dev_hist:
        plt.plot(epochs, dev_hist, label = "Dev cost")

    if save == True:
        plt.savefig(f"loss/{epochs_int}eps.pdf")
    else:
        pass

    plt.show()

if __name__ == "__main__":
    dataset, _ = set_and_loader(64, train = False)
    show_samples(dataset)