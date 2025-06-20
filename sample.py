import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 
from PIL import Image
from tqdm import tqdm
import imageio

import torch as torch
import torchvision as tv

from diffusion import *
from process_data import *
from unet import *


def infere_given_index(
        idx_ : int, 
        model : torch.nn.Module, 
        diffusion : classmethod, 
        img_size : int,
        dataset : torch.utils.data.Dataset,
        args,
        mask : np.ndarray = None
) -> None:
    """
    Run through the dataset for a given index, and plot inference.

    :param idx_: Dataset's index to sample
    :param model: PyTorch model
    :param diffusion: The diffusion model
    :param img_size: Height or weight of the image (if image shape is is n x n)
    """
    for idx, (image, cond) in enumerate(dataset):
        image = image[0]
        if idx == idx_:
            Tmax, Tmin = dataset.T_extremes()
            fig, ax = plt.subplots(1, 2, figsize = (6,5))
            xT = torch.randn(1, 1, img_size, img_size)
            x0 = diffusion.sample(xT, model, cond[None, ...])
            img = tensor_to_pil(x0, dataset.T_extremes())
            image = unnormalize(image, dataset.T_extremes()) - 273.15
            vmin = np.minimum(np.min(image.numpy()), np.min(img))
            vmax = np.maximum(np.max(image.numpy()), np.max(img))
            ax[0].set_title(f"{args.attention}")
            ax[0].imshow(img)#, vmin = vmin, vmax = vmax)
            ax[1].imshow(image)#, vmin = vmin, vmax = vmax)
            ax[0].set_xticks([])
            ax[0].set_yticks([])
            ax[1].set_title(f"Ground Truth")
            ax[1].set_xticks([])
            ax[1].set_yticks([])
            cbar_ax = fig.add_axes([0.925, 0.15, 0.05, 0.7])
            fig.colorbar(ax[1].imshow(image), cax = cbar_ax, label = "Temperature (ºC)")
            cbar_ax.axhline(291.5 - 273.15, color = "k")
            
        else:
            continue
    
    

def sample_arrays(
        model : torch.nn.Module,
        diff_model : classmethod,
        dataset : torch.utils.data.Dataset,
        days : int,
        img_size : int,
        attention : str,
        args,
        ground_ : bool = False
) -> None:
    """
    Sample inferences and truth temperature maps and save them as arrays.

    :param model: PyTorch model
    :param diffusion: The diffusion model
    :param dataset: Set of training/test data
    :param days: Nº of days to sample
    :param img_size: Height or weight of the image (if image shape is is n x n)
    :param attention: str indicating the attention
    :param args: parsed arguments
    :param ground: boolean to indicate if targets will be sampled
    """
    maxim, minim = dataset.T_extremes()
    if ground_ == True:
        ground = np.zeros((days, img_size, img_size))
    samples = np.zeros((days, img_size, img_size))
    for idx, (image, cond) in tqdm(enumerate(dataset), total = days, desc = f"Sampling arrays..."):
            if idx <= days - 1:
                img = image[0]
                img = img.numpy()
                img = (img + 1)/2 * (maxim - minim) + minim
                xT = torch.randn(1, 1, img_size, img_size)
                x0 = diff_model.sample(xT, model, cond[None, ...])
                x0 = (x0[0, 0, :, :] + 1)/2 * (maxim - minim) + minim
                x0 = x0.numpy()
                if ground_ == True:
                    ground[idx, :, :] = img
                samples[idx, :, :] = x0
            else:
                continue
        

def inferences_singleidx(
        model : torch.nn.Module,
        diff_model : classmethod,
        dataset : torch.utils.data.Dataset,
        idx : int,
        n_samples : int,
        img_size : int,
        args
) -> None:
    """
    Sample inferences and truth temperature maps and save them as arrays.

    :param model: PyTorch model
    :param diffusion: The diffusion model
    :param dataset: Set of training/test data
    :param days: Nº of days to sample
    :param img_size: Height or weight of the image (if image shape is is n x n)
    :param attention: str indicating the attention
    :param args: parsed arguments
    :param ground: boolean to indicate if targets will be sampled
    """
    maxim, minim = dataset.T_extremes()
    
    samples = np.zeros((n_samples, img_size, img_size))
    noises = np.zeros((n_samples, img_size, img_size))
    for sample in tqdm(range(samples.shape[0]), total = n_samples, desc = f"Sampling idx {idx}..."):
        for idx_, (image, cond) in enumerate(dataset):
            if idx_ == idx:
                xT = torch.randn(1, 1, img_size, img_size)
                noises[sample, :, :] = xT.numpy()
                x0 = diff_model.sample(xT, model, cond[None, ...])
                x0 = unnormalize(x0, (maxim, minim))
                x0 = x0.numpy()
                samples[sample, :, :] = x0
            else:
                continue

    
    np.save(f"sampled_arrays/samples_i{args.island}_idx{idx}.npy", samples)
    np.save(f"sampled_arrays/noises_i{args.island}_idx{idx}.npy", noises)



def infere_random_indexes(
        model : torch.nn.Module, 
        diffusion : classmethod, 
        dataset : torch.utils.data.Dataset,
        img_size : int, 
        N : int
) -> None:
    """
    Infere N random indexes of the dataset.

    :param model: PyTorch model
    :param diffusion: The diffusion model
    :param img_size: Height or weight of the image (if image shape is is n x n)
    :param N: Nº of inferences
    """
    fig, ax = plt.subplots(N, 2)
    ax[0, 0].set_title("Inference")
    ax[0, 1].set_title("Ground Truth")
    for n in range(N):
        for idx, (image, cond) in enumerate(dataset):
            image = image[0]
            if idx == np.random.randint(0, len(dataset) - 1):
                xT = torch.randn(1, 1, img_size, img_size)
                x0 = diffusion.sample(xT, model, cond[None, ...])

                img = tensor_to_pil(x0, dataset.T_estremes())
                ax[n, 0].imshow(img)
                ax[n, 0].axis(False)
                ax[n, 1].imshow(image)
                ax[n, 1].axis(False)
                #xplt.colorbar(ax = ax[0])
            else:
                continue
    
def sample(
        model : torch.nn.Module, 
        diffusion : classmethod, 
        img_size : int,
        dataset : torch.utils.data.Dataset, 
        mask : np.ndarray,
        args,
        idx : int = None
) -> None:
    """
    Sample a single prediction for a random condition, if particular index not given.

    :param model: PyTorch model
    :param diffusion: The diffusion model
    :param img_size: Height or weight of the image (if image shape is is n x n)
    """
    if idx:
        try:
            assert idx <= len(dataset) - 1, "Index out of range"
        except AssertionError:
            raise
        
        print(f"Sampling {idx}th sample of the test set.....")
        infere_given_index(idx, model, diffusion, img_size, dataset, args, mask)
        plt.savefig(f"demonstrations/idx{idx}_b{args.batchsize}_e{args.epochs}_l{args.learningrate}_d{args.embdim}_t{args.timesteps}_a{args.attention}.png")
        print("Done! saved at demonstrations/")

    
    else:
        rand_idx = np.random.randint(0, len(dataset) - 1)
        print(f"Sampling {rand_idx}th sample of the test set.....")
        infere_given_index(rand_idx, model, diffusion, img_size, dataset, args, mask)
        plt.savefig(f"demonstrations/idx{rand_idx}_b{args.batchsize}_e{args.epochs}_l{args.learningrate}_d{args.embdim}_t{args.timesteps}_a{args.attention}.png")
        print("Done! saved at demonstrations/")
        return rand_idx



def see_sampling(
        model,
        diffusion,
        dataset,
        img_size,
        args,
        idx_ = None
):
    """
    Obtain T images of the backward process.

    :param model: PyTorch model
    :param diffusion: The diffusion model
    :param img_size: Height or weight of the image (if image shape is is n x n)
    _param idx_: Index of the dataset sample
    """
    if idx_ == None:
        idx_ = rand_idx = np.random.randint(0, len(dataset) - 1)

    for idx, (image, cond) in enumerate(dataset):
        if idx == idx_:
            xT = torch.randn(1, 1, img_size, img_size)
            x_dict = diffusion.sample(xT, model, cond[None, ...], saves = True)
    
    for t in range(len(x_dict)):
        if t%10 == 0 or t == 0:
            xt = x_dict[str(t)]
            xt = xt.detach().numpy()[0, 0]
            xt = unnormalize(xt, extremes = dataset.T_extremes())
            xt -= 273.15
            plt.imshow(xt, cmap = "jet")
            plt.axis(False)
            plt.savefig(f"inference/{args.island}{1000 - t}.png", transparent = True)
            plt.clf()
