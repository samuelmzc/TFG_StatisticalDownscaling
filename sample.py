import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch as torch
import torchvision as tv

from diffusion import *
from process_data import *
from unet import *

img_size = 32

# Hyperparameters
batch_size, T, epochs, learning_rate, _, condition = hyperparams()
dataset, _ = set_and_loader(batch_size, train = False)

# Load model
model = UNet()
diff_model = diffusion_cosine(timesteps = T)

# Load weights
path = "weights/"
name = "UNet_diff.pth"
model.load_state_dict(torch.load(path + name, weights_only=True))

mask_path = "netcdf/mask.nc"
mask = netCDF4.Dataset(mask_path)["T2MEAN"]
mask = mask[0, ::-1, :]


def infere_given_index(
        idx_, 
        model = model, 
        diffusion = diff_model, 
        img_size = img_size
):
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
            fig, ax = plt.subplots(1, 2)
            xT = torch.randn(1, 1, img_size, img_size)
            x0 = diffusion.sample(xT, model, cond[None, ...])
            T_mean_infer = unnormalize(x0[0, 0, 0, 0])
            T_mean_truth = unnormalize(image[0, 0])
            img = tensor_to_pil(x0)
            ax[0].set_title(r"Inference | $\bar{T} \approx %2f$" %T_mean_infer)
            ax[0].imshow(np.ma.masked_where(mask == 0, img))
            ax[0].axis(False)
            ax[1].set_title(r"Ground Truth | $\bar{T} \approx %2f$" %T_mean_truth)
            ax[1].imshow(np.ma.masked_where(mask == 0, image))
            ax[1].axis(False)
            #xplt.colorbar(ax = ax[0])
        else:
            continue
    

def sample_arrays(
        days = len(dataset)
):
    """
    Sample inferences and truth temperature maps and save them as arrays.

    :param days: Nº of days to sample, len(dataset) as default value
    """
    maxim, minim = T_max_min()
    ground = np.zeros((days, img_size, img_size))
    samples = np.zeros((days, img_size, img_size))
    for idx, (image, cond) in tqdm(enumerate(dataset), total = days, desc = "Sampling arrays..."):
            if idx <= days - 1:
                img = image[0]
                img = img.numpy()
                img = (img + 1)/2 * (maxim - minim) + minim
                xT = torch.randn(1, 1, img_size, img_size)
                x0 = diff_model.sample(xT, model, cond[None, ...])
                x0 = (x0[0, 0, :, :] + 1)/2 * (maxim - minim) + minim
                x0 = x0.numpy()
                ground[idx, :, :] = img
                samples[idx, :, :] = x0
            else:
                continue
    
    np.save("sampled_arrays/ground.npy", ground)
    np.save("sampled_arrays/samples.npy", samples)


def infere_random_indexes(
        model, 
        diffusion, 
        img_size, 
        N
):
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
                img = tensor_to_pil(x0)
                ax[n, 0].imshow(img)
                ax[n, 0].axis(False)
                ax[n, 1].imshow(image)
                ax[n, 1].axis(False)
                #xplt.colorbar(ax = ax[0])
            else:
                continue
    
def sample(
        model, 
        diffusion, 
        img_size = img_size, 
        idx = None
):
    """
    Sample a single prediction for a random condition, if particular index not given.

    :param model: PyTorch model
    :param diffusion: The diffusion model
    :param img_size: Height or weight of the image (if image shape is is n x n)
    """
    if idx:
        infere_given_index(idx, model, diffusion, img_size)
    
    else:
        rand_idx = np.random.randint(0, len(dataset) - 1)
        infere_given_index(rand_idx, model, diffusion, img_size)


if __name__ == "__main__":
    sample_arrays()
    

    
