# Stadistical Downscaling in Canary Islads.
Global Climate Models (GCM) has two major problems: coarse spatial resolution, and systematic biases. Major breaktrhoughs have been obtained with deep learning models. The aim of this work is to study the applicability of generative AI for stadistical downscaling in Canary Islands, in particular, a Denoising Diffusion Probabilistic Model (DDPM) has been implemented. 

This repository consist in 7 python scripts, and 4 folders, containing the model, results, and metrics of VALUE validation framework.

  * **TF.py**: Generate the  dataset, that contains $\approx 10.000$ samples of mean temperatures, and synoptic conditions of the selected island.
  * **process_data.py**: A utils script, containing usefull functions used along the rest of the code.
  * **diffusion.py**: The diffusion model.
  * **unet.py**: The UNet used to learn the noise of the diffusion.
  * **train.py**: Trains the model and save the weights (pre-trained weights aren't available).
  * **sample.py**: Infere samples from the model.
  * **stats.py**: Compute different metrics, according to the VALUE validation framework for stadistical downscaling.

The diffusion model, designed by Jonathan Ho, et al ([Paper](https://arxiv.org/abs/2006.11239)), consist on a generative AI model, with two steps. At first, there is a forward process, which, given a certain number of time steps $T$, adds a random gaussian noise at each step, until the distribution of the images is $\mathcal{N}(0, \mathbb{I})$. A UNet learns the noise added at each time step $t = 1, 2, ..., T$. The backward process generates a random image with distribution $\mathcal{N}(0, \mathbb{I})$, and substract the noise predicted at each time step, until $t = 0$, which gives the generated image.

![test](https://github.com/samuelmzc/SD_Tenerife/blob/main/metrics/sample.gif)

The UNet consist on blocks of double convolutions, along a positional encoding proposed in [Attention is all you need](https://arxiv.org/abs/1706.03762), Ashish vaswani, et al. The hyperparameters of the model can be found on the **process_data.py** script. The loss for 10, 50, 100 and 1000 epochs can be found at the *loss* folder.
