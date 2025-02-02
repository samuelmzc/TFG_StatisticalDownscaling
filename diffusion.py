import numpy as np
import matplotlib.pyplot as plt
import torch as torch
from process_data import tensor_to_pil

# Convert shape of shape (T), to shape (T, 1, 1, 1) for broadcasting purposes
unsqueeze3x = lambda x : x[..., None, None, None] 

class diffusion_cosine:
    """
    A diffusion model with cosine noise scheduler (https://arxiv.org/abs/2102.09672). Includes the forward and the backward (sampling) process.

    :param timesteps: Total nº of timesteps
    """
    def __init__(
            self, 
            timesteps
    ):
        self.timesteps = timesteps
        self.alpha_bar_schedule = (
            lambda t: np.cos((t/timesteps + 0.008)/(1 + 0.008) * np.pi/2)**2
        )
        diff_params = self.params(self.alpha_bar_schedule)
        self.beta, self.alpha, self.alpha_bar = diff_params["betas"], diff_params["alphas"], diff_params["alphas_bar"]
        self.beta_tilde = self.beta[1:] * (
            (1 - self.alpha_bar[:-1])
            /
            (1 - self.alpha_bar[1:])
        )
        self.beta_tilde = torch.cat(
            [self.beta_tilde[0:1], self.beta_tilde]
        )

    def params(
            self, 
            scheduler
    ):
        """
        Obtain the parameters of the diffusion model.

        β -> Noise scheduler parameter
        α -> 1 - β
        α_bar -> Cummulative product of α

        :param scheduler: Noise scheduler
        :return: Dictionary with parameters
        """
        diff_params = {}
        diff_params["betas"] = torch.from_numpy(
            np.array(
                [
                    min(
                        1 - scheduler(t + 1)/scheduler(t),
                        0.999,
                    )
                    for t in range(self.timesteps)
                ]
            )
        )
        diff_params["alphas"] = 1 - diff_params["betas"]
        diff_params["alphas_bar"] = torch.cumprod(diff_params["alphas"], dim = 0)
        return diff_params

    def forward(
            self, 
            x0, 
            t
    ):
        """
        Diffusion's forward process, i.e, add noise to input for certain timestep t

        :param x0: Input tensor
        :param t: timestep
        :return: xT, output tensor with noise added
        """
        noise = torch.randn_like(x0)
        xt = (
            unsqueeze3x(torch.sqrt(self.alpha_bar[t])) * x0
            +
            unsqueeze3x(torch.sqrt(1 - self.alpha_bar[t])) * noise
        )
        return xt.float(), noise

    def sample(
            self, 
            xT, 
            model, 
            cond
    ):
        """
        Sample process of the diffusion.

        :param xT: Random noise input tensor
        :param model: PyTorch model
        :param cond: Tensor with synoptic conditions
        :param timesteps:
        """
        model.eval()
        timesteps = self.timesteps
        sub_timesteps = np.linspace(0, timesteps - 1, timesteps)
        xt = xT
        
        for i, t in zip(np.arange(timesteps)[::-1], sub_timesteps[::-1]):
            with torch.no_grad():
                current_t = torch.full((1,), t)
                current_t_indexed = torch.full((1,), i)
                noise_pred = model.forward(xt, current_t, cond)

                mean = (
                    1
                    /
                    unsqueeze3x(self.alpha[current_t_indexed].sqrt())
                    *
                    (xt - (
                        unsqueeze3x(self.beta[current_t_indexed])
                        /
                        unsqueeze3x((1 - self.alpha_bar[current_t_indexed]).sqrt())
                    ) * noise_pred)
                )
        
                if i == 0:
                    xt = mean
                else:
                    xt = mean + unsqueeze3x(self.beta_tilde[current_t_indexed].sqrt()) * torch.randn_like(xt)
                
                xt = xt.float()
        return xt.detach()
