import torch as torch
from torch import nn
import torchvision as tv

from process_data import hyperparams

_, _, _, _, embedding_dim, _ = hyperparams()

class PositionalEmbedding():
    """
    Class to obtain the positional embedding vector for given t.

    :param dim: Dimension of the embedding
    :param theta: Parameter of the positional encoding
    """
    def __init__(
            self, 
            dim, 
            theta = torch.tensor(10000)
    ):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(
            self, 
            t
    ):
        """
        Compute the positional encoding.

        :param t: Tensor with shape (batch_size) with different timesteps
        :return: Tensor with the embeddings
        """
        half_dim = self.dim//2
        embedding = torch.log(self.theta)/(half_dim)
        embedding = torch.exp(- torch.arange(half_dim) * embedding)
        embedding = t[:, None] * embedding[None, :]
        embedding = torch.cat((embedding.sin(), embedding.cos()), dim = -1)
        return embedding


class Zpool(nn.Module):
    """
    Class to apply a Zpool for the Triplet Attention. Consist of the concatenation of a MaxPool and a AveragePool for the 0th dimension (channel) not including batch.
    """
    def forward(
            self, 
            x
    ):
        """
        Compute the Zpool.

        :param x: Input of shape (b, c, h, w)
        :return: Tensor of shape (b, 2, h, w)
        """
        maxpool0d = torch.max(x, 1)[0].unsqueeze(1) # Nota, sobre lo que hacemos el pool es sobre canales, y como tenemos batch size, entonces en vez de la dim 0 es la 1
        avgpool0d = torch.mean(x, 1).unsqueeze(1)
        return torch.cat((maxpool0d, avgpool0d), dim = 1) # shape = (2, H, W)


class Gate(nn.Module):
    """
    Gate of the Triplet Attention. Permutate the input, apply a Zpool, then a 2D convolution, permute again and apply sigmoid.

    :param permutation: Tuple indicating the permutation
    """
    def __init__(
            self, 
            permutation
    ):
        self.perm = permutation
        self.zpool = Zpool()
        self.conv = nn.Conv2d(in_channels = 2, out_channels = 1, kernel_size = 4)
        self.sigmoid = nn.Sigmoid()

    def forward(
            self, 
            x
    ):
        """
        Compute the gate.

        :param x: Input tensor
        :return: Output tensor weighted with attention
        """
        x0 = torch.permute(x, self.perm)
        x = self.zpool(x)
        x = self.conv(x)
        x = self.sigmoid()
        x *= x0
        x = torch.permute(x, self.perm)
        return x
    

class TripletAttention(nn.Module):
    """
    Triplet Attention used for images, inspired by https://arxiv.org/abs/2010.03045v2.
    """
    def __init__(
            self
    ):
        super().__init__()
        self.gate1 = Gate((0, 3, 2, 1))
        self.gate2 = Gate((0, 2, 1, 3))
        self.gate3 = Gate((0, 1, 2, 3))
        

    def forward(
            self, 
            x
    ):
        """
        Compute the Triplet Attention.

        :param x: Input tensor
        :return: The triplet attention
        """
        x_1 = self.gate1(x)
        x_2 = self.gate2(x)
        x_3 = self.gate3(x)
        mean = (1 / 3) * (x_1 + x_2 + x_3)
        return mean


class ResNetBlock(nn.Module):
    """
    Blocks of the UNet. Consist in a double convolution, adding the time embedding after the first one, with
    residual connections between the input and output for backpropagation optimization.

    :param input_channels: Nº of channels of the input
    :param output_channels: Nº of channels of the output
    :param time_embedding_dim: Dimension of the time embedding
    """
    
    def __init__(
            self,
            input_channels,
            output_channels,
            time_embedding_dim
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size = 3, padding = 1)
        self.convres = nn.Conv2d(input_channels, output_channels, kernel_size = 1)
        self.norm1 = nn.BatchNorm2d(output_channels)
        self.norm2 = nn.BatchNorm2d(output_channels)
        self.normres = nn.BatchNorm2d(output_channels)
        self.silu = nn.SiLU()
        self.mlp = nn.Linear(time_embedding_dim, 2 * output_channels)


    def forward(
            self, 
            x, 
            time_embedding
    ):
        """
        Propagate trough a ResNet Block.

        :param x: Tensor input
        :param time_embedding: timestep after time embedding applyed
        :return: Tensor output
        """
        time_embedding = self.mlp(time_embedding)
        time_embedding = time_embedding[(...,) + (None,) * 2] # b c -> b c 1 1
        scale, shift = torch.chunk(time_embedding, 2, dim = 1)

        x_res = self.convres(x)
        x_res = self.normres(x_res)

        x = self.conv1(x)
        x = self.norm1(x)
        x = x * (scale + 1) + shift
        x = self.silu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x += x_res
        x = self.silu(x)        
        return x


class UNet(nn.Module):
    """
    Model of the downscaling, that learns the noise added at certain step t. UNet consisting of ResNet blocks, with conditions concatenated at the beginning.

    :param input_channels: Nº of channels of the input
    :param input_channels: Nº of classes / channels at the output
    :param time_embedding_dim: Dimension of the time embedding
    """
    def __init__(
            self,
            input_channels = 1,
            output_channels = 1,
            time_embedding_dim = embedding_dim
    ):
        super().__init__()

        # Prepare channels
        condition_channels = 20
        channels = [2**(5 + i) for i in range(5)]       
        input_channels += condition_channels

        # Layers
        self.downs = nn.ModuleList()
        self.upstconvs = nn.ModuleList()
        self.upsblock = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)

        # Encoder
        for channel in channels:
            self.downs.append(ResNetBlock(input_channels, channel, time_embedding_dim))
            input_channels = channel

        # Bottleneck
        self.bottleneck = ResNetBlock(channels[-1], 2 * channels[-1], time_embedding_dim)

        # Decoder
        for channel in reversed(channels):
            self.upstconvs.append(nn.Sequential(
                nn.Upsample(scale_factor = 2, mode = "nearest"),
                nn.Conv2d(2 * channel, channel, kernel_size = 3, padding = 1)
            ))
            self.upsblock.append(ResNetBlock(2 * channel, channel, time_embedding_dim))

        # Last convolution
        self.last_conv = nn.Conv2d(channels[0], output_channels, kernel_size = 1, stride = 1, padding = 0)

        # time embedding
        self.embedding = PositionalEmbedding(dim = time_embedding_dim)
        self.emb_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim)
        )

    def forward(
            self, 
            x, 
            time, 
            condition
    ):
        """
        Propagate the input with the noise at timestep t added, trought the UNet.

        :param x: Input with the added noise at timestep t.
        :param time: timestep
        :param condition: Tensor with synoptic conditions
        :return: Noise predicted by the network
        """
        embedding = self.emb_mlp(self.embedding.forward(time))
        x = torch.cat((x, condition), dim = 1)
        residual = []
        for down in self.downs:
            x = down(x, embedding)
            residual.append(x)
            x = self.pool(x)

        x = self.bottleneck(x, embedding)

        for tconv, block in zip(self.upstconvs, self.upsblock):
            x = tconv(x)
            x = torch.cat((x, residual.pop()), dim = 1)
            x = block(x, embedding)

        x = self.last_conv(x)

        return x
