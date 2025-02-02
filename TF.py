import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import transforms


# Load the dataset (netCDF4 file)
path = "netcdf/tenerife.nc"
path_condition = "netcdf/conditions.nc"
nc = netCDF4.Dataset(path)["T2MEAN"]
m_train = int(np.floor(0.8 * len(nc)))
m_test = len(nc) - m_train

times = []
for t in range(nc.shape[0]):
    img = nc[t, ::-1, :]
    times.append(img[..., None])


# Load conditions
condition_names = ["q", "t", "u", "v", "z"]
conds = []
for condition in condition_names:
    cond = np.array(netCDF4.Dataset(path_condition)[condition])
    cond = np.swapaxes(cond, 1, 3)
    cond = np.swapaxes(cond, 1, 2)
    conds.append(cond)

# Concatenate conditions to a total input array
all_conds = np.concatenate(conds, axis = -1) if len(conds) != 1 else conds[0] 
total_input = np.concatenate((times, all_conds), axis = -1)

# Compute the maximum and minimum value for each channel (Temperature channel + 20 condition channels)
# & normalize
maxs = []
mins = []
for c in range(total_input.shape[-1]):
    maximum, minimum = np.max(total_input[:, :, :, c]), np.min(total_input[:, :, :, c])
    maxs.append(maximum)
    mins.append(minimum)
    total_input[:, :, :, c] = (total_input[:, :, :, c] - minimum)/(maximum - minimum) * 2 - 1

def T_max_min():
    """
    Save the max and min values for temperature to unnormalize in other scripts.

    :return: Tuple with maximum and minimum value of the temperature
    """
    return (maxs[0], mins[0])


times = total_input[:, :, :, 0]
all_conds = total_input[:, :, :, 1:]


class TenerifeTempDataset(torch.utils.data.Dataset):
    """
    Dataset with the temperatures of Tenerife at ground level, and the synoptic conditions at 4 different pressure levels (1000, 850, 700, 500) hPa according to VALUE:

    q -> Humidity
    t -> Temperature
    u -> Meridional wind
    v -> Zonal wind
    z -> Geopotential height

    :param transformation: Transformations applyed to the temperatures at ground level (input)
    :param cond_transform: Transformation applyed to the synoptic conditions
    :param train: Boolean. True for train set, and False for dev/test set
    :param conditions: Synoptic conditions    
    """
    def __init__(
            self, 
            transformation, 
            cond_transform, 
            train, conditions = all_conds
    ):
        self.transform = transformation
        self.cond_transform = cond_transform

        # Split between train and test set
        if train == True:
            self.list = times[:m_train]
            self.conds = conditions[:m_train]
        else:
            self.list = times[m_train:]
            self.conds = conditions[m_train:]

    def __len__(
            self
    ):
        """
        Obtain the lenght of the dataset (nº of samples)

        :return: Nº of samples
        """
        return len(self.list)
    
    def __getitem__(
            self, 
            idx
    ):
        """
        Get the input and the conditions of given index

        :param idx: Index of the set
        :return: Tuple with input and conditions
        """
        return (
            self.transform(self.list[idx]), 
            self.transform(self.conds[idx])
            )
    
    def shape(
            self
    ):
        """
        Shape of the images.

        :return: Tuple with the shape of the input
        """
        return self.transform(self.list[0]).shape


