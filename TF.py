import netCDF4
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import transforms


def get_mask(
    nc_file : str
) -> None:
    ds = xr.open_dataset(nc_file)
    data = ds["T2MEAN"][:, ::-1, :]
    value_remove = data[0, 0].values
    mask = data[1] - data[0]
    ds["MASK"] = mask
    ds.to_netcdf(nc_file[:-3] + "_mask.nc")
    return mask, value_remove


def prepare_data(
        island_path : str,
        conds_path : str,
        conds_names : str = ["q", "t", "u", "v", "z"],
        variable_name : str = "T2MEAN",
        split_percent : float = 80
) -> tuple:
    """
    Prepares the data from the netcdf directory.

    :param island path: source path of island's .nc
    :param conds_path: source path of synoptic conditions's .nc
    :param conds_names: variable names from condition's .nc
    :param variable_name: variable name from island's .nc
    :param split_percent: percentage to do train/test splitting
    :return: tuple (Temperatures, Conditions, nº training samples, (T_max, T_min))
    """
    nc = netCDF4.Dataset(island_path)[variable_name]
    m_train = int(np.floor(split_percent * len(nc) / 100))
    m_test = len(nc) - m_train

    # Prepare temperatures
    T = []
    for t in range(nc.shape[0]):
        img = nc[t, ::-1, :]
        T.append(img[..., None])
    
    # Prepare conditions
    conds = []
    for condition in conds_names:
        cond = np.array(netCDF4.Dataset(conds_path)[condition])
        cond = np.swapaxes(cond, 1, 3)
        cond = np.swapaxes(cond, 1, 2)
        conds.append(cond)

    conds = np.concatenate(conds, axis = -1) if len(conds) != 1 else conds[0] 
    T_conds = np.concatenate((T, conds), axis = -1)

    # Normalize
    maxs, mins = [], []
    for c in range(T_conds.shape[-1]):
        maxim, minim = np.max(T_conds[:, :, :, c]), np.min(T_conds[:, :, :, c])
        maxs.append(maxim)
        mins.append(minim)
        T_conds[:, :, :, c] = (T_conds[:, :, :, c] - minim)/(maxim - minim) * 2 - 1
        
    
    T = T_conds[:, :, :, 0]
    conds = T_conds[:, :, :, 1:]
    return (T, conds, m_train, (maxs[0], mins[0]))


class IslandTempDataset(torch.utils.data.Dataset):
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
            path,
            transformation,
            train,
            conds_path = "netcdf/conditions.nc",
            split_percent = 80
    ):
        self.T, self.conds, self.m, (self.max_T, self.min_T) = prepare_data(island_path = path, conds_path = conds_path, split_percent = split_percent)
        self.transform = transformation

        # Split between train and test set
        if train == True:
            self.T = self.T[:self.m]
            self.conds = self.conds[:self.m]
        else:
            self.T = self.T[self.m:]
            self.conds = self.conds[self.m:]

    def __len__(
            self
    ):
        """
        Obtain the lenght of the dataset (nº of samples)

        :return: Nº of samples
        """
        return len(self.T)
    
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
            self.transform(self.T[idx]) if self.transform is not None else self.T[idx], 
            self.transform(self.conds[idx] if self.transform is not None else self.conds[idx])
            )
    
    def shape(
            self
    ):
        """
        Shape of the images.

        :return: Tuple with the shape of the input
        """
        return self.transform(self.T[0]).shape
    
    def show_sample(
            self,
            rows,
            columns
    ):
        fig, axs = plt.subplots(rows, columns)
        for ax in axs.flatten():
            random_idx = np.random.randint(0, len(self.T))
            ax.imshow(self.T[random_idx])
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.savefig(f"demonstrations/{rows}_{columns}_samples.png")
        plt.clf()
    
    def T_extremes(
            self
    ):
        return (self.max_T, self.min_T)


if __name__ == "__main__":
    ids = IslandTempDataset(path = "./netcdf/tf.nc", transformation = transforms.ToTensor(), train = False)
    print(len(ids))



        


