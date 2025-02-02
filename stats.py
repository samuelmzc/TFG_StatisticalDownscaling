import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import netCDF4


def Temperature_vs_Time(
        sample_, 
        ground_, 
        px_x, 
        px_y
):
    """
    Plot temperature versus time for given pixel with coordinates (px_x, px_y), and plots a sample with a star at (px_x, px_y).

    :param sample_: inferences with shape (nº days, img_size, img_size)
    :param ground_: ground with shape (nº days, img_size, img_size)
    :param px_x: x coordinate of pixel
    :param px_y: y coordinate of pixel
    """
    time = np.linspace(1, sample_.shape[0], sample_.shape[0])
    img_size = sample_.shape[2]
    fig, axs = plt.subplots(2, 1)
    sample_ij = sample_[:, px_x, px_y] - 273.15
    ground_ij = ground_[:, px_x, px_y] - 273.15
    axs[0].plot(time, ground_ij, "k.", label = "Value")
    axs[0].plot(time, sample_ij, "r.", label = "Sample")
    axs[0].set_xlabel("Time (days)")
    axs[0].set_ylabel("Temperature (ºC)")
    axs[0].grid()
    axs[0].legend()
    # axs[1].imshow(ground[285])
    axs[1].imshow(sample_[220])
    axs[1].axis(False)
    axs[1].plot(px_x, px_y, 'r*')
    plt.tight_layout()
    plt.savefig(f"metrics/Tvst.pdf")
    plt.clf()


def Temperature_vs_TimeMAX(
        sample_, 
        ground_
):
    """
    Plot maximum temperature versus time.

    :param sample_: inferences with shape (nº days, img_size, img_size)
    :param ground_: ground with shape (nº days, img_size, img_size)
    """
    time = np.linspace(1, sample_.shape[0], sample_.shape[0])
    img_size = sample_.shape[2]
    plt.plot(time, np.max(ground_, axis = (1, 2)) - 273.15, "k.", label = "Value")
    plt.plot(time, np.max(sample_, axis = (1, 2)) - 273.15, "r.", label = "Sample")
    plt.xlabel("Time (days)")
    plt.ylabel("Temperature (ºC)")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"metrics/TvstMAX.pdf")
    plt.clf()


def Temperature_vs_Time_full(
        sample_, 
        ground_, 
        mask
):
    """
    Same as Temperature_vs_time, but for all the points in Tenerife.

    :param sample_: inferences with shape (nº days, img_size, img_size)
    :param ground_: ground with shape (nº days, img_size, img_size)
    :param mask: mask with 0s at sea, and 1s in Tenerife with shape (img_size, img_size)
    """
    time = np.linspace(1, sample_.shape[0], sample_.shape[0])
    img_size = sample_.shape[2]
    count = 1
    for i in range(img_size):
        for j in range(img_size):
            px_x = i
            px_y = j
            if mask[px_x, px_y] != 0:
                fig, axs = plt.subplots(2, 1)
                sample_ij = sample_[:, px_x, px_y] - 273.15
                ground_ij = ground_[:, px_x, px_y] - 273.15
                axs[0].plot(time, ground_ij, label = "Value")
                axs[0].plot(time, sample_ij, "--",label = "Sample")
                axs[0].set_xlabel("Time (days)")
                axs[0].set_ylabel("Temperature (ºC)")
                axs[0].grid()
                axs[0].legend()
                axs[1].imshow(mask)
                axs[1].plot(px_y, px_x, 'r*')
                plt.tight_layout()
                plt.savefig(f"graphs/{count}.png")
                plt.clf()
                count += 1
            else:
                continue


def BIAS(
        sample, 
        ground, 
        mask = None
):
    """
    Bias obtained as the absolute difference of the means of the inferences and truth values.

    :param sample_: inferences with shape (nº days, img_size, img_size)
    :param ground_: ground with shape (nº days, img_size, img_size)
    :param mask: mask with 0s at sea, and 1s in Tenerife with shape (img_size, img_size)
    """
    g_mean = np.mean(ground, axis = 0)
    s_mean = np.mean(samples, axis = 0)
    bias = abs(g_mean - s_mean)
    if mask is not None:
        bias = np.ma.masked_where(mask == 0, bias)
    
    plt.imshow(bias, cmap = "plasma")
    plt.title("Bias")
    plt.colorbar(label = "Bias (K)")
    plt.tight_layout()
    plt.savefig("metrics/bias.pdf")
    plt.clf()


def BIAS_P2(
        sample, 
        ground, 
        mask = None
):
    """
    Bias for the 2nd percentile.
    
    :param sample_: inferences with shape (nº days, img_size, img_size)
    :param ground_: ground with shape (nº days, img_size, img_size)
    :param mask: mask with 0s at sea, and 1s in Tenerife with shape (img_size, img_size)
    """
    m = sample.shape[0]
    P2 = int(m * 0.02)
    sample = sample[:P2]
    ground = ground[:P2]
    g_mean = np.mean(ground, axis = 0)
    s_mean = np.mean(samples, axis = 0)
    biasP2 = abs(g_mean - s_mean)
    if mask is not None:
        biasP2 = np.ma.masked_where(mask == 0, biasP2)
    
    plt.imshow(biasP2, cmap = "plasma")
    plt.title("Bias P2")
    plt.colorbar(label = "Bias P2 (K)")
    plt.tight_layout()
    plt.savefig("metrics/biasP2.pdf")
    plt.clf()


def BIAS_P98(
        sample, 
        ground, 
        mask = None
):
    """
    Bias for the 98th percentile.
    
    :param sample_: inferences with shape (nº days, img_size, img_size)
    :param ground_: ground with shape (nº days, img_size, img_size)
    :param mask: mask with 0s at sea, and 1s in Tenerife with shape (img_size, img_size)
    """
    m = sample.shape[0]
    P98 = int(m * 0.98)
    sample = sample[:P98]
    ground = ground[:P98]
    g_mean = np.mean(ground, axis = 0)
    s_mean = np.mean(samples, axis = 0)
    biasP98 = abs(g_mean - s_mean)
    if mask is not None:
        biasP98 = np.ma.masked_where(mask == 0, biasP98)
    
    plt.imshow(biasP98, cmap = "plasma")
    plt.title("Bias P98")
    plt.colorbar(label = "Bias P98 (K)")
    plt.tight_layout()
    plt.savefig("metrics/biasP98.pdf")
    plt.clf()


def RMSE(
        sample, 
        ground, 
        mask = None
):
    """
    Root Mean Square Error.
    
    :param sample_: inferences with shape (nº days, img_size, img_size)
    :param ground_: ground with shape (nº days, img_size, img_size)
    :param mask: mask with 0s at sea, and 1s in Tenerife with shape (img_size, img_size)
    """
    m = sample.shape[0]
    rmse = (1 / m) * np.sqrt(np.sum((sample - ground)**2, axis = 0))

    if mask is not None:
        rmse = np.ma.masked_where(mask == 0, rmse)
    
    plt.imshow(rmse, cmap = "plasma")
    plt.title("RMSE")
    plt.colorbar(label = "RMSE (K)")
    plt.tight_layout()
    plt.savefig("metrics/rmse.pdf")
    plt.clf()


def PearsonCoeff(
        sample, 
        ground, 
        mask = None
):
    """
    Pearson coefficient (r) to study the correlation at each pixel between inference and ground truth.
    
    :param sample_: inferences with shape (nº days, img_size, img_size)
    :param ground_: ground with shape (nº days, img_size, img_size)
    :param mask: mask with 0s at sea, and 1s in Tenerife with shape (img_size, img_size)
    :return: pearson coefficient at each pixel
    """
    n = sample.shape[0]
    sample_sum = np.sum(sample, axis = 0)
    ground_sum = np.sum(ground, axis = 0)
    sample_sum2 = np.sum(sample**2, axis = 0)
    ground_sum2 = np.sum(ground**2, axis = 0)
    sampleground = np.sum(sample*ground, axis = 0)
    pearson = (n * sampleground - sample_sum * ground_sum) / (np.sqrt((n * sample_sum2 - sample_sum**2) * (n * ground_sum2 - ground_sum**2)))
    if mask is not None:
        pearson = np.ma.masked_where(mask == 0, pearson)
    
    plt.imshow(pearson, cmap = "plasma")
    plt.title("Pearson Coefficient")
    plt.colorbar(label = "Pearson Coefficient (K)")
    plt.tight_layout()
    plt.savefig("metrics/pearson.pdf")
    plt.clf()
    return pearson


def correlation(
        sample, 
        ground, 
        px_x, 
        py_y
):
    """
    Plot temperatures of the inference, and the ground truth to see correlation.
    
    :param sample_: inferences with shape (nº days, img_size, img_size)
    :param ground_: ground with shape (nº days, img_size, img_size)
    :param px_x: x coordinate of pixel
    :param px_y: y coordinate of pixel
    """
    sample_points = sample[:, px_x, px_y]
    ground_points = ground[:, px_x, px_y]
    r = PearsonCoeff(sample, ground)[px_x, px_y]
    plt.plot(sample_points, ground_points, '.k')
    plt.title(f"r² = {r**2} | r = {r}")
    plt.xlabel("Sample temperatures (K)")
    plt.ylabel("Ground temperatures (K)")
    plt.grid()
    plt.tight_layout()
    plt.savefig("metrics/correlation.pdf")
    plt.clf()


if __name__ == "__main__":
    ground = np.load("sampled_arrays/ground.npy")
    samples = np.load("sampled_arrays/samples.npy")
    m = samples.shape[0]
    px_x = 14
    px_y = 20

    mask_path = "netcdf/mask.nc"
    mask = netCDF4.Dataset(mask_path)["T2MEAN"]
    mask = mask[0, ::-1, :]


    # Temperature vs Time
    Temperature_vs_TimeMAX(samples, ground)

    # Bias
    BIAS(samples, ground, mask)

    # Bias P2
    BIAS_P2(samples, ground, mask)

    # Bias P98
    BIAS_P98(samples, ground, mask)

    # RMSE
    RMSE(samples, ground, mask)

    # Pearson coefficient
    PearsonCoeff(samples, ground, mask)

    # See correlation
    correlation(samples, ground, px_x, px_y)