import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image
import netCDF4
from process_data import *


def Temperature_vs_Time(
        sample, 
        ground, 
        args,
        px_x, 
        px_y,
        save = False
):
    """
    Plot temperature versus time for given pixel with coordinates (px_x, px_y), and plots a sample with a star at (px_x, px_y).

    :param sample: inferences with shape (nº days, img_size, img_size)
    :param ground: ground with shape (nº days, img_size, img_size)
    :param args: parsed arguments
    :param px_x: x coordinate of pixel
    :param px_y: y coordinate of pixel
    """
    time = np.linspace(1, sample.shape[0], sample.shape[0])
    img_size = sample.shape[2]
    fig, axs = plt.subplots(2, 1)
    sample_ij = sample[:, px_x, px_y] - 273.15
    ground_ij = ground[:, px_x, px_y] - 273.15
    axs[0].plot(time, ground_ij, "k.", label = "Value")
    axs[0].plot(time, sample_ij, "r.", label = "Sample")
    axs[0].set_xlabel("Time (days)")
    axs[0].set_ylabel("Temperature (ºC)")
    axs[0].grid()
    axs[0].legend()
    # axs[1].imshow(ground[285])
    axs[1].imshow(sample[220])
    axs[1].axis(False)
    axs[1].plot(px_x, px_y, 'r*')
    plt.tight_layout()
    if save == True:
        plt.savefig(f"metrics/deff/Tvst_b{args.batchsize}_e{args.epochs}_l{args.learningrate}_d{args.embdim}_t{args.timesteps}_a{args.attention}.pdf")
    plt.clf()


def Temperature_vs_TimeMAX(
        sample, 
        ground,
        args,
        save = False
):
    """
    Plot maximum temperature versus time.

    :param sample: inferences with shape (nº days, img_size, img_size)
    :param ground: ground with shape (nº days, img_size, img_size)
    :param args: parsed arguments
    """
    time = np.linspace(1, sample.shape[0], sample.shape[0])
    img_size = sample.shape[2]
    plt.plot(time, np.max(ground, axis = (1, 2)) - 273.15, "k.", label = "Value")
    plt.plot(time, np.max(sample, axis = (1, 2)) - 273.15, "r.", label = "Sample")
    plt.xlabel("Time (days)")
    plt.ylabel("Maximum temperature (ºC)")
    plt.legend()
    plt.tight_layout()
    if save == True:
        plt.savefig(f"metrics/deff/{args.island.upper()}TvstMAX_b{args.batchsize}_e{args.epochs}_l{args.learningrate}_d{args.embdim}_t{args.timesteps}_a{args.attention}.pdf")
    plt.clf()


def Temperature_vs_Time_full(
        sample_, 
        ground_, 
        args,
        mask,
        save = False
):
    """
    Same as Temperature_vs_time, but for all the points in Tenerife.

    :param sample_: inferences with shape (nº days, img_size, img_size)
    :param ground_: ground with shape (nº days, img_size, img_size)
    :param args: parsed arguments
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
                plt.savefig(f"graphs/{args.attention}/{count}_b{args.batchsize}_e{args.epochs}_l{args.learningrate}_d{args.embdim}_t{args.timesteps}_a{args.attention}.png")
                plt.clf()
                count += 1
            else:
                continue


def BIAS(
        sample, 
        ground, 
        args,
        mask = None,
        save = False
):
    """
    Bias obtained as the absolute difference of the means of the inferences and truth values.

    :param sample: inferences with shape (nº days, img_size, img_size)
    :param ground: ground with shape (nº days, img_size, img_size)
    :param args: parsed arguments
    :param mask: mask with 0s at sea, and 1s in Tenerife with shape (img_size, img_size)
    :return: Bias at each pixel
    """
    g_mean = np.mean(ground, axis = 0)
    s_mean = np.mean(sample, axis = 0)
    bias_ = abs(g_mean - s_mean)
    if mask is not None:
        bias = np.ma.masked_where(mask == 0, bias_)
        mean = np.mean(bias, axis = (0, 1))
        std = np.std(bias, axis = (0, 1))
        plt.text(x = 1, y = 8, s = r"$%.2f \pm %.2f$" %(mean, 2 * std))
        
    plt.imshow(bias, cmap = "plasma")
    plt.title("Bias")
    plt.colorbar(label = "Bias (K)")
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    if save == True:
        plt.savefig(f"metrics/deff/{args.island.upper()}bias_b{args.batchsize}_e{args.epochs}_l{args.learningrate}_d{args.embdim}_t{args.timesteps}_a{args.attention}.pdf")
        np.save(f"./arrays/bias{args.island}.npy", bias_)
        
    plt.clf()
    return bias


def BIAS_P(
        P,
        sample, 
        ground, 
        args,
        mask = None,
        save = False
):
    """
    Bias for the P'th percentile.
    
    :param sample: inferences with shape (nº days, img_size, img_size)
    :param ground: ground with shape (nº days, img_size, img_size)
    :param args: parsed arguments
    :param mask: mask with 0s at sea, and 1s in Tenerife with shape (img_size, img_size)
    :return: Bias P2 for each pixel
    """
    m = sample.shape[0]
    sample_percentile = np.percentile(a = sample, q = P, axis = 0)
    ground_percentile = np.percentile(a = ground,  q = P, axis = 0)
    biasP_ = abs(sample_percentile - ground_percentile)
    if mask is not None:
        biasP = np.ma.masked_where(mask == 0, biasP_)
        mean = np.mean(biasP, axis = (0, 1))
        std = np.std(biasP, axis = (0, 1))
        plt.text(x = 1, y = 8, s = r"$%.2f \pm %.2f$" %(mean, 2 * std))

    plt.imshow(biasP, cmap = "plasma")
    plt.title(f"Bias P{P}")
    plt.colorbar(label = f"Bias P{P} (K)")
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    if save == True:
        plt.savefig(f"metrics/deff/{args.island.upper()}biasP{P}_b{args.batchsize}_e{args.epochs}_l{args.learningrate}_d{args.embdim}_t{args.timesteps}_a{args.attention}.pdf")
        np.save(f"./arrays/biasp{P}{args.island}.npy", biasP_)
    plt.clf()
    return biasP


def RMSE(
        sample, 
        ground, 
        args,
        mask = None,
        save = False
):
    """
    Root Mean Square Error.
    
    :param sample: inferences with shape (nº days, img_size, img_size)
    :param ground: ground with shape (nº days, img_size, img_size)
    :param args: parsed arguments
    :param mask: mask with 0s at sea, and 1s in Tenerife with shape (img_size, img_size)
    :return: RMSE for each pixel
    """
    m = sample.shape[0]
    rmse_ = np.sqrt((1 / m) * np.sum((sample - ground)**2, axis = 0))
    if mask is not None:
        rmse = np.ma.masked_where(mask == 0, rmse_)
        mean = np.mean(rmse, axis = (0, 1))
        std = np.std(rmse, axis = (0, 1))
        plt.text(x = 1, y = 8, s = r"$%.2f \pm %.2f$" %(mean, 2 * std))
    
    plt.imshow(rmse, cmap = "plasma")
    plt.title("RMSE")
    plt.colorbar(label = "RMSE (K)")
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    if save == True:
        plt.savefig(f"metrics/deff/{args.island.upper()}rmse_b{args.batchsize}_e{args.epochs}_l{args.learningrate}_d{args.embdim}_t{args.timesteps}_a{args.attention}.pdf")
        np.save(f"./arrays/RMSE{args.island}.npy", rmse_)
    plt.clf()
    return rmse


def std_ratio(
        sample,
        ground,
        args,
        mask = None,
        save = False
):
    """
    Ratio of standart deviations (predictions / observation). 

    :param sample: inferences with shape (nº days, img_size, img_size)
    :param ground: ground with shape (nº days, img_size, img_size)
    :param args: parsed arguments
    :param mask: mask with 0s at sea, and 1s in Tenerife with shape (img_size, img_size)
    :return: STD ratio for each pixel
    """
    sample_std = np.std(sample, axis = 0)
    ground_std = np.std(ground, axis = 0)
    ratio_ = sample_std / (ground_std + 1e-5)
    if mask is not None:
        ratio = np.ma.masked_where(mask == 0, ratio_)
        mean = np.mean(ratio, axis = (0, 1))
        std = np.std(ratio, axis = (0, 1))
        plt.text(x = 1, y = 8, s = r"$%.2f \pm %.2f$" %(mean, 2 * std))
    
    plt.imshow(ratio, cmap = "plasma")
    plt.title("Ratio of Standart Deviations")
    plt.colorbar(label = r"$\frac{\sigma_{pred}}{\sigma_{obs}}$")
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    if save == True:
        plt.savefig(f"metrics/deff/{args.island.upper()}std_ratio_b{args.batchsize}_e{args.epochs}_l{args.learningrate}_d{args.embdim}_t{args.timesteps}_a{args.attention}.pdf")
        np.save(f"./arrays/stdratio{args.island}.npy", ratio_)
    plt.clf()
    return ratio
    

def PearsonCoeff(
        sample, 
        ground, 
        args,
        mask = None,
        save = False
):
    """
    Pearson coefficient (r) to study the correlation at each pixel between inference and ground truth.
    
    :param sample: inferences with shape (nº days, img_size, img_size)
    :param ground: ground with shape (nº days, img_size, img_size)
    :param args: parsed arguments
    :param mask: mask with 0s at sea, and 1s in Tenerife with shape (img_size, img_size)
    :return: pearson coefficient at each pixel
    """
    n = sample.shape[0]
    sample_sum = np.sum(sample, axis = 0)
    ground_sum = np.sum(ground, axis = 0)
    sample_sum2 = np.sum(sample**2, axis = 0)
    ground_sum2 = np.sum(ground**2, axis = 0)
    sampleground = np.sum(sample*ground, axis = 0)
    pearson_ = (n * sampleground - sample_sum * ground_sum) / (np.sqrt((n * sample_sum2 - sample_sum**2) * (n * ground_sum2 - ground_sum**2)) + 1e-8)
    if mask is not None:
        pearson = np.ma.masked_where(mask == 0, pearson_)
        mean = np.mean(np.ma.masked_where((mask) == 0, pearson), axis = (0, 1))
        std = np.std(np.ma.masked_where(mask == 0, pearson), axis = (0, 1))
        plt.text(x = 1, y = 8, s = r"$%.2f \pm %.2f$" %(mean, 2 * std))
    else:
        pearson = pearson_
    plt.imshow(pearson, cmap = "plasma")
    plt.title("Pearson Coefficient")
    plt.colorbar(label = "Pearson Coefficient")
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    if save == True:
        plt.savefig(f"metrics/deff/{args.island.upper()}pearson_b{args.batchsize}_e{args.epochs}_l{args.learningrate}_d{args.embdim}_t{args.timesteps}_a{args.attention}.pdf")
        np.save(f"./arrays/pearson{args.island}.npy", pearson_)
    plt.clf()
    return pearson


def correlation(
        sample, 
        ground, 
        px_x, 
        px_y,
        args,
        save = False
):
    """
    Plot temperatures of the inference, and the ground truth to see correlation.
    
    :param sample: inferences with shape (nº days, img_size, img_size)
    :param ground: ground with shape (nº days, img_size, img_size)
    :param px_x: x coordinate of pixel
    :param px_y: y coordinate of pixel
    :param args: parsed arguments
    """
    sample_points = sample[:, px_y, px_x]
    ground_points = ground[:, px_y, px_x]
    pearson = PearsonCoeff(sample, ground, args)
    r = pearson[px_y, px_x]
    temperatures = np.linspace(min(sample_points), max(sample_points), len(sample_points))
    
    fig, ax1 = plt.subplots(figsize = (6, 6))
    plt.rcParams['font.size'] = 10
    plt.rcParams['text.color'] = "w"
    ax2 = fig.add_axes([0.15, 0.65, 0.2, 0.2])
    ax1.plot(sample_points, ground_points, 'ow', alpha = 0.3)
    ax1.plot(temperatures, temperatures, "--", color = "gray")
    ax1.tick_params(color = "w", labelcolor = "w")
    ax2.imshow(ground[220], cmap = "jet")
    ax2.plot(px_x, px_y, "w*")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax1.set_title(f"r = {r:.2f} | r² = {r**2:.2f}", color = "w")
    ax1.set_xlabel("Predicted temperatures (K)", color = "w")
    ax1.set_ylabel("WRF temperatures (K)", color = "w")
    for spine1, spine2 in zip(ax1.spines.values(), ax2.spines.values()):
        spine1.set_edgecolor('w')
        spine2.set_edgecolor("w")

    if save == True:
        plt.savefig(f"metrics/deff/{args.island.upper()}correlation_b{args.batchsize}_e{args.epochs}_l{args.learningrate}_d{args.embdim}_t{args.timesteps}_a{args.attention}.png", transparent = True)
    plt.clf()


def idx_hist(
    idx,
    island,
    bins,
    px_x,
    px_y
):
    try:
        array_path = f"./sampled_arrays/samples_i{island}_idx{idx}.npy"
        assert os.path.exists(array_path) == True
        samples = np.load(array_path)
        ground = np.load(f"./sampled_arrays/ground_i{island}.npy")
        data = samples[:, px_y, px_x]

        ground_value = ground[idx, px_y, px_x]
        data_mean = np.mean(data)

        fig, ax = plt.subplots()
        for spine1 in ax.spines.values():
            spine1.set_edgecolor('w')

        plt.rcParams.update({'text.color': "w", 'axes.labelcolor': "w"})
        ax.tick_params(axis='x', colors='w')
        ax.tick_params(axis='y', colors='w')
        ax.hist(data, bins = bins, color = "w")
        ax.axvline(x = ground_value, color = "g", label = "WRF simulation")
        ax.axvline(x = data_mean, color = "r", label = "Distribution mean")
        plt.xlim(273, 300)
        ax.set_xlabel("Temperature (K)", color = "w")
        ax.set_ylabel("Counts", color = "w")
        plt.legend(loc = "upper left", framealpha = 0.2)
        plt.tight_layout()
        plt.savefig(f"./histograms/hist{island}_{idx}.png", transparent = True)
        plt.close()
    except AssertionError:
        print("Index out of range, or correspondent array not sampled.")


def save_figures_given_attention(
        samples,
        ground,
        args,
        px_x,
        px_y,
        mask = None
    ):
    """
    Save all the figures of the metrics given an attention model.

    :param samples: inferences with shape (nº days, img_size, img_size)
    :param ground: ground with shape (nº days, img_size, img_size)
    :param args: parsed arguments
    :param mask: mask with 0s at sea, and 1s in Tenerife with shape (img_size, img_size)
    """
    # Temperature vs Time
    Temperature_vs_TimeMAX(samples, ground, args, save = True)

    # Bias
    BIAS(samples, ground, args, mask, save = True)

    # Bias P2
    BIAS_P(2, samples, ground, args, mask, save = True)

    # Bias P98
    BIAS_P(98, samples, ground, args, mask, save = True)

    # RMSE
    RMSE(samples, ground, args, mask, save = True)

    # Std ratio
    std_ratio(samples, ground, args, mask, save = True)

    # Pearson coefficient
    PearsonCoeff(samples, ground, args, mask, save = True)

    # See correlation
    plt.figure(figsize = (7, 7))
    correlation(samples, ground, px_x, px_y, args, save = True)

    
