import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xarray as xr
import cartopy
import cartopy.crs as ccrs

tf = xr.open_dataset(f"./netcdf/tf.nc")
mask_tf = xr.open_dataset(f"./netcdf/tf_mask.nc")
gc = xr.open_dataset(f"./netcdf/gc.nc")
mask_gc = xr.open_dataset(f"./netcdf/gc_mask.nc")
lp = xr.open_dataset(f"./netcdf/lp.nc")
mask_lp = xr.open_dataset(f"./netcdf/lp_mask.nc")

conds = xr.open_dataset("./netcdf/conditions.nc")
m_train = int(np.floor(80 * len(tf.Times) / 100))
tf = tf.where(tf.Times >= tf.Times[m_train], drop = True)
tf["T2MEAN"] = tf["T2MEAN"] - 273.15
gc = gc.where(gc.Times >= gc.Times[m_train], drop = True)
gc["T2MEAN"] = gc["T2MEAN"] - 273.15
lp = lp.where(lp.Times >= lp.Times[m_train], drop = True)
lp["T2MEAN"] = lp["T2MEAN"] - 273.15
tf["pred"] = (("Times", "lat", "lon"), np.load("sampled_arrays/samples_itf_b128_e50_l0.0005_d2048_t1000_anone.npy")[:, ::-1, :] - 273.15)
gc["pred"] = (("Times", "lat", "lon"), np.load("sampled_arrays/samples_igc_b128_e50_l0.0005_d2048_t1000_anone.npy")[:, ::-1, :] - 273.15)
lp["pred"] = (("Times", "lat", "lon"), np.load("sampled_arrays/samples_ilp_b128_e50_l0.0005_d2048_t1000_anone.npy")[:, ::-1, :] - 273.15)

tf = tf.assign(dict(
    bias = (("lat", "lon"), np.load("./arrays/biastf.npy")[::-1, :]),
    biasP2 = (("lat", "lon"), np.load("./arrays/biasp2tf.npy")[::-1, :]),
    biasP98 = (("lat", "lon"), np.load("./arrays/biasp98tf.npy")[::-1, :]),
    rmse = (("lat", "lon"), np.load("./arrays/RMSEtf.npy")[::-1, :]),
    stdratio = (("lat", "lon"), np.load("./arrays/stdratiotf.npy")[::-1, :]),
    pearson = (("lat", "lon"), np.load("./arrays/pearsontf.npy")[::-1, :]),
))

gc = gc.assign(dict(
    bias = (("lat", "lon"), np.load("./arrays/biasgc.npy")[::-1, :]),
    biasP2 = (("lat", "lon"), np.load("./arrays/biasp2gc.npy")[::-1, :]),
    biasP98 = (("lat", "lon"), np.load("./arrays/biasp98gc.npy")[::-1, :]),
    rmse = (("lat", "lon"), np.load("./arrays/RMSEgc.npy")[::-1, :]),
    stdratio = (("lat", "lon"), np.load("./arrays/stdratiogc.npy")[::-1, :]),
    pearson = (("lat", "lon"), np.load("./arrays/pearsongc.npy")[::-1, :]),
))

lp = lp.assign(dict(
    bias = (("lat", "lon"), np.load("./arrays/biaslp.npy")[::-1, :]),
    biasP2 = (("lat", "lon"), np.load("./arrays/biasp2lp.npy")[::-1, :]),
    biasP98 = (("lat", "lon"), np.load("./arrays/biasp98lp.npy")[::-1, :]),
    rmse = (("lat", "lon"), np.load("./arrays/RMSElp.npy")[::-1, :]),
    stdratio = (("lat", "lon"), np.load("./arrays/stdratiolp.npy")[::-1, :]),
    pearson = (("lat", "lon"), np.load("./arrays/pearsonlp.npy")[::-1, :]),
))


def get_time(
    ds : xr.Dataset
) -> str:
    times = ds.Times.values
    str_times = []
    for time in times:
        str_time = str(time)
        str_times.append(str_time[6:8] + "-" + str_time[4:6] + "-" + str_time[:4])
    
    return str_times

def samples(tf, gc, lp):
    for idx in range(len(tf.Times)):
        if idx%100 == 0:
            fig = plt.figure(figsize = (20, 10))
            plt.rcParams['font.size'] = 20
            plt.rcParams["text.color"] = "white"
            gs = fig.add_gridspec(4, 4)
            ax_real = fig.add_subplot(gs[:2, :], projection=ccrs.PlateCarree())
            ax_real.set_extent((np.min(lp.lon), np.max(gc.lon) + 1.7, np.min(gc.lat), np.max(lp.lat) + 0.5))
            ax_pred = fig.add_subplot(gs[2:, :], projection=ccrs.PlateCarree())
            ax_pred.set_extent((np.min(lp.lon), np.max(gc.lon) + 1.7, np.min(gc.lat), np.max(lp.lat) + 0.5))

            cbar_ax = fig.add_axes([0.75, 0.11, 0.02, 0.77])
            cbar_ax.tick_params(labelcolor = "white")
            cbar_ax.yaxis.label.set_color('white')

            tfidx = tf.isel(Times = idx)
            gcidx = gc.isel(Times = idx)
            lpidx = lp.isel(Times = idx)

            min_ = np.min([tfidx.T2MEAN, tfidx.pred, gcidx.T2MEAN, gcidx.pred, lpidx.T2MEAN, lpidx.pred])
            max_ = np.max([tfidx.T2MEAN, tfidx.pred, gcidx.T2MEAN, gcidx.pred, lpidx.T2MEAN, lpidx.pred])

            mappable = tfidx.pred.where(tfidx.T2MEAN != tfidx.T2MEAN.values[0, 0]).plot(ax = ax_pred, vmax = max_, vmin = min_, cmap = "jet", add_colorbar = False)
            gcidx.pred.where(gcidx.T2MEAN != tfidx.T2MEAN.values[0, 0]).plot(ax = ax_pred, vmax = max_, vmin = min_, cmap = "jet", add_colorbar = False)
            lpidx.pred.where(lpidx.T2MEAN != tfidx.T2MEAN.values[0, 0]).plot(ax = ax_pred, vmax = max_, vmin = min_, cmap = "jet", add_colorbar = False)

            tfidx.T2MEAN.where(tfidx.T2MEAN != tfidx.T2MEAN.values[0, 0]).plot(ax = ax_real, vmax = max_, vmin = min_, cmap = "jet", add_colorbar = False)
            gcidx.T2MEAN.where(gcidx.T2MEAN != tfidx.T2MEAN.values[0, 0]).plot(ax = ax_real, vmax = max_, vmin = min_, cmap = "jet", add_colorbar = False)
            lpidx.T2MEAN.where(lpidx.T2MEAN != tfidx.T2MEAN.values[0, 0]).plot(ax = ax_real, vmax = max_, vmin = min_, cmap = "jet", add_colorbar = False)

            plt.colorbar(mappable, label = "Temperature (ºC)", cax = cbar_ax)

            ax_real.set_title(f"Date: {get_time(tf)[idx]} \n \n WRF")
            ax_real.coastlines()
            ax_real.add_feature(cartopy.feature.OCEAN,facecolor=("lightblue"))
            ax_real.add_feature(cartopy.feature.BORDERS, linestyle = ":")
            ax_pred.set_title("Inference")
            ax_pred.coastlines()
            ax_pred.add_feature(cartopy.feature.OCEAN, facecolor=("lightblue"))

            plt.savefig(f"./cartopy_tests/samples_pres/{idx}.png", transparent = True)
            plt.close()


def metrics():
    for metric, title in zip(["bias", "biasP2", "biasP98", "rmse", "stdratio", "pearson"], ["Bias", "2nd Percentile Bias", "98th Percentile Bias", "RMSE", "Ratio of standard deviations", "Pearson coefficient"]):
        fig = plt.figure(figsize = (20, 10))
        ax = plt.axes(projection = ccrs.PlateCarree(central_longitude=0))
        ax.set_extent((np.min(lp.lon), np.max(gc.lon) + 1.7, np.min(gc.lat), np.max(lp.lat) + 0.5))
        plt.rcParams['font.size'] = 30
        plt.rcParams["text.color"] = "white"


        min_ = np.min([tf[metric], gc[metric], lp[metric]])
        max_ = np.max([tf[metric], gc[metric], lp[metric]])

        if metric == "pearson":
            min_ = 0.7

        if metric =="stdratio":
            max_ = 1.3

        mapp = tf.where(tf.isel(Times = 0).T2MEAN != tf.isel(Times = 0).T2MEAN.values[0, 0])[metric].plot(ax = ax, cmap = "jet", vmin = min_, vmax = max_, add_colorbar = False)
        gc.where(gc.isel(Times = 0).T2MEAN != tf.isel(Times = 0).T2MEAN.values[0, 0])[metric].plot(ax = ax, cmap = "jet", vmin = min_, vmax = max_, add_colorbar = False)
        lp.where(lp.isel(Times = 0).T2MEAN != tf.isel(Times = 0).T2MEAN.values[0, 0])[metric].plot(ax = ax, cmap = "jet", vmin = min_, vmax = max_, add_colorbar = False)    
        cbar = fig.colorbar(mapp, shrink = 0.7)
        cbar.ax.tick_params(labelcolor = "white")
        
        

        ax.coastlines()
        ax.add_feature(cartopy.feature.OCEAN, facecolor=("lightblue"))

        plt.rcParams.update({'font.size': 20})
        ax.set_title(title)
        plt.rcParams.update({"text.color" : "white"})
        plt.savefig(f"./cartopy_tests/metrics_pres/{metric}.png", transparent = True)
        plt.close()

def time(ds, island):
    max_temp = ds.T2MEAN.max(dim = "lat").max(dim = "lon")
    plt.plot(get_time(ds), max_temp, "k.")
    plt.plot(get_time(ds)[800], max_temp[800], "gs", label = "Good prediction")
    plt.plot(get_time(ds)[1300], max_temp[1300], "rs", label = "Worse prediction")
    plt.xticks(["01-01-2005", "01-01-2006", "01-01-2007", "01-01-2008", "01-01-2009"], labels = ["2005", "2006", "2007", "2008", "2009"])
    plt.ylabel(r"Maximum temperature ($^\circ C$)")
    plt.xlabel("Date")
    plt.grid()
    plt.tight_layout()
    plt.legend()

    plt.savefig(f"./cartopy_tests/time{island}.png")
    plt.clf()

def plot_samples(ds, var, name):

    lat, lon = ds.lat, ds.lon
    fig = plt.figure(figsize = (20, 10))
    plt.rcParams['font.size'] = 30
    ax = plt.axes(projection = ccrs.PlateCarree(central_longitude=0))
    ax.set_extent((np.min(lon), np.max(lon), np.min(lat), np.max(lat)))
    
    mapp = ds[var][0, 0].plot(ax = ax, cmap = "jet", add_colorbar = False)
    fig.colorbar(mappable = mapp, label = r"Temperature ($^\circ C$)")
    ax.coastlines()
    ax.set_title("")
    plt.savefig(f"./cartopy_tests/wrf/{name}.png")
    plt.close()

def plot_conds(conds, var):
    fig = plt.figure(figsize = (20, 10))
    ax = plt.axes(projection = ccrs.PlateCarree(central_longitude=0))
    ax.set_extent((np.min(conds.lon), np.max(conds.lon), np.min(conds.lat), np.max(conds.lat)))
    conds = conds.isel(time = 0).isel(plev = 0)
    conds[var].plot(ax = ax, cmap = "jet", add_colorbar = False)
    ax.coastlines()
    plt.axis(False)
    plt.savefig(f"./cartopy_tests/conds/{var}.png", transparent = True)


