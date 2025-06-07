from train import *
from sample import *
from stats import *
from TF import *
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = "Diffusion model",
        description = "Implementation of diffusion model for statistical downscaling"
    )
    parser.add_argument("-b", "--batchsize", type = int, help = "Size of the batch")
    parser.add_argument("-e", "--epochs", type = int, help = "Number of epochs")
    parser.add_argument("-l", "--learningrate", type = float, help = "Learning rate")
    parser.add_argument("-t", "--timesteps", type = int, help = "Number of timesteps")
    parser.add_argument("-d", "--embdim", type = int, help = "Dimension of the time embedding")
    parser.add_argument("-a", "--attention", type = str, choices = ["none", "triplet"],help = "Type of attention")
    parser.add_argument("-i", "--imgsize", type = int, default = 32, help = "Input size for nxn shape")
    parser.add_argument("-m", "--mode", type = str, choices = ["train", "sample", "infere", "stats"], help = "Mode of use")
    parser.add_argument("-s", "--seed", type = int, default = 14573, help = "Seed")
    parser.add_argument("--idx", type = int, default = None, help = "If sampling, index to infere")
    parser.add_argument("--island", type = str, default = "tf", choices = ["tf", "gc", "lp"], help = "Island to study")
    parser.add_argument("-v", "--verbose", type = bool, default = False, help = "Print progress from training at each epoch")
    
    args = parser.parse_args()

    # Fix the seed
    torch.manual_seed(args.seed)

    # Call the model
    model = UNet(attention = args.attention, time_embedding_dim = args.embdim)
    diffusion_model = diffusion_cosine(timesteps = args.timesteps)


    # Define the mask
    mask_path = "netcdf/mask.nc"
    mask = netCDF4.Dataset(mask_path)["T2MEAN"]
    mask = mask[0, ::-1, :]


    # Weights of the model
    path = "./weights/"
    name = f"UNET_i{args.island}_b{args.batchsize}_e{args.epochs}_l{args.learningrate}_d{args.embdim}_t{args.timesteps}_a{args.attention}.pth"

    # Train the diffusion model
    if args.mode == "train":
        # Dataset
        dataset, dataloader = set_and_loader(args.batchsize, island = args.island, train = True)
        n_batchs = (len(dataset)//args.batchsize) + 1
        ds_test, dl_test = set_and_loader(args.batchsize, island = args.island, train = False, verbose = False)
        n_batchs_test = (len(ds_test)//args.batchsize) + 1


        print("Training model...")
        print_info(model, args)
        opt = torch.optim.Adam(model.parameters(), args.learningrate)
        _, _, path = train(
            model = model,
            loss_fn = MSE_loss,
            diffusion = diffusion_model,
            timesteps = args.timesteps,
            optimizer = opt,
            epochs = args.epochs,
            attention = args.attention,
            n_batchs = n_batchs,
            loader = dataloader,
            n_batchs_test = n_batchs_test,
            test = dl_test,
            dataset = dataset,
            ds_test = ds_test,
            args = args,
            weights_path = path + name
        )
        print("Model succesfully trained. Weights located at " + path + name)
    
    # Sample arrays
    if args.mode == "sample":
        # Dataset
        dataset, dataloader = set_and_loader(args.batchsize, island = args.island, train = False)
        n_batchs = (len(dataset)//args.batchsize) + 1
        extremes = dataset.T_extremes()

        # Load model weights
        path = "weights/"
        name = f"UNET_i{args.island}_b{args.batchsize}_e{args.epochs}_l{args.learningrate}_d{args.embdim}_t{args.timesteps}_a{args.attention}.pth"
        model.load_state_dict(torch.load(path + name, weights_only=True))
        print("Selected weights from " + path + name)
        
        inferences_singleidx(
            model = model,
            diff_model = diffusion_model,
            dataset = dataset,
            idx = 0,
            n_samples = 3,
            img_size = args.imgsize,
            args = args
        )

        """sample_arrays(
            model = model,
            diff_model = diffusion_model,
            dataset = dataset,
            days = len(dataset),
            img_size = args.imgsize,
            attention = args.attention,
            args = args,
            ground_ = True
        )"""

        print("Arrays has been sampled succesfully")

    if args.mode == "infere":
        # Dataset
        dataset, dataloader = set_and_loader(args.batchsize, island = args.island, train = False)
        n_batchs = (len(dataset)//args.batchsize) + 1

        # Load model weights
        path = "weights/"
        name = f"UNET_i{args.island}_b{args.batchsize}_e{args.epochs}_l{args.learningrate}_d{args.embdim}_t{args.timesteps}_a{args.attention}.pth"
        model.load_state_dict(torch.load(path + name, weights_only=True))
        print("Selected weights from " + path + name)

        sample(
            model = model,
            diffusion = diffusion_model,
            dataset = dataset,
            img_size = args.imgsize,
            mask = mask,
            args = args,
            idx = args.idx
        )

    # Compute metrics
    if args.mode == "stats":
        # Dataset
        dataset, dataloader = set_and_loader(args.batchsize, island = args.island, train = False)
        n_batchs = (len(dataset)//args.batchsize) + 1

        boolean = False
        ground = np.load(f"sampled_arrays/ground_i{args.island}.npy")
        samples = np.load(f"sampled_arrays/samples_i{args.island}_b{args.batchsize}_e{args.epochs}_l{args.learningrate}_d{args.embdim}_t{args.timesteps}_a{args.attention}.npy")
        m = samples.shape[0]

        mask = xr.open_dataset(f"./netcdf/{args.island}_mask.nc")["MASK"][::-1, :]

        island_pxs = {
            "tf" : (19, 20),
            "gc" : (19, 20),
            "lp" : (7, 15)
        }

        px_x, px_y = island_pxs[args.island]
        print("Selected weights from " + path + name)

        print("Computing model metrics....")
        save_figures_given_attention(
            samples = samples, 
            ground = ground, 
            args = args,
            px_x = px_x,
            px_y = px_y,
            mask = mask
        )
        print("Done!")
