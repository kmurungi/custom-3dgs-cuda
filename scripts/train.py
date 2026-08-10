import argparse 
import random
import torch as t
from tqdm import tqdm 
from gaussian_splatting.scene.camera import camera
from gaussian_splatting.scene.gaussian import gaussian 
from gaussian_splatting.rasterizer import rasterize
from gaussian_splatting.utils.loss import calculate_loss
from gaussian_splatting.utils.saveweights import save_weights
from gaussian_splatting.utils.dataloader import load_colmap
from gaussian_splatting.utils.optim import setup_optimizer

def isRefinementIteration(i): 
    if 500 <= i <= 15000 and i % 100 == 0: 
        return True
    return False


def train(args): 
    """
    Training loop Custom Gaussian Splatting
    """
    # initialize gaussian and camera objects
    gaussians, cameras = load_colmap(args.dataset) 
    images = cameras.sample_camera_view() 
    iteration = 0 

    # learning rate 
    learning_rates = {
        "mu": 0.01
    }

    # initialize optimizer
    optimizer = setup_optimizer(gaussians, lrs = learning_rates, lr = args.lr)

    total_iterations = args.epochs * len(images)
    pbar = tqdm(range(total_iterations), desc="Training 3DGS")

    for epoch in args.epochs: 
        random.shuffle(images) # shuffle image order each epoch
        for image in images: 
            rendered_img = rasterize.forward(gaussians, image) # forward pass
            loss = calculate_loss(image, rendered_img)
            rasterize.backward(rendered_img, loss) # backward pass 
        
            if isRefinementIteration(iteration): 
                Adaptive_Density_Control()

            iteration+=1

        pbar.update(1) 

        if epoch % (args.epoch//args.c) == 0: 
            save_weights(gaussians, args.checkpoint_path) # save checkpoint

    pbar.close()
    save_weights(gaussians, args.output)


def parse_arguments(): 
    parser = argparse.ArgumentParser(
        description="3D Gaussian Splatting Training & De-Lighting Engine"
    )
    parser.add_argument("-d", "--dataset", type = str, help = "path for loading data")

    # optional arguments
    parser.add_argument("-d", "--delight", type = bool, default = "store_true", help = "activate delighting")
    parser.add_argument("--lr", type = float, default = 0.001, help = "learning rate")
    parser.add_argument("-e", "--epochs", type = int, default = 100, help = "epochs")
    parser.add_argument("-o", "--output", type = str, default = "./output", help = "output folder")
    parser.add_argument("-c", "--checkpoints", type = int, default = 5, help = "number of checkpoints")
    parser.add_argument("-cp", "--checkpoint_path", type = str, default = "./checkpoints")
    parser.add_argument("-b", "--benchmark", type = bool, default = "store_false", help = "save benchmark")

    args = parser.parse_args()
    return args 


def main(): 
    args = parse_arguments()
    train(args)


if __name__ == "__main__": 
    main()