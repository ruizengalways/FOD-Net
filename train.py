"""FOD-Net
Fiber orientation distribution super resolution
Licensed under the CC BY-NC-SA 4.0 License (see LICENSE for details)
Written by Rui Zeng @ The University of Sydney (r.zeng@outlook.com / rui.zeng@sydney.edu.au)

Example:
    Train a fodnet model:
        python train.py --dataroot ./dataset/ --dataset_mode hcp --cfg ./config/fodnet_updated_config.yml

"""
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from tqdm import tqdm
from util.visualizer import Visualizer
from torch.backends import cudnn
import torch

import numpy as np
torch.backends.cudnn.benchmark = True
import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    opt = TrainOptions().parse() 
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)  
    visualizer = Visualizer(opt)  
    total_iters = 0 

    for epoch in range(opt.epoch_count,
                       opt.epoch_end + 1):

        t = tqdm(dataset, desc='Epoch %d' % epoch)
        model.epoch = epoch
        model.total_metric = 0.

        for i, data in enumerate(t):  # inner loop within one epoch
            total_iters += 1
            model.set_input(data)
            model.optimize_parameters()

            with torch.no_grad():
                model.total_metric += model.loss_total.item()
                t.set_description(
                    "Total loss: {0:.6f} Step loss: {1:.6f}".format(model.total_metric / (i + 1), model.loss_total.item()))
                if total_iters % opt.print_freq == 0:
                    visualizer.plot_current_losses_tensorboard(model.get_current_losses(),
                                                               global_step=(epoch - 1) * dataset_size + i)

                if total_iters % opt.save_latest_freq == 0:
                    print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    save_suffix = 'latest'
                    model.save_networks(save_suffix)

        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks(epoch)
        
