"""FOD-Net
Fiber orientation distribution super resolution
Licensed under the CC BY-NC-SA 4.0 License (see LICENSE for details)
Written by Rui Zeng @ The University of Sydney (r.zeng@outlook.com / rui.zeng@sydney.edu.au)
"""

from re import I
import torch
from .base_model import BaseModel
from . import networks
import torch.nn
import torch.nn.functional
import torch.optim
import os
import nibabel as nib
import sys
import numpy as np
from dipy.segment.mask import bounding_box, crop
from tqdm import trange


fodlr_final_mean = np.load("./util/fodlr_final_mean.npy")
fodlr_final_std = np.load("./util/fodlr_final_std.npy")
fodgt_final_mean = np.load("./util/fodgt_final_mean.npy")
fodgt_final_std = np.load("./util/fodgt_final_std.npy")


class fodnetModel(BaseModel):
    """
    This class implements the fodnet model, for learning fod super resolution.

    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        """
        if is_train:
            pass

        return parser

    def __init__(self, opt):
        """Initialize the SMC GAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super(fodnetModel, self).__init__(opt)
        # specify the training losses you want to print out. The training/test scripts will
        self.loss_names = ['loss_total']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = [opt.model]
        else:
            self.model_names = [opt.model]

        # define networks
        self.fodnet = networks.define_network(init_type=opt.init_type,
                                              init_gain=opt.init_gain,
                                              gpu_ids=self.gpu_ids)
        if self.isTrain:
            self.optimizer_names = ['optimizer_1']
            self.optimizer_1 = torch.optim.Adam(self.fodnet.parameters(), eps=1e-7,
                                                lr=opt.lr)
            self.optimizers.append(self.optimizer_1)
            self.l2loss = torch.nn.MSELoss()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        """
        if self.opt.isTrain == True:
            self.fodgt = input['fodgt'].to(self.device)
            self.fodlr = input['fodlr'].to(self.device)

    def set_input_for_test(self, fodlr, brain_mask, fod_affine, fod_header):
        """ data preparation for FOD super resolution inference

        Input:
            fodlr: low angular resolution FOD array
            brain_mask: brain mask array
            fod_affine: used to indicate which affine we should use when saving super resolved fod image
            fod_header: used to indicate which header info we should use when saving super resolved fod image
        
        """
        self.fod_affine = fod_affine
        self.fod_header = fod_header

        # Load stats for normalisation
        self.fodlr_mean = fodlr_final_mean
        self.fodlr_std = fodlr_final_std
        self.fodgt_mean = fodgt_final_mean
        self.fodgt_std = fodgt_final_std

        self.fodlr_mean = np.asarray(self.fodlr_mean).reshape(
            1, 1, 1, -1).astype(np.float32)
        self.fodlr_std = np.asarray(self.fodlr_std).reshape(
            1, 1, 1, -1).astype(np.float32)
        self.fodgt_mean = np.asarray(self.fodgt_mean).reshape(
            1, 1, 1, -1).astype(np.float32)
        self.fodgt_std = np.asarray(self.fodgt_std).reshape(
            1, 1, 1, -1).astype(np.float32)

        self.fodlr_mean = torch.from_numpy(self.fodlr_mean).to(self.device)
        self.fodlr_std = torch.from_numpy(self.fodlr_std).to(self.device)
        self.fodgt_mean = torch.from_numpy(self.fodgt_mean).to(self.device)
        self.fodgt_std = torch.from_numpy(self.fodgt_std).to(self.device)

        # # Create the bounding box for the foreground region and crop it for fast testing
        # brain_mask = np.asarray(brain_mask, dtype=np.float32)
        # self.mins, self.maxs = bounding_box(brain_mask)
        # self.mins.append(None)
        # self.maxs.append(None)
        # self.affine = fod_affine
        # self.output_shape = fodlr.shape
        # fodlr = crop(fodlr, self.mins, self.maxs)

        # Move data to torch tensor
        self.brain_mask = torch.from_numpy(
            brain_mask.astype(np.float32)).to(self.device)

        self.brain_mask = torch.nn.functional.pad(self.brain_mask, (5, 5, 5, 5, 5, 5), "constant", 0)
        brain_mask = self.brain_mask.cpu().numpy() # copy zero-padded brain mask tensor to brain mask array
        index_mask = np.where(brain_mask)
        self.index_mask = np.asarray(index_mask)
        self.index_length = len(self.index_mask[0])

        # Get normalised low resolution FOD images
        fodlr = torch.from_numpy(fodlr).to(self.device)
        fodlr = torch.nn.functional.pad(fodlr, (0, 0, 5, 5, 5, 5, 5, 5), "constant", 0)
        self.normalised_fodlr = (fodlr - self.fodlr_mean) / self.fodlr_std

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>.
        """
        self.fodpred = self.fodnet(self.fodlr)

    def backward(self):
        """Calculate the loss """
        self.loss_total = self.l2loss(self.fodpred, self.fodgt)
        self.loss_total.backward()
        # torch.nn.utils.clip_grad_norm(self.fodnet.parameters(), 1.)

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.forward()
        self.optimizer_1.zero_grad()
        self.backward()
        self.optimizer_1.step()

    @torch.no_grad()
    def test(self, sr_fod_path):
        """Perform FOD super resolution using FOD-Net
        """
        output_directory_path = os.path.dirname(sr_fod_path)
        os.makedirs(output_directory_path, exist_ok=True)
        
        fodsr = torch.zeros_like(self.normalised_fodlr)
        self.normalised_fodlr = self.normalised_fodlr.permute(3, 0, 1, 2)
        
        size_3d_patch = 9
        margin = int(size_3d_patch/2)

        fodgt_std = self.fodgt_std.squeeze(0).squeeze(0).squeeze(0)
        fodgt_mean = self.fodgt_mean.squeeze(0).squeeze(0).squeeze(0)

        print('Start FOD super resolution:')
        for i in trange(self.index_length):
            # get x, y, and z coordinates for each voxel we want to perform super resolution
            x = self.index_mask[0, i]
            y = self.index_mask[1, i]
            z = self.index_mask[2, i]

            x_start = x - margin
            x_end = x_start + size_3d_patch
            y_start = y - margin
            y_end = y_start + size_3d_patch
            z_start = z - margin
            z_end = z_start + size_3d_patch

            self.fodlr = self.normalised_fodlr[:, x_start:x_end, y_start:y_end, z_start:z_end]
            tensor_helper = self.fodlr
            self.fodlr = torch.stack(
                [self.fodlr.float(), tensor_helper.float()])
            self.forward()
            fodsr[x, y, z, :] = self.fodpred[0, :] * fodgt_std + fodgt_mean

        # Mask out zero regions
        fodsr *= self.brain_mask.unsqueeze(-1)
        fodsr = fodsr.detach().cpu().numpy()

        fodsr = fodsr[5:-5, 5:-5, 5:-5, :]

        # Save super resolved FOD image
        nii = nib.Nifti1Image(
            fodsr, affine=self.fod_affine, header=self.fod_header)
        nib.save(nii, sr_fod_path)

    def inference(self, fod_path, output_path):
        """Inference for a given fodlr subject"""
        size_3d_patch = 9
        with torch.no_grad():
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            except:
                sys.exit('we cannot create the dir')

            height, width, depth, channels = self.fodlr_whole.shape
            self.fodlr_whole = self.fodlr_whole.permute(3, 0, 1, 2)
            template = torch.zeros_like(self.fodlr_whole)
            final_result = torch.zeros(
                tuple(self.output_shape)).to(self.device)
            print('Start FOD super resolution:')
            for i in tqdm(range(height - size_3d_patch + 1)):
                for j in range(width - size_3d_patch + 1):
                    for k in range(depth - size_3d_patch + 1):
                        self.fodlr = self.fodlr_whole[:, i:i + size_3d_patch,
                                                      j:j + size_3d_patch, k:k + size_3d_patch]
                        tensor_helper = self.fodlr
                        self.fodlr = torch.stack(
                            [self.fodlr.float(), tensor_helper.float()])
                        self.forward()
                        template[:, int((2 * i + size_3d_patch) / 2),
                                 int((2 * j + size_3d_patch) / 2),
                                 int((2 * k + size_3d_patch) / 2)] = self.fodpred[0:1, :] * self.fodgt_std + self.fodgt_mean

            # Fill the result into
            final_result[self.mins[0]:self.maxs[0], self.mins[1]:self.maxs[1],
                         self.mins[2]:self.maxs[2], :] = template.permute(1, 2, 3, 0)

            final_result = final_result * self.brain_mask.unsqueeze(-1)

            # Dump template into a nii gz file for further evaluation using mrtrix
            final_result = final_result.detach().cpu().numpy()

            # load header from the lr data
            lr_info = nib.load(fod_path)
            nii = nib.Nifti1Image(final_result, affine=lr_info.affine,
                                  header=lr_info.header)

            nib.save(nii, output_path)
