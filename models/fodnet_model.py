"""
Fiber orientation super resolution
Licensed under the CC BY-NC-SA 4.0 License (see LICENSE for details)
Written by Rui Zeng @ USyd Brain and Mind Centre (r.zeng@outlook.com / rui.zeng@sydney.edu.au)

"""

import torch
from . import networks
import torch.nn
import torch.nn.functional
import torch.optim
import os

import nibabel as nib
import sys
import numpy as np
from dipy.segment.mask import bounding_box, crop
from tqdm import tqdm


class fodnetModel():
    """
    This class implements the fodnet model, for learning fod super resolution.

    """
    def __init__(self, gpu_id):
        """Initialize the SMC GAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # specify the training losses you want to print out. The training/test scripts will
        self.gpu_id = gpu_id
        if gpu_id != -1:
            self.device = torch.device('cuda:{}'.format(gpu_id))
        else:
            self.device = torch.device('cpu')

        # define networks
        self.net = networks.define_network(device=self.device)

    def set_input(self, fodlr, brain_mask, fod_affine, act_mask):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        """
        brain_mask = np.asarray(brain_mask, dtype=np.float32)

        self.mins, self.maxs = bounding_box(brain_mask)
        self.mins.append(None)
        self.maxs.append(None)
        self.affine = fod_affine
        self.output_shape = fodlr.shape

        # Load stats
        stats = np.load('stats.npz')
        self.fodlr_mean = stats['fodlr_mean'].astype(np.float32)
        self.fodlr_std = stats['fodlr_std'].astype(np.float32)
        self.fodgt_mean = stats['fodgt_mean'].astype(np.float32)
        self.fodgt_std = stats['fodgt_std'].astype(np.float32)

        fodlr = crop(fodlr, self.mins, self.maxs)

        if act_mask is not None:
            cropped_act_mask = crop(act_mask, self.mins, self.maxs)
            cropped_brain_mask = crop(brain_mask, self.mins[:-1], self.maxs[:-1])
            csfmask = cropped_act_mask[:, :, :, 3]
            csfmask = csfmask > 0.9
            # wmmask = cropped_act_mask[:, :, :, 2]
            cropped_brain_mask = ~(cropped_brain_mask.astype(np.bool))
            final_mask = np.logical_or(cropped_brain_mask, csfmask)
            voxel_lr_mask = np.tile(np.expand_dims(final_mask, -1), (1, 1, 1, fodlr.shape[-1]))
            fodlr = np.ma.array(fodlr, mask=voxel_lr_mask, dtype=np.float32)
            fodlr = (fodlr - self.fodlr_mean) / self.fodlr_std
            fodlr = np.ma.filled(fodlr, 0.).astype(np.float32)

        fodlr = torch.from_numpy(fodlr)
        brain_mask = torch.from_numpy(brain_mask)

        self.brain_mask = brain_mask.to(self.device)
        # Load numpy to GPU
        self.fodlr_mean = torch.from_numpy(self.fodlr_mean).to(self.device)
        self.fodlr_std = torch.from_numpy(self.fodlr_std).to(self.device)
        self.fodgt_mean = torch.from_numpy(self.fodgt_mean).to(self.device)
        self.fodgt_std = torch.from_numpy(self.fodgt_std).to(self.device)

        if act_mask is None:
            self.fodlr_whole = fodlr.to(self.device)
            self.fodlr_whole = (self.fodlr_whole - self.fodlr_mean)/self.fodlr_std
        else:
            self.fodlr_whole = fodlr.to(self.device)





    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>.
        Regarding res3dunet architecture, we regress the residual between fodgt and fodlr, and thus we need to add the
            fodlr back to recover the fodpred
        """
        self.fodpred = self.net(self.fodlr)


    def test(self, fod_path, output_dir):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        size_3d_patch = 9
        with torch.no_grad():
            try:
                os.makedirs(os.path.dirname(output_dir), exist_ok=True)
            except:
                sys.exit('we cannot create the dir')

            height, width, depth, channels = self.fodlr_whole.shape
            self.fodlr_whole = self.fodlr_whole.permute(3, 0, 1, 2)
            template = torch.zeros_like(self.fodlr_whole)
            final_result = torch.zeros(tuple(self.output_shape)).to(self.device)
            print('Start FOD super resolution:')
            for i in tqdm(range(height - size_3d_patch + 1)):
                for j in range(width - size_3d_patch + 1):
                    for k in range(depth - size_3d_patch + 1):
                        self.fodlr = self.fodlr_whole[ :, i:i + size_3d_patch,
                                     j:j + size_3d_patch, k:k + size_3d_patch]
                        tensor_helper = self.fodlr
                        self.fodlr = torch.stack([self.fodlr.float(), tensor_helper.float()])
                        self.forward()
                        template[ :, int((2 * i + size_3d_patch) / 2),
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

            nib.save(nii, output_dir)



    def setup(self, weights_path):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.load_networks(weights_path)


    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, load_path):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        print('loading the model from %s' % load_path)
        # if you are using PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        # patch InstanceNorm checkpoints prior to 0.4
        for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
            self.__patch_instance_norm_state_dict(state_dict, self.net, key.split('.'))
        self.net.load_state_dict(state_dict)

