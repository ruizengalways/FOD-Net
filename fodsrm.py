"""
Fiber orientation super resolution
Licensed under the CC BY-NC-SA 4.0 License (see LICENSE for details)
Written by Rui Zeng @ The University of Sydney (r.zeng@outlook.com / rui.zeng@sydney.edu.au)

"""
import os
import numpy as np
from models.fodnet_model import fodnetModel
import argparse
from dipy.io.image import load_nifti

def nullable_string(val):
    if not val:
        return None
    return val


def build_argparser():
    DESCRIPTION = "Preprocessing dwi and corresponding t1 jointly."
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument('--fod_path', type=str, help="The path to the fod image")
    p.add_argument('--output_path', type=str, help="The output path")
    p.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    p.add_argument('--weights_path', type=str, default='placeholder')
    p.add_argument('--brain_mask_path', type=str, default='placeholder')
    p.add_argument('--act_mask_path', type=nullable_string, default=None)
    return p


if __name__ == '__main__':
    parser = build_argparser()
    args = parser.parse_args()
    gpu_ids = args.gpu_ids
    weights_path = args.weights_path
    fod_path = args.fod_path
    output_path = args.output_path
    brain_mask_path = args.brain_mask_path
    act_mask_path = args.act_mask_path

    fodlr, fod_affine = load_nifti(fod_path)
    brain_mask, brain_mask_affine = load_nifti(brain_mask_path)
    if act_mask_path is not None:
        act_mask, act_mask_affine = load_nifti(act_mask_path)
        if np.sign(act_mask_affine[0, 3])*np.sign(fod_affine[0, 3]) == -1:
            act_mask = act_mask[::-1, :, :, :]

    assert fodlr.shape[:3] == brain_mask.shape, 'Input fod and mask should have the same shape'

    if act_mask_path is not None:
        assert fodlr.shape[:3] == act_mask.shape[:3], 'Input fod and mask should have the same shape'

    if act_mask_path is None:
        act_mask = None

    model = fodnetModel(gpu_ids)      # create a model given opt.model and other options
    model.setup(weights_path=weights_path)                     # regular setup: load and print networks; create schedulers
    model.net.eval()
    model.set_input(fodlr, brain_mask, fod_affine, act_mask)  # unpack data from data loader
    model.test(fod_path, output_path)



