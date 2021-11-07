"""FOD-Net
Fiber orientation distribution super resolution
Licensed under the CC BY-NC-SA 4.0 License (see LICENSE for details)
Written by Rui Zeng @ The University of Sydney (r.zeng@outlook.com / rui.zeng@sydney.edu.au)


"""
from data.hcp_dataset import flip_axis_to_match_HCP_space
from models.fodnet_model import fodnetModel
import nibabel as nib
from options.test_options import TestOptions

if __name__ == '__main__':
    opt = TestOptions().parse()
    gpu_ids = opt.gpu_ids
    weights_path = opt.weights_path
    fod_path = opt.fod_path
    output_path = opt.output_path
    brain_mask_path = opt.brain_mask_path

    fodlr_file = nib.load(fod_path)
    brain_mask_file = nib.load(brain_mask_path)

    fixed_fodlr, fixed_fodlr_affine, flipped_fodlr_axis = flip_axis_to_match_HCP_space(fodlr_file.get_data(), fodlr_file.affine)
    fixed_brain_mask, fixed_brain_mask_affine, flipped_brain_mask_axis = flip_axis_to_match_HCP_space(brain_mask_file.get_data(), brain_mask_file.affine)

    assert fixed_fodlr.shape[:3] == fixed_brain_mask.shape, 'Input fod and mask should have the same shape'

    model = fodnetModel(opt)
    model.load_weights(weights_path=weights_path)                 
    model.eval()
    model.set_input_for_test(fixed_fodlr, fixed_brain_mask, fixed_fodlr_affine, fodlr_file.header)
    model.test(output_path)



