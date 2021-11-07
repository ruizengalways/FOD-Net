"""FOD-Net
Fiber orientation distribution super resolution
Licensed under the CC BY-NC-SA 4.0 License (see LICENSE for details)
Written by Rui Zeng @ The University of Sydney (r.zeng@outlook.com / rui.zeng@sydney.edu.au)
"""

import os
import argparse
from tqdm import tqdm

def build_argparser():
    DESCRIPTION = "Generating normalised SS3T CSD and MSMT CSD FOD files using the Human Connectome Project dataset"
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument("--hcp_dataset_path", type=str, help="hcp_correct_50_path")
    return p

def create_fods(input_path, output_path, bvecs, bvals, brain_mask, csd_type, normalisation=True):
    output_path = os.path.join(output_path, csd_type)

    os.makedirs(output_path, exist_ok=True)

    RF_WM = os.path.join(output_path, 'RF_WM.txt')
    RF_GM = os.path.join(output_path, 'RF_GM.txt')
    RF_CSF = os.path.join(output_path, 'RF_CSF.txt')

    WM_FODs_path = os.path.join(output_path, 'WM_FODs.nii.gz')
    GM_FODs_path = os.path.join(output_path, 'GM_FODs.nii.gz')
    CSF_FODs_path = os.path.join(output_path, 'CSF_FODs.nii.gz')

    WM_FODs_normalised_path = os.path.join(
        output_path, 'WM_FODs_normalised.nii.gz')
    GM_FODs_normalised_path = os.path.join(
        output_path, 'GM_FODs_normalised.nii.gz')
    CSF_FODs_normalised_path = os.path.join(
        output_path, 'CSF_FODs_normalised.nii.gz')

    if csd_type == "msmt_csd":
        print("csd_msmt Creating peaks (1 of 2)...")
        os.system("dwi2response dhollander" + " " + input_path + " " +
                  RF_WM + ' ' +
                  RF_GM + ' ' +
                  RF_CSF + ' ' +
                  '-fslgrad' + ' ' + bvecs + " " + bvals + ' ' +
                  "-mask" + '  ' + brain_mask + ' ' +
                  '-force')
        print("csd_msmt Creating peaks (2 of 2)...")
        os.system("dwi2fod msmt_csd " + input_path + " " +
                  RF_WM + ' ' +
                  WM_FODs_path + ' ' +
                  RF_GM + ' ' +
                  GM_FODs_path + ' ' +
                  RF_CSF + ' ' +
                  CSF_FODs_path + ' ' +
                  '-fslgrad' + ' ' + bvecs + " " + bvals + ' ' +
                  "-mask" + ' ' + brain_mask + ' ' +
                  '-force')

        if normalisation:
            print("normalise FOD...")
            os.system("mtnormalise" + " " +
                      WM_FODs_path + " " +
                      WM_FODs_normalised_path + " " +
                      GM_FODs_path + " " +
                      GM_FODs_normalised_path + " " +
                      CSF_FODs_path + " " +
                      CSF_FODs_normalised_path + " " +
                      "-mask" + ' ' + brain_mask + " " +
                      '-force')

    elif csd_type == 'ss3t_csd':
        # Because ss3t csd does not support nii.gz file
        basename = os.path.basename(input_path)
        basename_replaced = basename.replace('.nii.gz', '.mif')
        mif_filename = os.path.join(output_path, basename_replaced)

        print("3tissue csd Creating peaks (1 of 3)...")
        os.system("dwi2response dhollander" + " " + input_path + " " +
                  RF_WM + ' ' +
                  RF_GM + ' ' +
                  RF_CSF + ' ' +
                  '-fslgrad' + ' ' + bvecs + " " + bvals + ' ' +
                  "-mask" + '  ' + brain_mask + ' ' +
                  '-force')

        print("3tissue csd Creating peaks (2 of 3)...")
        os.system("mrconvert" + " " +
                  input_path + " " +
                  mif_filename + " " +
                  "-fslgrad" + ' ' + bvecs + " " + bvals + " " +
                  "-quiet" + " " +
                  "-force")

        print("3tissue csd Creating peaks (3 of 3)...")
        os.system("ss3t_csd_beta1 " + mif_filename + " " +
                  RF_WM + ' ' +
                  WM_FODs_path + ' ' +
                  RF_GM + ' ' +
                  GM_FODs_path + ' ' +
                  RF_CSF + ' ' +
                  CSF_FODs_path + ' ' +
                  "-mask" + ' ' + brain_mask + ' -force')

        if normalisation:
            print("normalise FOD...")
            os.system("mtnormalise" + " " +
                      WM_FODs_path + " " +
                      WM_FODs_normalised_path + " " +
                      GM_FODs_path + " " +
                      GM_FODs_normalised_path + " " +
                      CSF_FODs_path + " " +
                      CSF_FODs_normalised_path + " " +
                      "-mask" + ' ' + brain_mask + " " +
                      '-force')
    else:
        raise ValueError("'csd_type' contains invalid String")

    if normalisation:
        return WM_FODs_normalised_path
    else:
        return WM_FODs_path


def main():
    parser = build_argparser()
    args = parser.parse_args()
    hcp_dataset_path = args.hcp_dataset_path

    subject_ids = os.listdir(hcp_dataset_path) # get the subject id list

    for i, subject_id in enumerate(subject_ids):
        HARDI_dwi_path = os.path.join(hcp_dataset_path, subject_id, "HARDI_data", 'data.nii.gz')
        HARDI_bvecs_path = os.path.join(hcp_dataset_path, subject_id, "HARDI_data", 'bvecs')
        HARDI_bvals_path = os.path.join(hcp_dataset_path, subject_id, "HARDI_data", 'bvals')

        LARDI_dwi_path = os.path.join(hcp_dataset_path, subject_id, "LARDI_data", 'data_b1000_g32.nii.gz')
        LARDI_bvecs_path = os.path.join(hcp_dataset_path, subject_id, "LARDI_data", 'data_b1000_g32_bvecs')
        LARDI_bvals_path = os.path.join(hcp_dataset_path, subject_id, "LARDI_data", 'data_b1000_g32_bvals')

        brain_mask_path = os.path.join(hcp_dataset_path, subject_id, "brain_mask.nii.gz")

        ss3t_csd_fod_path = os.path.join(hcp_dataset_path, subject_id, "ss3t_csd")
        msmt_csd_fod_path = os.path.join(hcp_dataset_path, subject_id, "msmt_csd")

        create_fods(LARDI_dwi_path, ss3t_csd_fod_path, LARDI_bvecs_path, LARDI_bvals_path, brain_mask_path, csd_type="ss3t_csd", normalisation=True)
        create_fods(HARDI_dwi_path, msmt_csd_fod_path, HARDI_bvecs_path, HARDI_bvals_path, brain_mask_path, csd_type="msmt_csd", normalisation=True)

if __name__ == "__main__":
    main()
