import os
import numpy as np
import nibabel as nib


# Import dataset
def mask_dataset_import(pred_path, truth_path):
    """
        Calculating the dice of predicted labels versus
        truths for a given dataset
        """
    # Lists all truth labels within the directory path provided
    truth_uid = os.listdir(truth_path)
    print(truth_uid)

    # Lists all predicted labels within the directory path provided
    pred_uid = os.listdir(pred_path)
    dice = 0
    for uid in truth_uid:
        print(uid)
        pred_file_path = os.path.join(pred_path, uid.replace('.nii', '_pred.nii.gz'))
        print(pred_file_path)
        truth_file_path = os.path.join(truth_path, uid)
        print(truth_file_path)

        # Obtaining array data of the predicted label
        pred_nib = nib.load(pred_file_path)
        pred_data = pred_nib.get_fdata()

        # Obtaining array data of the GT label
        truth_nib = nib.load(truth_file_path)
        truth_data = truth_nib.get_fdata()

        # Utilizing dice function above for each z slice
        uid_dice = dice_coefficient(pred_data, truth_data)
        print(uid_dice)


def dice_coefficient(seg_pred, seg_gt):
    """
    Compute the Dice Coefficient for binary segmentations.
    """
    intersection = np.logical_and(seg_pred, seg_gt).sum()
    volume_sum = seg_pred.sum() + seg_gt.sum()

    if volume_sum == 0:
        return 1.0 if intersection == 0 else 0.0  # Handle empty cases

    return 2.0 * intersection / volume_sum


mask_dataset_import('data/preds/',
                    'data/truth/')
