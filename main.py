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
        uid_dice = calculate_dice(pred_data, truth_data)
        print(uid_dice)

        # Adding the dice for each slice
        dice += uid_dice
        
    print('dice score:', dice/len(truth_uid))


def calculate_dice(pred_data, truth_data):
    z_range = range(pred_data.shape[-1])
    z_len = len(z_range)
    dice_sum = 0
    for z in z_range:
        pred_slice = pred_data[:, :, z]
        truth_slice = truth_data[:, :, z]
        slice_dice = single_dice(pred_slice, truth_slice)
        dice_sum += slice_dice

    return dice_sum / z_len


def single_dice(pred, truth):
    intersection = np.sum(truth * pred)
    if (np.sum(truth) == 0) and (np.sum(pred) == 0):
        return 1
    return (2 * intersection) / (np.sum(truth) + np.sum(pred))


mask_dataset_import('data/preds/',
                    'data/truth/')
