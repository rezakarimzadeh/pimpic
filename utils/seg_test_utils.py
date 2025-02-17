import os
import torch
import numpy as np
from monai.transforms import LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd, ScaleIntensityRanged, ToTensord, Compose, CropForegroundd
from monai.data import CacheDataset, DataLoader
from monai.transforms import KeepLargestConnectedComponent
from monai.inferers import sliding_window_inference
from dataloaders import data_handler


def keep_largest_components(predicted):
    """
    Use MONAI's KeepLargestConnectedComponent to retain the largest connected component
    for pancreas (label 1) and tumor (label 2).

    Args:
        predicted (torch.Tensor): Tensor of shape [B, D, H, W] with predicted labels.

    Returns:
        torch.Tensor: Tensor with only the largest connected components for pancreas and tumor.
    """
    keep_largest_cc = KeepLargestConnectedComponent(applied_labels=[1, 2], connectivity=1)
    cleaned_predictions = []

    for pred in predicted:  # Loop over batch
        cleaned_pred = keep_largest_cc(pred)
        cleaned_predictions.append(cleaned_pred)

    return torch.stack(cleaned_predictions, dim=0)


def dice_score(pred, target, class_label):
    """
    Calculate Dice score for a specific class.
    
    Args:
        pred (torch.Tensor): Predicted tensor of shape [B, D, H, W].
        target (torch.Tensor): Ground truth tensor of shape [B, D, H, W].
        class_label (int): The class label for which to calculate Dice score.
    
    Returns:
        float: Dice score for the given class.
    """
    pred_class = (pred == class_label).float()
    target_class = (target == class_label).float()
    
    intersection = (pred_class * target_class).sum()
    union = pred_class.sum() + target_class.sum()
    
    dice = (2.0 * intersection) / (union + 1e-8)  # Add epsilon to avoid division by zero
    return dice.item()


def whole_pancreas_dice_fn(pred, target):
    """
    Calculate Dice score for the whole pancreas (pancreas + tumor).
    
    Args:
        pred (torch.Tensor): Predicted tensor of shape [B, D, H, W].
        target (torch.Tensor): Ground truth tensor of shape [B, D, H, W].
    
    Returns:
        float: Dice score for the whole pancreas.
    """
    pred_whole_pancreas = ((pred == 1) | (pred == 2)).float()  # Merge labels 1 and 2
    target_whole_pancreas = ((target == 1) | (target == 2)).float()
    
    intersection = (pred_whole_pancreas * target_whole_pancreas).sum()
    union = pred_whole_pancreas.sum() + target_whole_pancreas.sum()
    
    dice = (2.0 * intersection) / (union + 1e-8)  # Add epsilon to avoid division by zero
    return dice.item()


def evaluate_model(test_loader, test_datalist, model, device, roi_size):
    """
    Perform evaluation with connected component filtering and Dice score calculation.
    
    Args:
        val_loader (DataLoader): DataLoader for validation data.
        model (torch.nn.Module): Trained segmentation model.
        device (torch.device): Device for computation (e.g., 'cuda').
        roi_size (tuple): ROI size for sliding-window inference.
    
    Returns:
        dict: Dice scores for pancreas and tumor.
    """
    pancreas_dice = []
    tumor_dice = []
    whole_pancreas_dice = []
    image_names = []
    with torch.no_grad():
        for batch, img_dir in zip(test_loader, test_datalist):
            img, label = batch["image"].to(device), batch["label"].to(device)
            img_name = img_dir["image"].split('/')[-1]
            image_names.append(img_name)

            # Perform inference
            predicted_logits = inference(img, model, roi_size=roi_size)  # [B, C, D, H, W]
            predicted = torch.argmax(predicted_logits, dim=1)  # Convert to class labels [B, D, H, W]

            # Remove small components
            cleaned_predictions = keep_largest_components(predicted)

            # Calculate Dice scores for pancreas (label 1) and tumor (label 2)
            for i in range(cleaned_predictions.shape[0]):  # Iterate over batch
                pancreas_dice.append(dice_score(cleaned_predictions[i], label[i], class_label=1))
                tumor_dice.append(dice_score(cleaned_predictions[i], label[i], class_label=2))
                whole_pancreas_dice.append(whole_pancreas_dice_fn(cleaned_predictions[i], label[i]))
    
    # Calculate mean Dice scores
    dice_scores = {
        "pancreas_dice_mean": np.mean(pancreas_dice),
        "pancreas_dice_std": np.std(pancreas_dice),
        "tumor_dice_mean": np.mean(tumor_dice),
        "tumor_dice_std": np.std(tumor_dice),
        "whole_pancreas_dice_mean": np.mean(whole_pancreas_dice),
        "whole_pancreas_dice_std": np.std(whole_pancreas_dice),
    }

    evaluation_results = dict()
    evaluation_results["image_names"] = image_names
    evaluation_results["pancreas_dice"] = pancreas_dice
    evaluation_results["tumor_dice"] = tumor_dice
    evaluation_results["whole_pancreas_dice"] = whole_pancreas_dice
    evaluation_results["dice_scores"] = dice_scores
    return evaluation_results


def inference(input, model, roi_size):
    def _compute(input, model, roi_size):
        return sliding_window_inference(
        inputs=input,
        roi_size=roi_size,
        sw_batch_size=1,
        predictor=model,
        overlap=0.5,
    )
    with torch.amp.autocast('cuda'):
        return _compute(input, model, roi_size)


def add_root_dir(data_list, root_dir):
    """
    Adds a root directory to the 'image' and 'label' paths in a list of dictionaries.

    :param data_list: List of dictionaries containing 'image' and 'label' keys.
    :param root_dir: Root directory to prepend to the paths.
    :return: Updated list of dictionaries with modified paths.
    """
    for item in data_list:
        if 'image' in item:
            item['image'] = os.path.join(root_dir, item['image'])
        if 'label' in item:
            item['label'] = os.path.join(root_dir, item['label'])
    return data_list

def get_test_dataloader(config):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '..', 'dataloaders', 'decathlon_pancreas_splits.json') 
    datadirs = data_handler.read_json_file(file_path)

    test_dirs = datadirs['test']
    test_datalist = add_root_dir(test_dirs, config['data_root_dir'])

    test_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=config["spacing"], mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=config["min_intensity"], a_max=config["max_intensity"], b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image", "label"]),
    ])
    # test_ds = Dataset(data=test_datalist, transform=test_transforms)
    # test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)
    test_ds = CacheDataset(
        data=test_datalist,
        transform=test_transforms,
        cache_rate=1.0,
        num_workers=1,
    )
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=1)
    return test_loader, test_datalist
