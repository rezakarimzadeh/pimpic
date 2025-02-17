import os
from monai.transforms import (
    LoadImage,
    Compose,
    EnsureChannelFirst,
    Lambda, 
    SpatialCrop,
    Spacing,
    Orientation,
    ScaleIntensityRange,
    RandGaussianNoise,
)
from monai.data import Dataset, DataLoader
import numpy as np
from monai.data import NibabelReader
import torch
from .data_handler import read_json_file
import math

class RandCropWithCoords:
    def __init__(self, roi_size, num_samples):
        self.roi_size = roi_size
        self.num_samples = num_samples
        self.roi_area = math.prod(roi_size)

    def __call__(self, image):
        patches, coords = [], []
        end_coord_limits = [int(s)-r for s, r in zip(image.shape[-3:], self.roi_size)]
        random_start_point=[np.random.randint(0, high=end_coord_s) for end_coord_s in end_coord_limits]
        for _ in range(self.num_samples):
            roi_start=[np.random.randint(max(0,rsp-4*roi_dim//5) , high=min(ecl, rsp+4*roi_dim//5)) for rsp, ecl, roi_dim in zip(random_start_point, end_coord_limits, self.roi_size)]
            roi_end=[s+self.roi_size[i] for i, s in enumerate(roi_start)]
            self.cropper = SpatialCrop(roi_size=self.roi_size, roi_start=roi_start, roi_end=roi_end)
            patch =  self.cropper(image)
            patches.append(patch)
            coords.append((roi_start, roi_end))
        intersection_matrix = self.calculate_intersection(coords)
        return patches, intersection_matrix, coords, image.shape
    
    def calculate_intersection(self, coords):
        intersection_matrix = np.zeros((self.num_samples, self.num_samples))
        
        for i in range(self.num_samples):
            for j in range(self.num_samples):
                if i == j:
                    # The intersection of a patch with itself is the entire patch volume
                    intersection_matrix[i, j] = np.prod(
                        [end - start for start, end in zip(coords[i][0], coords[i][1])]
                    )/self.roi_area
                else:
                    # Calculate the intersection of two patches
                    overlap = 1
                    for dim in range(len(coords[i][0])):  # Iterate through x, y, z dimensions
                        start_max = max(coords[i][0][dim], coords[j][0][dim])
                        end_min = min(coords[i][1][dim], coords[j][1][dim])
                        if start_max < end_min:
                            overlap *= (end_min - start_max)
                        else:
                            overlap = 0
                            break
                    intersection_matrix[i, j] = overlap/self.roi_area
        return intersection_matrix


def stack_patches_collate(batch):
    """
    Custom collate function to stack patches and coordinates as a batch.
    Args:
        batch: List of tuples, where each tuple contains:
               - patches: List of patches (torch.Tensor or ndarray).
               - coords: List of corresponding coordinates.
    Returns:
        stacked_patches: Tensor of shape [B * N, C, H, W, D], where:
                         B is the batch size, N is the number of patches per item.
        stacked_coords: Tensor of shape [B * N, 3, 2], where:
                        Each entry is the coordinate range for a patch.
    """
    all_patches = []
    all_coords = []
    all_intersection_matrices = []

    for patches, intersection, coords, img_shape in batch:
        # Ensure patches are cloned and detached properly
        all_patches.extend([p for p in patches])
        all_coords.extend(coords)
        all_intersection_matrices.extend(intersection)

    # Stack patches and coordinates into tensors
    stacked_patches = torch.stack(all_patches)
    stacked_coords = torch.tensor(all_coords)
    stacked_intersection = torch.tensor(np.array(all_intersection_matrices))
    return stacked_patches, stacked_intersection, stacked_coords, img_shape[1:]


def get_training_dataloaders(config):
    '''
    This function returns a dataloader which takes an image and randomly selects #batch, patches in the 3d image
    outputs for dataloader:
            - stacked_patches
            - stacked_coords
            - image shape
    '''
    roi_size = config["roi_size"]  # Desired patch size
    num_samples = config["batch_size"]  # Number of patches to extract per image

    transforms = Compose([
        LoadImage(image_only=True, reader=NibabelReader()),  # Use the correct reader
        EnsureChannelFirst(),
        Spacing(pixdim=config['spacing'], mode="bilinear"),
        Orientation(axcodes="RAS"),
        ScaleIntensityRange(a_min=config["min_intensity"], a_max=config["max_intensity"], b_min=0.0, b_max=1.0, clip=True),
        RandGaussianNoise(prob=0.9, mean=0.0, std=0.01),
        Lambda(lambda img: RandCropWithCoords(roi_size=roi_size, num_samples=num_samples)(img)),
    ])

    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'decathlon_pancreas_splits.json') 
    datadirs = read_json_file(file_path)
    train_100_percent = datadirs['train_100_percent']
    images = [os.path.join(config["data_root_dir"], dir['image']) for dir in train_100_percent]
    
    dataset = Dataset(data=images, transform=transforms)

    # Create DataLoader
    train_dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=True, collate_fn=stack_patches_collate)
    return train_dataloader


