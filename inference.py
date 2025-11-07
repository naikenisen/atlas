#!/usr/bin/env python3
"""
3D U-Net Inference Script
Performs inference using trained U-Net model and visualizes results with adjustable threshold.
"""

import os
import argparse
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down1 = ConvBlock(in_channels, 32)
        self.down2 = ConvBlock(32, 64)
        self.down3 = ConvBlock(64, 128)
        self.down4 = ConvBlock(128, 256)

        self.center = ConvBlock(256, 512)

        self.up4 = nn.ConvTranspose3d(512, 256, 2, stride=2)
        self.upconv4 = ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.upconv3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.upconv2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.upconv1 = ConvBlock(64, 32)

        self.out = nn.Conv3d(32, out_channels, 1)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        x4 = self.down4(self.pool(x3))

        center = self.center(self.pool(x4))

        up = self.up4(center)
        up = torch.cat([up, x4], dim=1)
        up = self.upconv4(up)

        up = self.up3(up)
        up = torch.cat([up, x3], dim=1)
        up = self.upconv3(up)

        up = self.up2(up)
        up = torch.cat([up, x2], dim=1)
        up = self.upconv2(up)

        up = self.up1(up)
        up = torch.cat([up, x1], dim=1)
        up = self.upconv1(up)

        return self.out(up)


def load_model(model_path, device):
    """Load the trained U-Net model."""
    print(f"Loading model from {model_path}")
    model = UNet3D().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f"Model loaded successfully (validation dice: {checkpoint.get('val_dice', 'N/A')})")
    return model


def preprocess_image(image_path):
    """Load and preprocess NIfTI image for inference."""
    print(f"Loading image: {image_path}")
    img = nib.load(image_path)
    data = img.get_fdata(dtype=np.float32)
    original_shape = data.shape
    print(f"Original image shape: {original_shape}")
    
    # Normalize image
    data = (data - data.mean()) / (data.std() + 1e-7)
    
    # Pad to make dimensions divisible by 16 (for U-Net with 4 downsampling levels)
    def pad_to_multiple(size, multiple=16):
        remainder = size % multiple
        if remainder != 0:
            padding = multiple - remainder
            return padding
        return 0
    
    pad_x = pad_to_multiple(data.shape[0])
    pad_y = pad_to_multiple(data.shape[1])
    pad_z = pad_to_multiple(data.shape[2])
    
    if pad_x > 0 or pad_y > 0 or pad_z > 0:
        data = np.pad(data, ((0, pad_x), (0, pad_y), (0, pad_z)), mode='constant', constant_values=0)
        print(f"Padded image shape: {data.shape}")
    
    # Add batch and channel dimensions
    data = np.expand_dims(data, 0)  # Add channel dimension
    data = np.expand_dims(data, 0)  # Add batch dimension
    
    return torch.from_numpy(data), img, original_shape


def perform_inference(model, image_tensor, device, original_shape):
    """Perform inference on the preprocessed image."""
    print("Performing inference...")
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits)
    
    # Remove batch dimension and convert to numpy
    probabilities = probabilities.squeeze(0).squeeze(0).cpu().numpy()
    
    # Crop back to original shape if padding was applied
    probabilities = probabilities[:original_shape[0], :original_shape[1], :original_shape[2]]
    
    return probabilities


def visualize_results(original_data, prediction, threshold=0.47, slice_idx=None, save_path=None):
    """
    Visualize original image with segmentation overlay.
    
    Args:
        original_data: Original image data (3D numpy array)
        prediction: Prediction probabilities (3D numpy array)
        threshold: Threshold for segmentation (0-1)
        slice_idx: Slice index to visualize (if None, uses middle slice)
        save_path: Path to save the visualization
    """
    print(f"Creating visualization with threshold: {threshold}")
    
    # Get middle slice if not specified
    if slice_idx is None:
        slice_idx = original_data.shape[2] // 2
    
    # Extract slice
    img_slice = original_data[:, :, slice_idx]
    pred_slice = prediction[:, :, slice_idx]
    
    # Apply threshold
    mask = pred_slice > threshold
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img_slice.T, cmap='gray', origin='lower')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Prediction heatmap
    axes[1].imshow(img_slice.T, cmap='gray', origin='lower')
    im = axes[1].imshow(pred_slice.T, cmap='Reds', alpha=0.6, origin='lower', vmin=0, vmax=1)
    axes[1].set_title(f'Prediction Probabilities')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Thresholded overlay
    axes[2].imshow(img_slice.T, cmap='gray', origin='lower')
    if mask.any():  # Only show overlay if there are positive predictions
        # Create a red colormap for the mask
        red_cmap = ListedColormap(['none', 'red'])
        axes[2].imshow(mask.T, cmap=red_cmap, alpha=0.7, origin='lower')
    axes[2].set_title(f'Segmentation (threshold={threshold})')
    axes[2].axis('off')
    
    plt.suptitle(f'Slice {slice_idx} / {original_data.shape[2]-1}', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()
    
    # Print some statistics
    positive_voxels = np.sum(mask)
    total_voxels = mask.size
    percentage = (positive_voxels / total_voxels) * 100
    print(f"Segmented voxels: {positive_voxels:,} / {total_voxels:,} ({percentage:.2f}%)")


def main():
    # Configuration - you can modify these values
    MODEL_PATH = "/home/naiken/coding/atlas/best_unet3d.pt"
    IMAGE_PATH = "/home/naiken/coding/atlas/atlas-train-dataset-1.0.1/train/imagesTr/im18.nii.gz"
    THRESHOLD = 0.35  # Adjust this value between 0 and 1
    SLICE_IDX = None  # None for middle slice, or specify slice number
    SAVE_PATH = "inference_result.png"  # None to not save
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Check if files exist
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Image file not found: {IMAGE_PATH}")
    
    # Load model
    model = load_model(MODEL_PATH, device)
    
    # Load and preprocess image
    image_tensor, original_nifti, original_shape = preprocess_image(IMAGE_PATH)
    original_data = original_nifti.get_fdata()
    
    # Perform inference
    prediction = perform_inference(model, image_tensor, device, original_shape)
    
    # Visualize results
    visualize_results(
        original_data, 
        prediction, 
        threshold=THRESHOLD, 
        slice_idx=SLICE_IDX,
        save_path=SAVE_PATH
    )
    
    print("\nInference completed!")
    print(f"To adjust the threshold, modify the THRESHOLD variable in the script (current: {THRESHOLD})")


if __name__ == "__main__":
    main()