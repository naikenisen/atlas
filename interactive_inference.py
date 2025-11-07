#!/usr/bin/env python3
"""
Interactive 3D U-Net Inference Script
Performs inference using trained U-Net model with interactive threshold adjustment.
"""

import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.widgets as widgets


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


class InteractiveVisualizer:
    def __init__(self, original_data, prediction):
        self.original_data = original_data
        self.prediction = prediction
        self.current_slice = original_data.shape[2] // 2
        self.current_threshold = 0.1  # Start with a lower threshold
        
        # Create figure and subplots
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 10))
        self.fig.suptitle('3D U-Net Inference Results', fontsize=16)
        
        # Create slider axes
        ax_slice = plt.axes([0.2, 0.02, 0.5, 0.03])
        ax_threshold = plt.axes([0.2, 0.06, 0.5, 0.03])
        
        # Create sliders
        self.slice_slider = widgets.Slider(
            ax_slice, 'Slice', 0, original_data.shape[2]-1, 
            valinit=self.current_slice, valfmt='%d'
        )
        self.threshold_slider = widgets.Slider(
            ax_threshold, 'Threshold', 0.0, 1.0, 
            valinit=self.current_threshold, valfmt='%.3f'
        )
        
        # Connect slider events
        self.slice_slider.on_changed(self.update_slice)
        self.threshold_slider.on_changed(self.update_threshold)
        
        # Initial plot
        self.update_plot()
        
    def update_slice(self, val):
        self.current_slice = int(self.slice_slider.val)
        self.update_plot()
        
    def update_threshold(self, val):
        self.current_threshold = self.threshold_slider.val
        self.update_plot()
        
    def update_plot(self):
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
            
        # Get current slice
        img_slice = self.original_data[:, :, self.current_slice]
        pred_slice = self.prediction[:, :, self.current_slice]
        mask = pred_slice > self.current_threshold
        
        # Plot original image
        self.axes[0, 0].imshow(img_slice.T, cmap='gray', origin='lower')
        self.axes[0, 0].set_title('Original Image')
        self.axes[0, 0].axis('off')
        
        # Plot prediction probabilities
        self.axes[0, 1].imshow(img_slice.T, cmap='gray', origin='lower')
        im = self.axes[0, 1].imshow(pred_slice.T, cmap='Reds', alpha=0.6, origin='lower', vmin=0, vmax=1)
        self.axes[0, 1].set_title('Prediction Probabilities')
        self.axes[0, 1].axis('off')
        
        # Plot thresholded segmentation
        self.axes[1, 0].imshow(img_slice.T, cmap='gray', origin='lower')
        if mask.any():
            red_cmap = ListedColormap(['none', 'red'])
            self.axes[1, 0].imshow(mask.T, cmap=red_cmap, alpha=0.7, origin='lower')
        self.axes[1, 0].set_title(f'Segmentation (threshold={self.current_threshold:.3f})')
        self.axes[1, 0].axis('off')
        
        # Plot histogram of prediction values
        self.axes[1, 1].hist(pred_slice.flatten(), bins=50, alpha=0.7, edgecolor='black')
        self.axes[1, 1].axvline(self.current_threshold, color='red', linestyle='--', linewidth=2)
        self.axes[1, 1].set_xlabel('Prediction Probability')
        self.axes[1, 1].set_ylabel('Number of Voxels')
        self.axes[1, 1].set_title('Histogram of Predictions')
        self.axes[1, 1].grid(True, alpha=0.3)
        
        # Add statistics
        positive_voxels = np.sum(mask)
        total_voxels = mask.size
        percentage = (positive_voxels / total_voxels) * 100
        max_prob = np.max(pred_slice)
        min_prob = np.min(pred_slice)
        
        stats_text = f'Slice: {self.current_slice}/{self.original_data.shape[2]-1}\n'
        stats_text += f'Segmented: {positive_voxels:,} / {total_voxels:,} ({percentage:.2f}%)\n'
        stats_text += f'Prob range: [{min_prob:.3f}, {max_prob:.3f}]'
        
        self.fig.text(0.75, 0.15, stats_text, fontsize=10, 
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        self.fig.canvas.draw()


def analyze_prediction_statistics(prediction):
    """Analyze and print prediction statistics."""
    print("\n=== Prediction Statistics ===")
    print(f"Min probability: {prediction.min():.6f}")
    print(f"Max probability: {prediction.max():.6f}")
    print(f"Mean probability: {prediction.mean():.6f}")
    print(f"Std probability: {prediction.std():.6f}")
    
    # Analyze different thresholds
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    print("\nVoxels above different thresholds:")
    for thresh in thresholds:
        count = np.sum(prediction > thresh)
        percentage = (count / prediction.size) * 100
        print(f"  {thresh:.1f}: {count:,} voxels ({percentage:.2f}%)")


def main():
    # Configuration
    MODEL_PATH = "/home/naiken/coding/atlas/best_unet3d.pt"
    IMAGE_PATH = "/home/naiken/coding/atlas/atlas-train-dataset-1.0.1/train/imagesTr/im20.nii.gz"
    
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
    
    # Analyze statistics
    analyze_prediction_statistics(prediction)
    
    # Create interactive visualizer
    print("\nCreating interactive visualization...")
    print("Use the sliders to adjust slice and threshold!")
    
    visualizer = InteractiveVisualizer(original_data, prediction)
    plt.show()
    
    print("\nInference completed!")


if __name__ == "__main__":
    main()