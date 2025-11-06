"""
Blocks for building 2D and 3D U-Net architectures.

Classes:
    DoubleConv(nn.Module):
        Applies two consecutive 2D convolutional layers, each followed by batch normalization and ReLU activation.

    DoubleConv3D(nn.Module):
        Applies two consecutive 3D convolutional layers, each followed by batch normalization and ReLU activation.

    Down(nn.Module):
        Downsamples input using MaxPool2d, followed by a DoubleConv block.

    Down3D(nn.Module):
        Downsamples input using MaxPool3d, followed by a DoubleConv3D block.

    Up(nn.Module):
        Upsamples input using ConvTranspose2d, concatenates with skip connection, followed by a DoubleConv block.
        Handles cropping of skip connection to match upsampled tensor size.

    Up3D(nn.Module):
        Upsamples input using ConvTranspose3d, concatenates with skip connection, followed by a DoubleConv3D block.
        Handles cropping and padding of skip connection to match upsampled tensor size.

    OutConv(nn.Module):
        Final 2D convolutional layer with kernel size 1, typically used for output layer.

    OutConv3D(nn.Module):
        Final 3D convolutional layer with kernel size 1, typically used for output layer.

"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(convolution => [Bn] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleConv3D(nn.Module):
    """(convolution => [Bn3d] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        # Calculate padding for 'same' convolution in 3D
        if isinstance(kernel_size, int):
            padding = kernel_size // 2
        elif isinstance(kernel_size, tuple) and len(kernel_size) == 3:
            padding = tuple(k // 2 for k in kernel_size)
        else:
            raise ValueError("kernel_size must be int or 3-tuple")

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),  # Downsample by factor of 2
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Down3D(nn.Module):
    """Downscaling with maxpool3d then double conv3d"""

    def __init__(self, in_channels, out_channels, conv_kernel_size=3):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),  # Halve D, H, W
            DoubleConv3D(in_channels, out_channels, kernel_size=conv_kernel_size),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        x1: tensor from the previous layer
        x2: tensor from the encoder layer (skip connection)
        """
        x1 = self.up(x1)

        target_h, target_w = x1.shape[2], x1.shape[3]
        skip_h, skip_w = x2.shape[2], x2.shape[3]

        # Crop step to ensure that x1 and x2 have the same shape
        if skip_h > target_h or skip_w > target_w:
            diff_y = skip_h - target_h
            diff_x = skip_w - target_w

            crop_top = diff_y // 2
            crop_bottom = diff_y - crop_top
            crop_left = diff_x // 2
            crop_right = diff_x - crop_left

            if crop_bottom < 0:
                crop_bottom = 0
            if crop_right < 0:
                crop_right = 0

            x2 = x2[
                :, :, crop_top : skip_h - crop_bottom, crop_left : skip_w - crop_right
            ]

        x = torch.cat([x1, x2], dim=1)  # Concatenate along the channel dimension
        return self.conv(x)


class Up3D(nn.Module):
    """Upscaling with conv_transpose3d then double conv3d"""

    def __init__(self, in_channels, out_channels, conv_kernel_size=3):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv3D(
            out_channels + out_channels, out_channels, kernel_size=conv_kernel_size
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Handle size mismatches (cropping or padding x2 to match x1)
        target_d, target_h, target_w = x1.shape[2:]
        skip_d, skip_h, skip_w = x2.shape[2:]

        if any(
            s != t for s, t in zip(x2.shape[2:], x1.shape[2:])
        ):  # Cropping helpers
            diff_d = skip_d - target_d
            diff_h = skip_h - target_h
            diff_w = skip_w - target_w

            # Calculate cropping (similar to 2D, but for D, H, W)
            crop_d_front = diff_d // 2 if diff_d > 0 else 0
            crop_d_back = (diff_d - crop_d_front) if diff_d > 0 else 0
            crop_h_top = diff_h // 2 if diff_h > 0 else 0
            crop_h_bottom = (diff_h - crop_h_top) if diff_h > 0 else 0
            crop_w_left = diff_w // 2 if diff_w > 0 else 0
            crop_w_right = (diff_w - crop_w_left) if diff_w > 0 else 0

            x2_processed = x2[
                :,
                :,
                crop_d_front : skip_d - crop_d_back,
                crop_h_top : skip_h - crop_h_bottom,
                crop_w_left : skip_w - crop_w_right,
            ]

            # Check if padding is needed (if skip was smaller, after potential crop)
            current_d, current_h, current_w = x2_processed.shape[2:]
            pad_d = target_d - current_d
            pad_h = target_h - current_h
            pad_w = target_w - current_w

            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                # F.pad format: (pad_W_left, pad_W_right, pad_H_top, pad_H_bottom, pad_D_front, pad_D_back)
                x2_processed = F.pad(
                    x2_processed,
                    [
                        pad_w // 2,
                        pad_w - pad_w // 2,
                        pad_h // 2,
                        pad_h - pad_h // 2,
                        pad_d // 2,
                        pad_d - pad_d // 2,
                    ],
                )
            x2 = x2_processed

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class OutConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
