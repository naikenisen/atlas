"""
This module defines 2D and 3D U-Net architectures for image segmentation tasks, along with a utility function for loading model weights.

Classes:
    UNet(nn.Module):
        Implements a standard 2D U-Net architecture.
            n_channels (int): Number of input channels.
            n_classes (int)
            : Number of output classes.
        Methods:
            forward(x): Forward pass through the network.

    UNet3D(nn.Module):
        Implements a 3D U-Net architecture for volumetric data.
            n_channels (int): Number of input channels.
            n_classes (int): Number of output classes.
            feature_scale_factor (int, optional): Scales the number of feature channels. Default is 1.
        Methods:
            forward(x): Forward pass through the network.

Functions:
    load_model(
        model_path: str,
        ModelClass: type,
        device: torch.device,
        **model_init_kwargs
    ) -> nn.Module:
        Loads a PyTorch model from a saved state_dict file.
            model_path (str): Path to the model's state_dict file (.pth or .pt).
            ModelClass (type): The model class to instantiate (e.g., UNet or UNet3D).
            device (torch.device): Device to load the model onto ('cpu' or 'cuda').
            **model_init_kwargs: Additional keyword arguments for model initialization.
        Returns:
            nn.Module: The loaded model in evaluation mode, or None if loading fails.
"""

from src.model.blocks import (
    DoubleConv,
    DoubleConv3D,
    Down,
    Down3D,
    Up,
    Up3D,
    OutConv,
    OutConv3D,
)
import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        f = [64, 128, 256, 512, 1024]
        # encoder
        self.inc = DoubleConv(n_channels, f[0])
        self.down1 = Down(f[0], f[1])
        self.down2 = Down(f[1], f[2])
        self.down3 = Down(f[2], f[3])
        self.down4 = Down(f[3], f[4])

        # decoder
        self.up1 = Up(f[4], f[3])
        self.up2 = Up(f[3], f[2])
        self.up3 = Up(f[2], f[1])
        self.up4 = Up(f[1], f[0])

        # output
        self.outc = OutConv(f[0], n_classes)

    def forward(self, x):
        # encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # decoder
        d1 = self.up1(x5, x4)  # x4, x5, ... are the skip connection
        d2 = self.up2(d1, x3)
        d3 = self.up3(d2, x2)
        d4 = self.up4(d3, x1)

        # output
        out = self.outc(d4)
        return out


class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes, feature_scale_factor=1):
        super().__init__()
        self.n_channels_in = n_channels
        self.n_classes_out = n_classes

        # Standard U-Net feature progression, can be scaled
        f = [int(s * feature_scale_factor) for s in [64, 128, 256, 512]]

        # encoder
        self.inc = DoubleConv3D(n_channels, f[0])
        self.down1 = Down3D(f[0], f[1])
        self.down2 = Down3D(f[1], f[2])
        self.down3 = Down3D(f[2], f[3])

        # decoder
        self.up1 = Up3D(f[3], f[2])
        self.up2 = Up3D(f[2], f[1])
        self.up3 = Up3D(f[1], f[0])

        # output
        self.outc = OutConv3D(f[0], n_classes)

    def forward(self, x):
        # encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4_bottleneck = self.down3(x3)

        # decoder
        d = self.up1(x4_bottleneck, x3)
        d = self.up2(d, x2)
        d = self.up3(d, x1)

        logits = self.outc(d)
        return logits


def load_model(
    model_path: str,
    ModelClass: type,  # e.g., UNet or UNet3D
    device: torch.device,
    **model_init_kwargs,  # Arguments needed to initialize ModelClass (e.g., n_channels, n_classes_out)
) -> nn.Module:
    """
    Loads a PyTorch model from a saved state_dict.

    Args:
        model_path (str): Path to the .pth or .pt file containing the model's state_dict.
        ModelClass (type): The class of the model architecture (e.g., UNet).
        device (torch.device): The device to load the model onto ('cpu' or 'cuda').
        **model_init_kwargs: Keyword arguments required to initialize the ModelClass.
                             Example: n_channels_in=1, n_classes_out=1

    Returns:
        torch.nn.Module: The loaded model, in evaluation mode.
                         Returns None if loading fails.
    """
    print(f"Attempting to load model from: {model_path}")
    try:
        model = ModelClass(**model_init_kwargs)
        print(
            f"Instantiated model class: {ModelClass.__name__} with args: {model_init_kwargs}"
        )

        state_dict = torch.load(model_path, map_location=device)
        print(f"Successfully loaded state_dict from path.")

        model.load_state_dict(state_dict)
        print(f"Successfully loaded state_dict into the model instance.")

        model.to(device)
        print(f"Model moved to device: {device}")

        model.eval()
        print(f"Model set to evaluation mode.")

        return model
    except Exception as e:
        print(f"An unexpected error occurred during model loading: {e}")
        return None
