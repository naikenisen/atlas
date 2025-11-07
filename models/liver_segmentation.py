import os
import random
from glob import glob
from typing import List, Tuple

import numpy as np
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Reprendre les classes ConvBlock et UNet3D de votre notebook
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

class LiverSegDataset(Dataset):
    """Dataset pour la segmentation du FOIE (label 1)"""
    def __init__(self, images: List[str], labels: List[str], patch_size: Tuple[int,int,int]=None, augment: bool=True):
        assert len(images) == len(labels)
        self.images = images
        self.labels = labels
        self.patch_size = patch_size
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = nib.load(self.images[idx]).get_fdata(dtype=np.float32)
        lbl = nib.load(self.labels[idx]).get_fdata(dtype=np.float32)

        # SEGMENTATION DU FOIE : label 1 OU 2 (foie + tumeur = foie complet)
        liver_mask = ((lbl == 1) | (lbl == 2)).astype(np.uint8)
        
        # Normalisation
        img = (img - img.mean()) / (img.std() + 1e-7)

        img = np.expand_dims(img, 0).astype(np.float32)
        liver_mask = np.expand_dims(liver_mask, 0).astype(np.float32)

        img_t = torch.from_numpy(img)
        lbl_t = torch.from_numpy(liver_mask)

        if self.patch_size is not None:
            img_t, lbl_t = self.random_crop(img_t, lbl_t, self.patch_size)

        if self.augment:
            img_t, lbl_t = self.random_flip(img_t, lbl_t)

        return img_t, lbl_t

    @staticmethod
    def random_crop(img: torch.Tensor, lbl: torch.Tensor, size: Tuple[int,int,int]):
        _, D, H, W = img.shape
        sd, sh, sw = size
        
        pad_d = max(0, sd - D)
        pad_h = max(0, sh - H)
        pad_w = max(0, sw - W)
        
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, pad_w, 0, pad_h, 0, pad_d))
            lbl = F.pad(lbl, (0, pad_w, 0, pad_h, 0, pad_d))
            _, D, H, W = img.shape
        
        d1 = random.randint(0, max(0, D - sd))
        h1 = random.randint(0, max(0, H - sh))
        w1 = random.randint(0, max(0, W - sw))
        
        img_crop = img[:, d1:d1+sd, h1:h1+sh, w1:w1+sw]
        lbl_crop = lbl[:, d1:d1+sd, h1:h1+sh, w1:w1+sw]
        
        return img_crop, lbl_crop

    @staticmethod
    def random_flip(img: torch.Tensor, lbl: torch.Tensor):
        for axis in range(1, 4):
            if random.random() > 0.5:
                img = torch.flip(img, [axis])
                lbl = torch.flip(lbl, [axis])
        return img, lbl

# ENTRAINEMENT DU RÉSEAU FOIE
def train_liver_segmentation():
    # Configuration spécifique au foie
    IMAGES_DIR = "/home/naiken/coding/atlas/atlas-train-dataset-1.0.1/train/imagesTr"
    LABELS_DIR = "/home/naiken/coding/atlas/atlas-train-dataset-1.0.1/train/labelsTr"
    EPOCHS = 15
    BATCH_SIZE = 1
    PATCH_SIZE = (64, 64, 32)
    VAL_SPLIT = 0.2
    LR = 1e-4
    SAVE_PATH = "weights/best_liver_unet3d.pt"
    
    # Créer le dossier weights
    os.makedirs("weights", exist_ok=True)
    
    # Vos fonctions existantes
    def find_pairs(images_dir, labels_dir):
        imgs = sorted(glob(os.path.join(images_dir, "im*.nii*")))
        lbls = sorted(glob(os.path.join(labels_dir, "lb*.nii*")))
        
        base_to_img = {}
        for p in imgs:
            basename = os.path.basename(p)
            base_num = basename.replace('im', '').replace('.nii.gz', '').replace('.nii', '')
            base_to_img[base_num] = p
        
        base_to_lbl = {}
        for p in lbls:
            basename = os.path.basename(p)
            base_num = basename.replace('lb', '').replace('.nii.gz', '').replace('.nii', '')
            base_to_lbl[base_num] = p
        
        common = set(base_to_img) & set(base_to_lbl)
        return [base_to_img[b] for b in sorted(common, key=int)], [base_to_lbl[b] for b in sorted(common, key=int)]

    class DiceBCELoss(nn.Module):
        def forward(self, inputs, targets, smooth=1):
            inputs = torch.sigmoid(inputs)
            inputs = inputs.view(-1)
            targets = targets.view(-1)
            intersection = (inputs * targets).sum()
            dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
            BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
            return BCE + (1 - dice)

    def dice_coefficient(pred, target, smooth=1e-6):
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()
        intersection = (pred * target).sum()
        return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    def train_one_epoch(model, loader, optimizer, loss_fn, device):
        model.train()
        total_loss = 0
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = loss_fn(logits, lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def evaluate(model, loader, device):
        model.eval()
        dices = []
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                logits = model(imgs)
                dices.append(dice_coefficient(logits, lbls))
        return float(np.mean(dices))

    # Setup data
    images, labels = find_pairs(IMAGES_DIR, LABELS_DIR)
    print(f"🥇 ÉTAPE 1: Entraînement segmentation FOIE")
    print(f"Found {len(images)} image/label pairs")
    
    # Split data
    idx = list(range(len(images)))
    random.shuffle(idx)
    split = int(len(idx)*(1-VAL_SPLIT))
    train_idx, val_idx = idx[:split], idx[split:]

    # Create datasets
    train_ds = LiverSegDataset([images[i] for i in train_idx], [labels[i] for i in train_idx], patch_size=PATCH_SIZE)
    val_ds = LiverSegDataset([images[i] for i in val_idx], [labels[i] for i in val_idx], patch_size=PATCH_SIZE, augment=False)

    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # Setup model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model = UNet3D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = DiceBCELoss()

    # Training loop
    best_val_dice = 0
    for epoch in range(1, EPOCHS + 1):
        print(f"\n=== Liver Epoch {epoch}/{EPOCHS} ===")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_dice = evaluate(model, val_loader, device)
        
        print(f"Train Loss: {train_loss:.4f}, Val Dice: {val_dice:.4f}")
        
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_dice': val_dice
            }, SAVE_PATH)
            print(f"✓ Saved best liver model (val_dice={val_dice:.4f})")

    print(f"\n🎉 Liver training complete! Best validation Dice: {best_val_dice:.4f}")
    return SAVE_PATH

if __name__ == "__main__":
    train_liver_segmentation()