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

# Réutiliser les mêmes classes ConvBlock et UNet3D
from liver_segmentation import ConvBlock, UNet3D

class TumorSegDataset(Dataset):
    """Dataset pour la segmentation de la TUMEUR dans le foie segmenté"""
    def __init__(self, images: List[str], labels: List[str], liver_model_path: str, 
                 patch_size: Tuple[int,int,int]=None, augment: bool=True):
        assert len(images) == len(labels)
        self.images = images
        self.labels = labels
        self.patch_size = patch_size
        self.augment = augment
        
        # Charger le modèle de segmentation du foie
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.liver_model = UNet3D().to(self.device)
        checkpoint = torch.load(liver_model_path, map_location=self.device)
        self.liver_model.load_state_dict(checkpoint['model_state'])
        self.liver_model.eval()
        print(f"✓ Loaded liver model from {liver_model_path}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = nib.load(self.images[idx]).get_fdata(dtype=np.float32)
        lbl = nib.load(self.labels[idx]).get_fdata(dtype=np.float32)

        # 1. Segmenter le foie avec le premier réseau
        img_normalized = (img - img.mean()) / (img.std() + 1e-7)
        img_tensor = torch.from_numpy(img_normalized[None, None, ...]).to(self.device)
        
        # Padding pour compatibilité
        target_size = [64, 64, 32]
        pad_needed = [max(0, target_size[i] - img_tensor.shape[i+2]) for i in range(3)]
        if any(p > 0 for p in pad_needed):
            img_tensor = F.pad(img_tensor, (0, pad_needed[2], 0, pad_needed[1], 0, pad_needed[0]))
        
        with torch.no_grad():
            liver_logits = self.liver_model(img_tensor)
            liver_prob = torch.sigmoid(liver_logits)
            liver_mask = (liver_prob > 0.5).float()
        
        # Retour aux dimensions originales
        liver_mask_np = liver_mask.cpu().numpy()[0, 0]
        if any(p > 0 for p in pad_needed):
            liver_mask_np = liver_mask_np[:img.shape[0], :img.shape[1], :img.shape[2]]

        # 2. Masquer l'image originale avec le foie segmenté
        masked_img = img_normalized * liver_mask_np
        
        # 3. Target: seulement la tumeur (label 2) DANS le foie
        tumor_mask = (lbl == 2).astype(np.uint8)
        
        # S'assurer que la tumeur est bien dans le foie segmenté
        tumor_mask = tumor_mask * liver_mask_np

        # Préparer les tensors
        masked_img = np.expand_dims(masked_img, 0).astype(np.float32)
        tumor_mask = np.expand_dims(tumor_mask, 0).astype(np.float32)

        img_t = torch.from_numpy(masked_img)
        lbl_t = torch.from_numpy(tumor_mask)

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

def train_tumor_segmentation(liver_model_path):
    # Configuration spécifique à la tumeur
    IMAGES_DIR = "/home/naiken/coding/atlas/atlas-train-dataset-1.0.1/train/imagesTr"
    LABELS_DIR = "/home/naiken/coding/atlas/atlas-train-dataset-1.0.1/train/labelsTr"
    EPOCHS = 20  # Plus d'époques pour la tumeur (plus difficile)
    BATCH_SIZE = 1
    PATCH_SIZE = (64, 64, 32)
    VAL_SPLIT = 0.2
    LR = 5e-5  # Learning rate plus petit
    SAVE_PATH = "weights/best_tumor_unet3d.pt"
    
    # Vos fonctions existantes (réutilisées)
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
    print(f"\n🥈 ÉTAPE 2: Entraînement segmentation TUMEUR")
    print(f"Found {len(images)} image/label pairs")
    
    # Split data
    idx = list(range(len(images)))
    random.shuffle(idx)
    split = int(len(idx)*(1-VAL_SPLIT))
    train_idx, val_idx = idx[:split], idx[split:]

    # Create datasets (avec le modèle du foie)
    train_ds = TumorSegDataset([images[i] for i in train_idx], [labels[i] for i in train_idx], 
                              liver_model_path, patch_size=PATCH_SIZE)
    val_ds = TumorSegDataset([images[i] for i in val_idx], [labels[i] for i in val_idx], 
                            liver_model_path, patch_size=PATCH_SIZE, augment=False)

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
        print(f"\n=== Tumor Epoch {epoch}/{EPOCHS} ===")
        
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
            print(f"✓ Saved best tumor model (val_dice={val_dice:.4f})")

    print(f"\n🎉 Tumor training complete! Best validation Dice: {best_val_dice:.4f}")
    return SAVE_PATH

if __name__ == "__main__":
    liver_model_path = "weights/best_liver_unet3d.pt"
    if not os.path.exists(liver_model_path):
        print("❌ Liver model not found! Train liver segmentation first.")
    else:
        train_tumor_segmentation(liver_model_path)