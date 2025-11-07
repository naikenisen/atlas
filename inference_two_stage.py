"""
Inférence avec le pipeline à 2 étapes :
1. Segmentation du foie
2. Segmentation de la tumeur dans le foie
"""

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# Importer les modèles
import sys
sys.path.append('models')
from liver_segmentation import UNet3D

def two_stage_inference(image_path, liver_model_path, tumor_model_path, 
                       liver_threshold=0.5, tumor_threshold=0.3, slice_idx=None):
    """
    Inférence complète en 2 étapes
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Charger les modèles
    liver_model = UNet3D().to(device)
    tumor_model = UNet3D().to(device)
    
    liver_checkpoint = torch.load(liver_model_path, map_location=device)
    tumor_checkpoint = torch.load(tumor_model_path, map_location=device)
    
    liver_model.load_state_dict(liver_checkpoint['model_state'])
    tumor_model.load_state_dict(tumor_checkpoint['model_state'])
    
    liver_model.eval()
    tumor_model.eval()
    
    print(f"✅ Modèles chargés:")
    print(f"   - Foie: dice={liver_checkpoint['val_dice']:.4f}")
    print(f"   - Tumeur: dice={tumor_checkpoint['val_dice']:.4f}")
    
    # Charger l'image
    img = nib.load(image_path)
    original_data = img.get_fdata(dtype=np.float32)
    
    # Préprocessing
    normalized_data = (original_data - original_data.mean()) / (original_data.std() + 1e-7)
    input_tensor = torch.from_numpy(normalized_data[None, None, ...]).to(device)
    
    # Padding pour compatibilité U-Net
    target_size = [64, 64, 32]
    pad_needed = [max(0, target_size[i] - input_tensor.shape[i+2]) for i in range(3)]
    if any(p > 0 for p in pad_needed):
        input_tensor = F.pad(input_tensor, 
                           (0, pad_needed[2], 0, pad_needed[1], 0, pad_needed[0]))
    
    # ÉTAPE 1: Segmentation du foie
    with torch.no_grad():
        liver_logits = liver_model(input_tensor)
        liver_probabilities = torch.sigmoid(liver_logits)
        liver_mask = (liver_probabilities > liver_threshold).float()
    
    # ÉTAPE 2: Masquer l'image avec le foie et segmenter la tumeur
    masked_input = input_tensor * liver_mask
    
    with torch.no_grad():
        tumor_logits = tumor_model(masked_input)
        tumor_probabilities = torch.sigmoid(tumor_logits)
    
    # Retour aux dimensions originales
    liver_pred = liver_probabilities.cpu().numpy()[0, 0]
    tumor_pred = tumor_probabilities.cpu().numpy()[0, 0]
    
    if any(p > 0 for p in pad_needed):
        liver_pred = liver_pred[:original_data.shape[0], :original_data.shape[1], :original_data.shape[2]]
        tumor_pred = tumor_pred[:original_data.shape[0], :original_data.shape[1], :original_data.shape[2]]
    
    # Choisir la slice
    if slice_idx is None:
        slice_idx = original_data.shape[2] // 2
    
    # Extraire les slices
    original_slice = original_data[:, :, slice_idx]
    liver_slice = liver_pred[:, :, slice_idx]
    tumor_slice = tumor_pred[:, :, slice_idx]
    
    liver_binary = liver_slice > liver_threshold
    tumor_binary = tumor_slice > tumor_threshold
    
    # Créer la visualisation
    plt.figure(figsize=(20, 5))
    
    # Image originale
    plt.subplot(1, 4, 1)
    plt.imshow(np.rot90(original_slice), cmap="gray")
    plt.title("Image Originale")
    plt.axis("off")
    
    # Segmentation du foie
    plt.subplot(1, 4, 2)
    plt.imshow(np.rot90(original_slice), cmap="gray", alpha=0.8)
    plt.imshow(np.rot90(liver_binary), cmap="Greens", alpha=0.6)
    plt.title(f"Foie (t={liver_threshold})")
    plt.axis("off")
    
    # Segmentation de la tumeur
    plt.subplot(1, 4, 3)
    plt.imshow(np.rot90(original_slice), cmap="gray", alpha=0.8)
    plt.imshow(np.rot90(tumor_binary), cmap="Reds", alpha=0.6)
    plt.title(f"Tumeur (t={tumor_threshold})")
    plt.axis("off")
    
    # Segmentation complète
    plt.subplot(1, 4, 4)
    plt.imshow(np.rot90(original_slice), cmap="gray", alpha=0.8)
    plt.imshow(np.rot90(liver_binary), cmap="Greens", alpha=0.4)
    plt.imshow(np.rot90(tumor_binary), cmap="Reds", alpha=0.8)
    plt.title("Foie + Tumeur")
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(f"two_stage_inference_lt{liver_threshold}_tt{tumor_threshold}_s{slice_idx}.png", 
                dpi=150, bbox_inches='tight')
    plt.show()
    
    # Statistiques
    liver_pixels = liver_binary.sum()
    tumor_pixels = tumor_binary.sum()
    total_pixels = liver_binary.size
    
    print(f"\n📊 Statistiques slice {slice_idx}:")
    print(f"   - Foie: {liver_pixels} pixels ({100*liver_pixels/total_pixels:.2f}%)")
    print(f"   - Tumeur: {tumor_pixels} pixels ({100*tumor_pixels/total_pixels:.2f}%)")
    if liver_pixels > 0:
        print(f"   - Tumeur/Foie: {100*tumor_pixels/liver_pixels:.2f}%")
    
    return liver_pred, tumor_pred

# UTILISATION
if __name__ == "__main__":
    # Tester l'inférence complète
    image_path = "atlas-train-dataset-1.0.1/train/imagesTr/im0.nii.gz"
    liver_model_path = "weights/best_liver_unet3d.pt"
    tumor_model_path = "weights/best_tumor_unet3d.pt"
    
    # Vérifier que les modèles existent
    import os
    if not os.path.exists(liver_model_path):
        print(f"❌ Modèle foie non trouvé: {liver_model_path}")
        print("Lancez d'abord: python train_two_stage.py")
    elif not os.path.exists(tumor_model_path):
        print(f"❌ Modèle tumeur non trouvé: {tumor_model_path}")
        print("Lancez d'abord: python train_two_stage.py")
    else:
        print("🚀 INFÉRENCE PIPELINE 2 ÉTAPES")
        liver_pred, tumor_pred = two_stage_inference(
            image_path, liver_model_path, tumor_model_path,
            liver_threshold=0.5, tumor_threshold=0.3
        )