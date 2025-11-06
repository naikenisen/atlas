"""
Main training script for U-Net segmentation model.

This script performs the following steps:
1. Loads configuration parameters from environment variables or uses default values.
2. Loads patient data from the specified dataset path.
3. Splits the patient data into training and validation sets.
4. Initializes the NiftDataset for both training and validation sets.
5. Creates DataLoader instances for efficient batch processing.
6. Instantiates the U-Net model with the specified input channels and output classes.
7. Sets up the loss function (BCEWithLogitsLoss) and optimizer (AdamW).
8. Runs the training loop for a specified number of epochs:
    - Trains the model for one epoch.
    - Validates the model and computes loss and Dice score.
    - Saves the model if the validation Dice score improves.
    - Tracks and prints training/validation metrics.
9. Plots and saves the training history (losses and Dice scores).
10. Handles exceptions and prints error messages.

Environment Variables:
- DATASET_PATH: Path to the dataset folder.
- MODEL_SAVE_PATH: Path to save the trained model.
- INPUT_FILENAMES: Comma-separated list of input filenames.
- SLICE_AXIS: Axis along which to slice the data (0: x, 1: y, 2: z).
- BATCH_SIZE: Batch size for training and validation.
- NUM_WORKERS: Number of worker threads for data loading.
- LEARNING_RATE: Learning rate for the optimizer.
- VALIDATION_SPLIT: Fraction of data to use for validation.
- EPOCHS: Number of training epochs.
- RANDOM_SEED: Seed for reproducibility.

Raises:
    Exception: Any error encountered during training or data loading.
"""
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.dataset.data_loader import NiftDataset
from src.model.model import UNet
from src.utils import get_patients
from src.model.training_utils import train_epoch, validate_epoch
from src.visualisations.visualisation_utils import plot_metrics
import wandb  # Add this import
wandb.login(key="ab67e0f4c27fad7a0d47405f84a8a4deb80056ba")

if __name__ == "__main__":
    try:
        # --- Configuration ---
        DATASET_PATH = "/work/c-2iia/in156281/data_unet"
        MODEL_SAVE_PATH = "output/models/unet_model.pth"
        INPUT_FILENAMES = ["PET.nii.gz", "SEG.nii.gz"]
        SLICE_AXIS = 2  # 0: x, 1: y, 2: z
        BATCH_SIZE = 12
        NUM_WORKERS = 8
        LEARNING_RATE = 1e-3
        VALIDATION_SPLIT = 0.3
        NUM_EPOCHS = 20
        RANDOM_SEED = 42
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        NUM_OUTPUT_CHANNELS_MODEL = 1  # Do not change unless we want to detect more labels in the future
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        print(
            f"Configuration: \n"
            f"MODEL_SAVE_PATH: {MODEL_SAVE_PATH}\n"
            f"INPUT_FILENAMES: {INPUT_FILENAMES}\n"
            f"SLICE_AXIS: {SLICE_AXIS}\n"
            f"BATCH_SIZE: {BATCH_SIZE}\n"
            f"NUM_WORKERS: {NUM_WORKERS}\n"
            f"LEARNING_RATE: {LEARNING_RATE}\n"
            f"VALIDATION_SPLIT: {VALIDATION_SPLIT}\n"
            f"NUM_EPOCHS: {NUM_EPOCHS}\n"
            f"RANDOM_SEED: {RANDOM_SEED}\n"
            f"DEVICE: {DEVICE}\n"
        )

        # --- WandB Init ---
        wandb.init(
            project="unet_gang",
            config={
                "model_save_path": MODEL_SAVE_PATH,
                "input_filenames": INPUT_FILENAMES,
                "slice_axis": SLICE_AXIS,
                "batch_size": BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "learning_rate": LEARNING_RATE,
                "validation_split": VALIDATION_SPLIT,
                "num_epochs": NUM_EPOCHS,
                "random_seed": RANDOM_SEED,
                "device": str(DEVICE),
            }
        )

        # --- Load Patients ---
        patients = get_patients(data_folder_path=DATASET_PATH)

        # Sélectionner un quart des patients au hasard
        random.seed(RANDOM_SEED)
        patients = random.sample(patients, len(patients) // 4)

        # --- Dataset Init ---
        train_patients, val_patients = train_test_split(
            patients, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED
        )
        train_dataset = NiftDataset(
            patients=train_patients,
            filenames=INPUT_FILENAMES,
            slice_axis=SLICE_AXIS,
        )
        val_dataset = NiftDataset(
            patients=val_patients,
            filenames=INPUT_FILENAMES,
            slice_axis=SLICE_AXIS,
        )
        if len(train_dataset) > 0 and len(val_dataset) > 0:
            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=True if DEVICE.type == "cuda" else False,
            )
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=True if DEVICE.type == "cuda" else False,
            )

            # --- Model Init ---
            model = UNet(
                n_channels=len(INPUT_FILENAMES), n_classes=NUM_OUTPUT_CHANNELS_MODEL
            ).to(DEVICE)
            print(f"U-Net instantiated.")

            # --- Loss Function Init ---
            criterion = nn.BCEWithLogitsLoss()
            print(f"Loss function: BCEWithLogitsLoss")

            # --- Optimizer Init ---
            optimizer = optim.AdamW(
                model.parameters(), lr=LEARNING_RATE, weight_decay=0.01
            )
            print(f"Optimizer: AdamW with lr={LEARNING_RATE}")

            # --- Training Loop ---
            train_losses_history = []
            val_losses_history = []
            val_dice_scores_history = []
            best_val_dice = -1.0
            for epoch in range(1, NUM_EPOCHS + 1):
                train_loss = train_epoch(
                    model, train_loader, optimizer, criterion, DEVICE, epoch, NUM_EPOCHS
                )
                print(f"Epoch {epoch}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}")
                val_loss, val_dice = validate_epoch(
                    model, val_loader, criterion, DEVICE
                )
                print(f"Epoch {epoch}/{NUM_EPOCHS} - Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
                # Store metrics
                train_losses_history.append(train_loss)
                val_losses_history.append(val_loss)
                val_dice_scores_history.append(val_dice)

                # --- WandB log ---
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                    "best_val_dice": best_val_dice
                })

                # Save the model if validation Dice improves
                if val_dice > best_val_dice:
                    best_val_dice = val_dice
                    torch.save(model.state_dict(), MODEL_SAVE_PATH)
                    wandb.save(MODEL_SAVE_PATH)
                    print(
                        f"Epoch {epoch}: New best validation Dice: {val_dice:.4f}. Model saved to {MODEL_SAVE_PATH}"
                    )
                else:
                    print(
                        f"Epoch {epoch}: Validation Dice: {val_dice:.4f} (Best: {best_val_dice:.4f})"
                    )
            print("Training completed. Model saved.")

            # --- Save Training History ---
            plot_metrics(
                train_losses_history,
                val_losses_history,
                val_dice_scores_history,
                NUM_EPOCHS,
            )
            print("Training history saved.")

            # --- Finish WandB run ---
            wandb.finish()

        else:
            print("\nDataset is empty. Cannot proceed with testing.")

    except Exception as e:
        print(f"An error occurred: {e}")
        raise
