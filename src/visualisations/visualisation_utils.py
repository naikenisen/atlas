from typing import List
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader


def plot_metrics(
    train_losses_history: List[float],
    val_losses_history: List[float],
    val_dice_scores_history: List[float],
    num_epochs: int,
) -> None:
    if not os.path.exists("output/plots"):
        os.makedirs("output/plots")

    epochs_range = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 5))

    # Plot Training & Validation Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses_history, label="Training Loss", marker="o")
    plt.plot(epochs_range, val_losses_history, label="Validation Loss", marker="o")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (BCEWithLogitsLoss)")
    plt.legend()
    plt.grid(True)

    # Plot Validation Dice Score
    plt.subplot(1, 2, 2)
    plt.plot(
        epochs_range,
        val_dice_scores_history,
        label="Validation Dice Score",
        marker="o",
        color="green",
    )
    plt.title("Validation Dice Score")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Coefficient")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join("output/plots", "training_metrics_plot.png"))
    print("Saved training metrics plot to training_metrics_plot.png")


def visualize_sample_segmentation(
    input_slice_modalities,
    ground_truth_mask,
    predicted_mask_probs,
    modality_names=None,
    threshold=0.5,
    save_path=None,
    sample_idx=0,
):
    """
    Visualizes input modalities, ground truth, and model prediction for a single slice.
    """

    # Convert tensors to numpy arrays on CPU, remove batch if present
    def to_numpy_cpu(tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().squeeze().numpy()
        return np.squeeze(tensor)  # Assuming already numpy

    input_slice_modalities = to_numpy_cpu(input_slice_modalities)
    ground_truth_mask = to_numpy_cpu(ground_truth_mask)
    predicted_mask_probs = to_numpy_cpu(predicted_mask_probs)

    predicted_binary_mask = (predicted_mask_probs > threshold).astype(np.uint8)

    num_modalities = (
        input_slice_modalities.shape[0] if input_slice_modalities.ndim == 3 else 1
    )
    num_plots = num_modalities + 2  # Modalities + GT + Prediction

    plt.figure(figsize=(5 * num_plots, 5))

    # Plot input modalities
    if num_modalities == 1:
        plt.subplot(1, num_plots, 1)
        m_name = modality_names[0] if modality_names else "Input Modality"
        plt.imshow(input_slice_modalities, cmap="gray")
        plt.title(m_name)
        plt.axis("off")
    else:
        for i in range(num_modalities):
            plt.subplot(1, num_plots, i + 1)
            m_name = (
                modality_names[i]
                if modality_names and i < len(modality_names)
                else f"Input Modality {i+1}"
            )
            plt.imshow(
                input_slice_modalities[i],
                cmap="gray" if "CT" in m_name.upper() else "hot",
            )  # Guess cmap
            plt.title(m_name)
            plt.axis("off")

    # Plot Ground Truth
    plt.subplot(1, num_plots, num_modalities + 1)
    plt.imshow(ground_truth_mask, cmap="viridis")  # Or a specific cmap for masks
    plt.title("Ground Truth Mask")
    plt.axis("off")

    # Plot Prediction
    plt.subplot(1, num_plots, num_modalities + 2)
    plt.imshow(predicted_binary_mask, cmap="viridis")  # Match GT cmap
    plt.title(f"Predicted Mask (Threshold={threshold})")
    plt.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_sample_{sample_idx}.png")
        print(f"Sample segmentation saved to {save_path}_sample_{sample_idx}.png")
    plt.show()


def visualise_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_samples: int = 5,
    save_path_prefix: str = "val_pred_",
):
    """
    Visualises input, ground truth, and model prediction for a few samples.
    Assumes model is already loaded with best weights and in eval mode.
    """
    model.eval()
    if not os.path.exists("output/plots"):
        os.makedirs("output/plots")

    samples_shown = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            if samples_shown >= num_samples:
                break

            inputs_gpu, _ = inputs.to(device), targets.to(device)
            outputs_logits = model(inputs_gpu)
            outputs_probs = torch.sigmoid(outputs_logits)
            outputs_binary = (outputs_probs > 0.5).cpu()

            for j in range(inputs.shape[0]):
                if samples_shown >= num_samples:
                    break

                input_slice_np = inputs[j, 0, :, :].cpu().numpy()
                if inputs.shape[1] > 1:
                    input_slice2_np = inputs[j, 1, :, :].cpu().numpy()
                else:
                    input_slice2_np = None

                target_mask_np = targets[j, 0, :, :].cpu().numpy()
                pred_mask_np = outputs_binary[j, 0, :, :].numpy()

                fig, axes = plt.subplots(
                    1,
                    3 if input_slice2_np is None else 4,
                    figsize=(15 if input_slice2_np is None else 20, 5),
                )

                axes[0].imshow(input_slice_np, cmap="gray")
                axes[0].set_title(
                    f"Input Modality 1 (Slice {i*data_loader.batch_size+j})"
                )
                axes[0].axis("off")

                col_idx = 1
                if input_slice2_np is not None:
                    axes[col_idx].imshow(
                        input_slice2_np, cmap="gray"
                    )  # or 'hot' for PET
                    axes[col_idx].set_title(f"Input Modality 2")
                    axes[col_idx].axis("off")
                    col_idx += 1

                axes[col_idx].imshow(
                    target_mask_np, cmap="jet", vmin=0, vmax=1
                )  # Or 'viridis'
                axes[col_idx].set_title("Ground Truth Mask")
                axes[col_idx].axis("off")
                col_idx += 1

                axes[col_idx].imshow(
                    pred_mask_np, cmap="jet", vmin=0, vmax=1
                )  # Or 'viridis'
                axes[col_idx].imshow(
                    input_slice_np, cmap="gray", alpha=0.7
                )  # background
                axes[col_idx].contour(
                    pred_mask_np, colors="lime", linewidths=1.5, levels=[0.5]
                )
                axes[col_idx].contour(
                    target_mask_np,
                    colors="red",
                    linewidths=1.5,
                    levels=[0.5],
                    linestyles="dashed",
                )
                axes[col_idx].set_title("Prediction (lime) & GT (red, dashed)")
                axes[col_idx].axis("off")

                output_dir = os.path.join(
                    "output/plots", f"{save_path_prefix}{samples_shown+1}.png"
                )

                plt.tight_layout()
                plt.savefig(output_dir)
                plt.close(fig)

                samples_shown += 1

            if samples_shown >= num_samples:
                break
    print(
        f"Saved {samples_shown} prediction visualizations to 'visualizations/' folder."
    )
