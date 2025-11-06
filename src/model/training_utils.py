import torch
import wandb


def dice_coefficient(
    preds: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-6
) -> torch.Tensor:
    """
    Calculates the Dice coefficient for binary segmentation.
    """
    preds = (preds > 0.5).float()
    targets = targets.float()
    preds_flat = preds.contiguous().view(preds.shape[0], -1)
    targets_flat = targets.contiguous().view(targets.shape[0], -1)
    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)
    dice = (2.0 * intersection + epsilon) / (union + epsilon)
    return dice.mean()


def train_epoch(
    model, loader, optimizer, criterion, device, epoch_num, num_epochs
):
    model.train()
    running_loss = 0.0
    num_batches = len(loader)

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs_logits = model(inputs)
        loss = criterion(outputs_logits, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        wandb.log({
        "batch_loss": loss.item(),
        "epoch_num": epoch_num,
        "num_epochs": num_epochs
        })
        print(
            f"Epoch {epoch_num}/{num_epochs} - Training Batch {batch_idx+1}/{num_batches} - Loss: {loss.item():.4f}"
        )
    avg_epoch_loss = running_loss / num_batches
    print(
        f"Epoch {epoch_num}/{num_epochs} - Training Completed - Avg Loss: {avg_epoch_loss:.4f}{' '*20}"
    )
    return avg_epoch_loss


def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    num_batches = len(loader)

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs_logits = model(inputs)
            loss = criterion(outputs_logits, targets)
            running_loss += loss.item()
            outputs_probs = torch.sigmoid(outputs_logits)
            dice = dice_coefficient(outputs_probs, targets)
            running_dice += dice.item()

    avg_val_loss = running_loss / num_batches
    avg_val_dice = running_dice / num_batches
    print(f"Validation - Avg Loss: {avg_val_loss:.4f}, Avg Dice: {avg_val_dice:.4f}")
    return avg_val_loss, avg_val_dice


def test_epoch(model, loader, device):
    model.eval()
    running_dice = 0.0
    num_batches = len(loader)

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs_logits = model(inputs)
            outputs_probs = torch.sigmoid(outputs_logits)
            dice = dice_coefficient(outputs_probs, targets)
            running_dice += dice.item()

    avg_test_dice = running_dice / num_batches
    print(f"Test - Avg Dice: {avg_test_dice:.4f}")
    return avg_test_dice


