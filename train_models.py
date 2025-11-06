import mindspore as ms
from mindspore import nn, ops
import mindspore.dataset as ds
from mindspore.dataset import vision
import numpy as np
from pathlib import Path
import logging
from PIL import Image
import os
import shutil

# ========== Logging Setup ==========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ========== Utility: Scan & Move Corrupted Images ==========
def scan_and_move_corrupted(dataset_dir, move_to=None):
    """
    Scan a directory tree for corrupted images and move them to move_to.
    Returns list of moved files.
    """
    moved = []
    dataset_dir = Path(dataset_dir)
    if move_to is None:
        move_to = dataset_dir.parent / "corrupted"
    move_to = Path(move_to)
    move_to.mkdir(parents=True, exist_ok=True)

    for root, _, files in os.walk(dataset_dir):
        root_path = Path(root)
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
                path = root_path / f
                try:
                    Image.open(path).verify()
                except Exception:
                    dst = move_to / f
                    i = 1
                    while dst.exists():
                        dst = move_to / f"{path.stem}_{i}{path.suffix}"
                        i += 1
                    shutil.move(str(path), str(dst))
                    moved.append(str(dst))
                    print(f"[WARNING] Moved corrupted image: {path} -> {dst}")

    if moved:
        print(f"⚠️  Found and moved {len(moved)} corrupted images to: {move_to}")
    else:
        print("✅ No corrupted images found.")
    return moved


# ========== Dataset Handler ==========
class DiamondDataset:
    """Dataset handler for diamond images"""

    def __init__(self, data_dir, target_type='cut'):
        self.data_dir = Path(data_dir)
        self.target_type = target_type

        if target_type == 'cut':
            self.classes = ['Ideal', 'Excellent', 'Very Good', 'Good', 'Fair', 'Poor']
        elif target_type == 'shape':
            self.classes = ['Round', 'Princess', 'Emerald', 'Asscher', 'Oval',
                            'Radiant', 'Cushion', 'Marquise', 'Pear', 'Heart']
        elif target_type == 'color':
            self.classes = ['D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']
        else:
            raise ValueError(f"Invalid target_type: {target_type}")

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def create_dataset(self, batch_size=32, shuffle=True):
        """Create MindSpore dataset (assumes corrupted images already removed)"""
        dataset_dir = str(self.data_dir / self.target_type)

        dataset = ds.ImageFolderDataset(
            dataset_dir=dataset_dir,
            num_parallel_workers=4,
            shuffle=shuffle
        )

        transform_ops = [
            vision.Decode(),
            vision.Resize((224, 224)),
            vision.RandomHorizontalFlip(prob=0.5),
            vision.RandomRotation(degrees=15),
            vision.RandomColorAdjust(brightness=0.2, contrast=0.2),
            vision.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
            vision.HWC2CHW()
        ]

        dataset = dataset.map(
            operations=transform_ops,
            input_columns=["image"],
            num_parallel_workers=4
        )

        dataset = dataset.batch(batch_size, drop_remainder=True)
        return dataset


# ========== Model Definition ==========
class DiamondClassifierAdvanced(nn.Cell):
    """Advanced CNN model for diamond classification"""

    def __init__(self, num_classes):
        super(DiamondClassifierAdvanced, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, pad_mode='pad', padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, pad_mode='pad', padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, pad_mode='pad', padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3, pad_mode='pad', padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Dense(512 * 14 * 14, 1024)
        self.fc2 = nn.Dense(1024, 512)
        self.fc3 = nn.Dense(512, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(keep_prob=0.5)

    def construct(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# ========== Trainer ==========
class ModelTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        self.optimizer = nn.Adam(model.trainable_params(), learning_rate=learning_rate)

        def forward_fn(data, label):
            logits = model(data)
            loss = self.loss_fn(logits, label)
            return loss, logits

        self.grad_fn = ms.value_and_grad(forward_fn, None, self.optimizer.parameters, has_aux=True)

        def train_step(data, label):
            (loss, logits), grads = self.grad_fn(data, label)
            self.optimizer(grads)
            return loss, logits

        self.train_step = train_step

    def train_epoch(self, train_dataset):
        self.model.set_train()
        total_loss, total_correct, total_samples = 0, 0, 0

        for batch_idx, (data, label) in enumerate(train_dataset.create_tuple_iterator()):
            loss, logits = self.train_step(data, label)
            pred = ops.argmax(logits, dim=1)
            correct = ops.equal(pred, label).sum()

            total_loss += loss.asnumpy()
            total_correct += correct.asnumpy()
            total_samples += len(label)

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Batch {batch_idx + 1}, Loss: {loss.asnumpy():.4f}")

                # Save partial checkpoint every 100 batches
                if (batch_idx + 1) % 100 == 0:
                    partial_path = Path("./checkpoints") / f"{self.model.__class__.__name__.lower()}_partial.ckpt"
                    ms.save_checkpoint(self.model, str(partial_path))
                    logger.info(f"💾 Saved partial checkpoint: {partial_path.name}")

        avg_loss = total_loss / (batch_idx + 1)
        accuracy = total_correct / total_samples
        return avg_loss, accuracy


# ========== Training Function ==========
def train_model(model_type='cut', data_dir='./data', epochs=10, batch_size=32):
    """
    Train a diamond grading model with auto-save and crash recovery.
    Handles resuming, skipping mismatched checkpoints, and saving progress.
    """
    # Automatically clean corrupted images before training
    logger.info("🧹 Scanning for corrupted images...")
    scan_and_move_corrupted(data_dir)
    logger.info(f"Starting training for {model_type} model")

    # Set MindSpore context
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

    # Create checkpoint directory
    checkpoint_dir = Path("./checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    # Create dataset
    dataset_handler = DiamondDataset(data_dir, target_type=model_type)
    train_dataset = dataset_handler.create_dataset(batch_size=batch_size, shuffle=True)

    # Create model
    num_classes = len(dataset_handler.classes)
    model = DiamondClassifierAdvanced(num_classes)

    # --- Checkpoints setup ---
    latest_ckpt = checkpoint_dir / f"{model_type}_model_latest.ckpt"
    partial_ckpt = checkpoint_dir / "diamondclassifieradvanced_partial.ckpt"
    state_file = checkpoint_dir / f"{model_type}_training_state.txt"

    start_epoch = 0
    best_accuracy = 0.0

    # --- Resume logic ---
    if latest_ckpt.exists():
        logger.info(f"🟢 Found latest checkpoint: {latest_ckpt}")
        ms.load_checkpoint(str(latest_ckpt), model)
        logger.info("Resumed from latest checkpoint.")
        if state_file.exists():
            with open(state_file, 'r') as f:
                lines = f.readlines()
                try:
                    start_epoch = int(lines[0].split(':')[1].strip())
                    best_accuracy = float(lines[1].split(':')[1].strip())
                    logger.info(f"Resumed from epoch {start_epoch}, best acc: {best_accuracy:.4f}")
                except Exception:
                    logger.warning("⚠️ Could not read epoch/best accuracy from state file.")
    elif partial_ckpt.exists() and model_type == "cut":
        # Only resume partial checkpoint for cut model
        logger.info(f"⚠️ Found partial checkpoint (cut only): {partial_ckpt}")
        ms.load_checkpoint(str(partial_ckpt), model)
        logger.info("Resumed from partial checkpoint.")
    else:
        logger.info("No matching checkpoint found — training from scratch.")

    # Create trainer
    trainer = ModelTrainer(model, learning_rate=0.001)

    # --- Training loop ---
    for epoch in range(start_epoch, epochs):
        logger.info(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss, train_acc = trainer.train_epoch(train_dataset)
        logger.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        # Save latest checkpoint (for crash recovery)
        latest_path = checkpoint_dir / f"{model_type}_model_latest.ckpt"
        ms.save_checkpoint(model, str(latest_path))
        logger.info(f"✓ Saved checkpoint: {latest_path.name}")

        # Save training state
        with open(state_file, 'w') as f:
            f.write(f"epoch: {epoch + 1}\n")
            f.write(f"best_accuracy: {best_accuracy}\n")
            f.write(f"train_loss: {train_loss}\n")
            f.write(f"train_accuracy: {train_acc}\n")

        # Save best model
        if train_acc > best_accuracy:
            best_accuracy = train_acc
            best_path = checkpoint_dir / f"{model_type}_model_best.ckpt"
            ms.save_checkpoint(model, str(best_path))
            logger.info(f"🌟 NEW BEST! Saved: {best_path.name} (accuracy: {best_accuracy:.4f})")

        # Save periodic backups every 5 epochs
        if (epoch + 1) % 5 == 0:
            backup_path = checkpoint_dir / f"{model_type}_model_epoch{epoch + 1}.ckpt"
            ms.save_checkpoint(model, str(backup_path))
            logger.info(f"🗄️ Backup saved: {backup_path.name}")

    logger.info(f"\n✅ Training complete for {model_type} model!")
    logger.info(f"Best accuracy: {best_accuracy:.4f}")
    logger.info(f"Checkpoints saved in: {checkpoint_dir}")

    return model



# ========== Train All Models ==========
def train_all_models(data_dir='./data', epochs=5):
    checkpoint_dir = Path("./checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    logger.info("\n" + "=" * 60)
    logger.info("AUTOGEMGRADE - MODEL TRAINING")
    logger.info(f"Training all 3 models with {epochs} epochs each")
    logger.info(f"Checkpoints will be saved to: {checkpoint_dir.absolute()}")
    logger.info("=" * 60)

    logger.info("\n" + "=" * 50)
    logger.info("Training Color Classification Model (3/3)")
    logger.info("=" * 50)
    color_model = train_model('color', data_dir, epochs)

    logger.info("\n" + "=" * 60)
    logger.info("✅ ALL MODELS TRAINED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info(f"\nCheckpoints saved in: {checkpoint_dir.absolute()}")
    logger.info("\nFiles created:")
    logger.info("  • cut_model_best.ckpt")
    logger.info("  • shape_model_best.ckpt")
    logger.info("  • color_model_best.ckpt")
    logger.info("\nNext step: Upload .ckpt files to Huawei Cloud")
    logger.info("=" * 60)


# ========== Entry Point ==========
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train diamond classification models')
    parser.add_argument('--data_dir', type=str, default='../datasets/diamond_dataset',
                        help='Path to dataset directory')
    parser.add_argument('--model', type=str, choices=['cut', 'shape', 'color', 'all'],
                        default='all', help='Which model to train')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    args = parser.parse_args()

    if args.model == 'all':
        train_all_models(data_dir=args.data_dir, epochs=args.epochs)
    else:
        train_model(args.model, args.data_dir, args.epochs)
