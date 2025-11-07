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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def scan_and_move_corrupted(dataset_dir, move_to=None):
    moved = []
    dataset_dir = Path(dataset_dir)
    if move_to is None:
        move_to = dataset_dir.parent / "corrupted"
    move_to.mkdir(parents=True, exist_ok=True)

    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
                path = Path(root) / f
                try:
                    Image.open(path).verify()
                except:
                    dst = move_to / f
                    i = 1
                    while dst.exists():
                        dst = move_to / f"{path.stem}_{i}{path.suffix}"
                        i += 1
                    shutil.move(str(path), str(dst))
                    moved.append(str(dst))
                    print(f"[WARNING] Removed corrupted file: {path}")

    return moved


class DiamondDataset:
    def __init__(self, data_dir, target_type):
        self.data_dir = Path(data_dir)
        self.target_type = target_type

        if target_type == 'cut':
            self.classes = ['Ideal', 'Excellent', 'Very Good', 'Good', 'Fair', 'Poor']
        elif target_type == 'shape':
            self.classes = ['Round', 'Princess', 'Emerald', 'Asscher', 'Oval',
                            'Radiant', 'Cushion', 'Marquise', 'Pear', 'Heart']
        elif target_type == 'color':
            self.classes = ['D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def create_dataset(self, batch_size=32, shuffle=True):
        dataset = ds.ImageFolderDataset(str(self.data_dir / self.target_type), shuffle=shuffle)
        transform = [
            vision.Decode(),
            vision.Resize((224, 224)),
            vision.RandomHorizontalFlip(prob=0.5),
            vision.RandomRotation(15),
            vision.RandomColorAdjust(0.2, 0.2),
            vision.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            vision.HWC2CHW()
        ]
        dataset = dataset.map(transform, "image")
        dataset = dataset.batch(batch_size, drop_remainder=True)
        return dataset


class DiamondClassifierAdvanced(nn.Cell):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, pad_mode='pad', padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, pad_mode='pad', padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, pad_mode='pad', padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3, pad_mode='pad', padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(512 * 14 * 14, 1024)
        self.fc2 = nn.Dense(1024, 512)
        self.fc3 = nn.Dense(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(keep_prob=0.5)

    def construct(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.fc3(x)


class ModelTrainer:
    def __init__(self, model):
        self.model = model
        self.loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        self.optimizer = nn.Adam(model.trainable_params(), learning_rate=0.001)

    def train_epoch(self, dataset):
        self.model.set_train()
        total, correct, samples = 0, 0, 0
        for data, label in dataset.create_tuple_iterator():
            loss, logits = ms.value_and_grad(
                lambda x, y: (self.loss_fn(self.model(x), y), self.model(x)),
                None, self.optimizer.parameters, has_aux=True
            )(data, label)
            self.optimizer(loss[1])
            pred = ops.argmax(loss[1], 1)
            correct += (pred == label).sum().asnumpy()
            samples += len(label)
        return correct / samples


def train_model(t, data_dir, epochs):
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    scan_and_move_corrupted(data_dir)
    ds = DiamondDataset(data_dir, t).create_dataset()
    model = DiamondClassifierAdvanced(len(DiamondDataset(data_dir, t).classes))
    trainer = ModelTrainer(model)
    for ep in range(epochs):
        acc = trainer.train_epoch(ds)
        logger.info(f"{t.upper()} Epoch {ep+1} Acc: {acc:.4f}")
    ms.save_checkpoint(model, f"./checkpoints/{t}_model_best.ckpt")


def train_all_models(data_dir, epochs):
    for t in ['cut', 'shape', 'color']:
        train_model(t, data_dir, epochs)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="all")
    p.add_argument("--data_dir", default="../datasets/diamond_dataset")
    p.add_argument("--epochs", type=int, default=10)
    a = p.parse_args()
    train_all_models(a.data_dir, a.epochs) if a.model=="all" else train_model(a.model,a.data_dir,a.epochs)