!pip install torch torchvision
!pip install albumentations
!pip install tqdm

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models.resnet import ResNet50_Weights
from torchvision import transforms
from PIL import Image
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from tqdm import tqdm
import pandas as pd
import time
import datetime
from albumentations import Compose, RandomResizedCrop, HorizontalFlip, Rotate, ColorJitter
from albumentations import Normalize
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')

# Device setup and mixed precision
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
scaler = torch.cuda.amp.GradScaler()

# Configuration
config = {
    'patience': 3,
    'max_epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.00015,
    'num_workers': 4,
    'model_save_path': 'best_model.pth'
}

def get_train_transforms():
    return Compose([
        RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0), p=1.0),
        HorizontalFlip(p=0.5),
        Rotate(limit=30),
        ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
        ToTensorV2(),
    ], additional_targets={'image': 'image'})

def get_test_transforms():
    return Compose([
        RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0), p=1.0),
        ToTensorV2(),
    ], additional_targets={'image': 'image'})



class FoodDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []
        self.classes = []
        self.class_to_idx = {}

        self.classes = sorted(os.listdir(root))
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}

        for class_name in self.classes:
            class_folder = os.path.join(root, class_name)
            for filename in os.listdir(class_folder):
                if filename.endswith((".jpg", ".png")):
                    self.samples.append((os.path.join(class_folder, filename), self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
      image_path, label = self.samples[idx]
      image = Image.open(image_path).convert("RGB")

      if self.transform:
          image = np.array(image)
          augmented = self.transform(image=image)
          image = augmented["image"]

      # Convert to float and normalize if not already done by transforms
      if isinstance(image, torch.Tensor):
          image = image.float()  # Convert to float if it's a tensor
      elif isinstance(image, np.ndarray):
          image = image.astype(np.float32) / 255.0  # Convert to float and normalize

      return image, label

class ResNet50CLIPStyle(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
      if x.dtype != torch.float32:
          x = x.float()  # Ensure input is float32
      features = self.feature_extractor(x)
      features = features.view(features.size(0), -1)
      return self.classifier(features)

def main():

    # Load datasets

    from google.colab import drive
    drive.mount('/content/drive')

    train_dataset = FoodDataset(root="/content/drive/MyDrive/dataset", transform=get_train_transforms())
    test_dataset = FoodDataset(root="/content/drive/MyDrive/dataset", transform=get_test_transforms())
    num_classes = len(train_dataset.classes)

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    # Model setup
    model = ResNet50CLIPStyle(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    class EarlyStopper:
        def __init__(self, patience=3):
            self.patience = patience
            self.counter = 0
            self.best_accuracy = 0.0
            self.best_model_weights = None

        def check(self, current_accuracy, model):
            if current_accuracy > self.best_accuracy:
                self.best_accuracy = current_accuracy
                self.counter = 0
                self.best_model_weights = model.state_dict().copy()
                return False
            else:
                self.counter += 1
                return self.counter >= self.patience

    early_stopper = EarlyStopper(patience=config['patience'])

    class TimeTracker:
        def __init__(self):
            self.epoch_times = []
            self.start_time = None

        def start_epoch(self):
            self.start_time = time.time()

        def end_epoch(self):
            epoch_time = time.time() - self.start_time
            self.epoch_times.append(epoch_time)
            return epoch_time

        def get_avg_time(self):
            return np.mean(self.epoch_times)

        def get_total_time(self):
            return np.sum(self.epoch_times)

        @staticmethod
        def format_time(seconds):
            return str(datetime.timedelta(seconds=int(seconds)))

    time_tracker = TimeTracker()
    best_accuracy = 0.0
    metrics_history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'f1': [],
        'precision': [],
        'recall': [],
        'auroc': []
    }

    for epoch in range(config['max_epochs']):
        time_tracker.start_epoch()

        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        all_probs = []

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['max_epochs']}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1).cpu().detach().numpy())

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        metrics_history['train_loss'].append(train_loss)
        metrics_history['train_accuracy'].append(train_accuracy)

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)

        val_accuracy = val_correct / val_total
        metrics_history['val_accuracy'].append(val_accuracy)

        # Calculate metrics
        f1 = f1_score(all_labels, all_preds, average='macro')
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        auroc = roc_auc_score(np.eye(num_classes)[all_labels], all_probs, multi_class='ovr')

        metrics_history['f1'].append(f1)
        metrics_history['precision'].append(precision)
        metrics_history['recall'].append(recall)
        metrics_history['auroc'].append(auroc)

        scheduler.step()
        epoch_time = time_tracker.end_epoch()
        avg_time = time_tracker.get_avg_time()
        total_time = time_tracker.get_total_time()

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Time: {TimeTracker.format_time(epoch_time)}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f}")
        print(f"  Val Acc: {val_accuracy:.4f} | F1: {f1:.4f}")
        print(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | AUROC: {auroc:.4f}")
        print(f"  Avg Time/Epoch: {TimeTracker.format_time(avg_time)}")
        print(f"  Total Time: {TimeTracker.format_time(total_time)}")

        if early_stopper.check(val_accuracy, model):
            print(f"\nEarly stopping triggered after {epoch+1} epochs!")
            print(f"Best validation accuracy: {early_stopper.best_accuracy:.4f}")
            break

        remaining_patience = config['patience'] - early_stopper.counter
        if remaining_patience < config['patience']:
            print(f"  Early stopping countdown: {remaining_patience}/{config['patience']}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), config['model_save_path'])
            print(f"  New best model saved with accuracy: {best_accuracy:.4f}")

    # Load best model weights
    model.load_state_dict(torch.load(config['model_save_path']))

    # Final evaluation
    model.eval()
    test_correct = 0
    test_total = 0
    test_all_labels = []
    test_all_preds = []
    test_all_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Final Evaluation"):
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                test_correct += (predicted == labels).sum().item()
                test_total += labels.size(0)
                test_all_labels.extend(labels.cpu().numpy())
                test_all_preds.extend(predicted.cpu().numpy())
                test_all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    test_accuracy = test_correct / test_total
    test_f1 = f1_score(test_all_labels, test_all_preds, average='macro')
    test_precision = precision_score(test_all_labels, test_all_preds, average='macro')
    test_recall = recall_score(test_all_labels, test_all_preds, average='macro')
    test_auroc = roc_auc_score(np.eye(num_classes)[test_all_labels], test_all_probs, multi_class='ovr')

    print("\nFinal Test Results:")
    print(f"  Accuracy: {test_accuracy:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  AUROC: {test_auroc:.4f}")
    print(f"\nTotal Training Time: {TimeTracker.format_time(total_time)}")

    # Plot training history
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.plot(metrics_history['train_loss'], label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(2, 3, 2)
    plt.plot(metrics_history['train_accuracy'], label='Train Accuracy')
    plt.plot(metrics_history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(metrics_history['f1'], label='F1 Score')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1')

    plt.subplot(2, 3, 4)
    plt.plot(metrics_history['precision'], label='Precision')
    plt.title('Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')

    plt.subplot(2, 3, 5)
    plt.plot(metrics_history['recall'], label='Recall')
    plt.title('Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')

    plt.subplot(2, 3, 6)
    plt.plot(metrics_history['auroc'], label='AUROC')
    plt.title('AUROC')
    plt.xlabel('Epoch')
    plt.ylabel('AUROC')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
