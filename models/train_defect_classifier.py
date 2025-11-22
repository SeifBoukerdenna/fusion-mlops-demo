import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os
from pathlib import Path
import json
from tqdm import tqdm

CONFIG = {
    "model_name": "resnet18",
    "num_classes": 6,  # NEU dataset has 6 defect classes
    "epochs": 20,
    "batch_size": 32,
    "learning_rate": 0.001,
    "image_size": 224,
    "model_version": "resnet18-neu-v1.0"
}

class NEUDefectDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.samples = []

        # Structure: data_dir/train/images/class_name/*.jpg
        images_dir = self.data_dir / split / 'images'
        if images_dir.exists():
            for class_name in self.classes:
                class_dir = images_dir / class_name
                if class_dir.exists():
                    for img_path in class_dir.glob('*.jpg'):
                        self.samples.append((str(img_path), self.class_to_idx[class_name]))
        else:
            # Fallback: data_dir/class_name/*.jpg
            for class_name in self.classes:
                class_dir = self.data_dir / class_name
                if class_dir.exists():
                    for img_path in class_dir.glob('*.jpg'):
                        self.samples.append((str(img_path), self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def create_dataloaders(data_dir):
    """Create train and validation dataloaders"""

    train_transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = NEUDefectDataset(data_dir, split='train', transform=train_transform)
    val_dataset = NEUDefectDataset(data_dir, split='validation', transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=2
    )

    return train_loader, val_loader


def create_model():
    """Create ResNet18 model for defect classification"""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Replace final layer for our 6 classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, CONFIG['num_classes'])

    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'acc': 100. * correct / total
        })

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validating'):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def train_model(data_dir='data/NEU-DET', output_dir='models/model_artifacts'):
    """Main training function"""

    print("=" * 60)
    print("Training Defect Classifier")
    print("=" * 60)
    print(f"Configuration: {json.dumps(CONFIG, indent=2)}")

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"\nUsing device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    print("\nLoading data...")
    train_loader, val_loader = create_dataloaders(data_dir)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    print("\nCreating model...")
    model = create_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    print("\nStarting training...")
    best_val_acc = 0.0
    training_history = []

    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(output_dir, 'resnet18_neu.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': CONFIG
            }, model_path)
            print(f"✓ Saved best model (val_acc: {val_acc:.2f}%)")

    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Training complete! Best val accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {model_path}")
    print("=" * 60)

    return model_path


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        print("Test mode: Using dummy data")
        os.makedirs('data/NEU-DET/train/images/scratches', exist_ok=True)
        os.makedirs('data/NEU-DET/validation/images/scratches', exist_ok=True)

        from PIL import Image
        import numpy as np
        dummy_img = Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8))
        for i in range(10):
            dummy_img.save(f'data/NEU-DET/train/images/scratches/test_{i}.jpg')
            dummy_img.save(f'data/NEU-DET/validation/images/scratches/test_{i}.jpg')

        CONFIG['epochs'] = 2
        CONFIG['batch_size'] = 4

    train_model()