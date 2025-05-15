import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import timm
from timm.layers import DropPath  # 更新DropPath的导入路径
from torch.amp import GradScaler, autocast  # 更新GradScaler的导入
import numpy as np
from tqdm import tqdm
import pandas as pd
from torchvision import transforms
from wildlife_datasets.datasets import AnimalCLEF2025
from PIL import Image

# freeze swin backbone
def freeze_backbone_layers(model, unfreeze_blocks=2):
    """
    Freeze the first few layers of Swin, keeping only the last unfreeze_blocks trainable.
    For swin_base_patch4_window7_224, there are 4 stages in layers:
        model.backbone.layers[0]
        model.backbone.layers[1]
        model.backbone.layers[2]
        model.backbone.layers[3]
    Usually the later stages contain higher-level semantic information, so we try to fine-tune only the last 2 stages first.
    """
    # freeze all parameters
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    # unfreeze_blocks stages
    total_stages = len(model.backbone.layers)  # generally 4 stages
    for stage_idx in range(total_stages - unfreeze_blocks, total_stages):
        for param in model.backbone.layers[stage_idx].parameters():
            param.requires_grad = True

class AnimalDataset(Dataset):
    def __init__(self, subset_dataset, transform=None):
        """
        subset_dataset: subset filtered from original AnimalCLEF2025
        transform: data augmentation/preprocessing
        """
        self.dataset = subset_dataset
        self.transform = transform
        
        # ---------------------
        # Get all animal identities from the database
        # ---------------------
        all_metadata = self.dataset.dataset.metadata
        
        # Dataset validation
        print("\n[Dataset Validation]")
        print(f"Total samples: {len(all_metadata)}")
        print(f"Dataset columns: {all_metadata.columns.tolist()}")
        
        # Check label distribution
        label_counts = all_metadata['identity'].value_counts()
        print("\nLabel distribution statistics:")
        print(f"Unique labels: {len(label_counts)}")
        print(f"Max samples per label: {label_counts.max()}")
        print(f"Min samples per label: {label_counts.min()}")
        print(f"Average samples per label: {label_counts.mean():.2f}")
        print(f"Median samples per label: {label_counts.median():.2f}")
        
        # Check for null values
        nan_count = all_metadata['identity'].isna().sum()
        print(f"\nNull values: {nan_count}")
        
        # Valid identities (excluding NaN)
        valid_labels = [label for label in all_metadata['identity'].unique() if pd.notna(label)]
        # Add "new_individual" as label for unknown individuals
        valid_labels.append('new_individual')
        
        # Sort alphabetically
        valid_labels = sorted(valid_labels)
        self.label_to_idx = {label: idx for idx, label in enumerate(valid_labels)}
        
        # Validate label mapping
        print("\n[Label Mapping Validation]")
        print(f"Total labels: {len(self.label_to_idx)}")
        print(f"Label range: 0 - {len(self.label_to_idx)-1}")
        
        # Check label distribution
        label_distribution = {}
        for idx in range(len(self.dataset)):
            _, label = self.dataset[idx]
            if pd.isna(label):
                label = 'new_individual'
            if label not in self.label_to_idx:
                label = 'new_individual'
            label_idx = self.label_to_idx[label]
            label_distribution[label_idx] = label_distribution.get(label_idx, 0) + 1
        
        # print("\n标签分布:")
        # for idx, count in sorted(label_distribution.items()):
        #     label = [k for k, v in self.label_to_idx.items() if v == idx][0]
        #     print(f"  {label}: {count} 样本")
        
        # Count by species
        species_counts = {}
        for dataset_name in all_metadata['dataset'].unique():
            if pd.notna(dataset_name):
                species_counts[dataset_name] = len(all_metadata[all_metadata['dataset'] == dataset_name]['identity'].unique())
        
        print("\nNumber of individuals per species:")
        for species, count in species_counts.items():
            print(f"  {species}: {count}")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]  # returns (PIL_image, identity_str or nan)
        
        # Add image validation
        if image is None:
            raise ValueError(f"Empty image at idx {idx}")
        if not isinstance(image, Image.Image):
            raise ValueError(f"Invalid image type at idx {idx}: {type(image)}")
        
        if self.transform:
            image = self.transform(image)
        
        # NaN -> 'new_individual'
        label = 'new_individual' if pd.isna(label) else label
        
        if label not in self.label_to_idx:
            # If label not in label_to_idx, treat as new_individual
            label = 'new_individual'
        label_idx = self.label_to_idx[label]
        
        return image, torch.tensor(label_idx, dtype=torch.long)

class AnimalReIDModel(nn.Module):
    def __init__(self, num_classes, confidence_threshold=0.5):  # 降低置信度阈值
        super(AnimalReIDModel, self).__init__()
        # 使用更小的Swin模型
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        
        self.feature_dim = self.backbone.num_features
        
        # 增强分类头
        self.dropout1 = nn.Dropout(0.3)  # 第一层dropout
        self.bn1 = nn.BatchNorm1d(self.feature_dim)
        self.fc1 = nn.Linear(self.feature_dim, self.feature_dim // 2)
        self.dropout2 = nn.Dropout(0.3)  # 第二层dropout
        self.bn2 = nn.BatchNorm1d(self.feature_dim // 2)
        self.classifier = nn.Linear(self.feature_dim // 2, num_classes)
        
        self.confidence_threshold = confidence_threshold
        print(f"\nModel confidence threshold: {self.confidence_threshold}")

    def forward(self, x):
        features = self.backbone(x)
        features = self.bn1(features)
        features = self.dropout1(features)
        features = torch.relu(self.fc1(features))  # 添加激活函数
        features = self.bn2(features)
        features = self.dropout2(features)
        out = self.classifier(features)
        return out
    
    def predict(self, x):
        """Prediction function with new_individual detection logic"""
        outputs = self.forward(x)
        probs = torch.softmax(outputs, dim=1)
        max_probs, predicted = torch.max(probs, dim=1)
        
        # If below confidence threshold, classify as new_individual
        new_mask = max_probs < self.confidence_threshold
        predicted[new_mask] = -1  # -1 represents new_individual
        
        return predicted

def calculate_balanced_accuracy(predictions, labels, num_classes):
    """
    Calculate balanced accuracy
    """
    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)
    
    for i in range(num_classes):
        class_mask = (labels == i)
        class_total[i] = class_mask.sum().item()
        if class_total[i] > 0:
            class_correct[i] = (predictions[class_mask] == labels[class_mask]).sum().item()
    
    valid_classes = class_total > 0
    if valid_classes.sum() == 0:
        return 0.0
    
    class_acc = class_correct[valid_classes] / class_total[valid_classes]
    balanced_acc = class_acc.mean().item() * 100
    return balanced_acc

def calculate_baks(predictions, labels, num_classes):
    """
    Calculate BAKS (balanced accuracy on known samples)
    predictions: model predictions
    labels: ground truth labels
    num_classes: total number of classes
    """
    # Convert to numpy array
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    if len(labels) == 0:
        return 0.0
    
    # Calculate accuracy for each class
    unique_labels = np.unique(labels)
    accuracies = []
    for label in unique_labels:
        label_mask = (labels == label)
        if label_mask.sum() > 0:
            acc = np.mean(predictions[label_mask] == labels[label_mask])
            accuracies.append(acc)
    
    baks = np.mean(accuracies) * 100 if accuracies else 0.0
    return baks

def train_model(model, train_loader, val_loader, num_classes, num_epochs=20):
    """
    Using a more conservative training strategy
    """
    # Add debugging information
    print("\n[Model Structure Check]")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
    
    # Print label distribution of first batch to check for issues
    print("\n[Check First Batch Label Distribution]")
    first_batch = next(iter(train_loader))
    _, labels = first_batch
    unique_labels, counts = torch.unique(labels, return_counts=True)
    for lbl, cnt in zip(unique_labels.tolist(), counts.tolist()):
        print(f"  Label {lbl}: {cnt} samples")
    
    # Use label smoothing
    criterion = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.1)  # Add label smoothing
    
    # Use a more conservative learning rate
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)  # Further reduce weight decay
    
    # Use a more conservative learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-4,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1000,
        anneal_strategy='cos'
    )
    
    # Add gradient accumulation
    accumulation_steps = 2  # Update parameters every 2 steps
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Enable cudnn benchmark for performance
    torch.backends.cudnn.benchmark = True
    
    best_val_baks = 0.0
    best_model_path = 'test_final_ep20.pth'
    
    # Add gradient check
    def check_gradients(model):
        has_nan = False
        has_inf = False
        grad_norm = 0.0
        grad_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_count += 1
                if torch.isnan(param.grad).any():
                    has_nan = True
                    print(f"  NaN gradient in: {name}")
                if torch.isinf(param.grad).any():
                    has_inf = True
                    print(f"  Inf gradient in: {name}")
                grad_norm += torch.norm(param.grad).item()
        
        if grad_count > 0:
            avg_grad_norm = grad_norm / grad_count
            print(f"   Average gradient norm: {avg_grad_norm:.6f}")
        
        return has_nan or has_inf
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_labels = []
        batch_losses = []
        optimizer.zero_grad()  # Zero gradients at the start of epoch
        
        for i, (images, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} (Train)')):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels) / accumulation_steps  # Scale loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Lower gradient clipping threshold
            
            # Gradient accumulation
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Check and record gradients
            # if i % 100 == 0:
            #     print(f"\n[Batch {i}/{len(train_loader)}]")
            #     print(f"  Loss: {loss.item():.4f}")
            #     has_bad_grad = check_gradients(model)
            #     if has_bad_grad:
            #         print("  警告: 发现NaN或Inf梯度!")
            
            # Record
            batch_losses.append(loss.item())
            train_loss += loss.item()
            predicted = model.predict(images)
            train_predictions.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            # Every 100 batches, print current batch accuracy
            if i % 100 == 0:
                batch_preds = torch.tensor(predicted.cpu().numpy())
                batch_labels = torch.tensor(labels.cpu().numpy())
                batch_acc = (batch_preds == batch_labels).float().mean().item() * 100
                print(f"  Batch accuracy: {batch_acc:.2f}%")
                # Print label distribution for this batch
                unique_labels, counts = torch.unique(batch_labels, return_counts=True)
                print(f"  Batch label distribution: {list(zip(unique_labels.tolist(), counts.tolist()))}")
        
        # Check if training loss is abnormal
        mean_batch_loss = sum(batch_losses) / len(batch_losses)
        print(f"\n[Training Loss Check]")
        print(f"  Average batch loss: {mean_batch_loss:.4f}")
        print(f"  Max batch loss: {max(batch_losses):.4f}")
        print(f"  Min batch loss: {min(batch_losses):.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} (Val)'):
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                predicted = model.predict(images)
                val_predictions.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate loss and accuracy
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_predictions = torch.tensor(train_predictions)
        train_labels = torch.tensor(train_labels)
        val_predictions = torch.tensor(val_predictions)
        val_labels = torch.tensor(val_labels)
        
        # Calculate normal accuracy
        train_acc = (train_predictions == train_labels).float().mean().item() * 100
        val_acc = (val_predictions == val_labels).float().mean().item() * 100
        
        # Calculate balanced accuracy
        train_bal_acc = calculate_balanced_accuracy(train_predictions, train_labels, num_classes)
        val_bal_acc = calculate_balanced_accuracy(val_predictions, val_labels, num_classes)
        
        # Calculate BAKS
        val_baks = calculate_baks(val_predictions, val_labels, num_classes)
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}] Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, Balanced Acc: {train_bal_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, Balanced Acc: {val_bal_acc:.2f}%")
        print(f"  Val   BAKS: {val_baks:.2f}%")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model (based on BAKS)
        if val_baks > best_val_baks:
            best_val_baks = val_baks
            torch.save(model.state_dict(), best_model_path)
            print(f"  [*] Best model saved at epoch {epoch+1} (BAKS={val_baks:.2f}%)")
    
    print("\nTraining completed!")

def main():
    # -----------------------------
    # 0. Check CUDA
    # -----------------------------
    if not torch.cuda.is_available():
        print("[Error] CUDA not available! Please check environment.")
        return
    
    torch.cuda.empty_cache()
    print("CUDA available, starting training...")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Set random seed
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # -----------------------------
    # 1. Define data augmentation (极度简化)
    # -----------------------------
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # 减小颜色增强强度
        transforms.RandomRotation(10),  # 减小旋转角度
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # 减小平移范围
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # -----------------------------
    # 2. Load original AnimalCLEF2025 data
    # -----------------------------
    root = '.'
    dataset = AnimalCLEF2025(root, transform=None, load_label=True)
    print("\n[Dataset Basic Information]")
    print(f"  Total samples: {len(dataset)}")
    print("  metadata columns:", dataset.metadata.columns)
    print("  Different splits:", dataset.metadata['split'].unique())
    print("  Different datasets:", dataset.metadata['dataset'].unique())
    
    # -----------------------------
    # 3. Select all animal data from the database subset
    # -----------------------------
    # Only select data from the database (no filter on species now)
    db_mask = (dataset.metadata['split'] == 'database')
    print(f"\n[Filtering all animal database subset], Total count: {db_mask.sum()}")
    
    # Create Subset
    indices = [i for i, valid in enumerate(db_mask) if valid]
    all_animals_subset = torch.utils.data.Subset(dataset, indices)
    
    # -----------------------------
    # 4. Build dataset (with label mapping)
    # -----------------------------
    full_animal_dataset = AnimalDataset(all_animals_subset, transform=train_transform)
    
    # Use simple random split instead of complex stratified sampling
    # Split into train and validation sets (80:20)
    train_size = int(0.8 * len(full_animal_dataset))
    val_size = len(full_animal_dataset) - train_size
    
    train_dataset, val_dataset = random_split(full_animal_dataset, [train_size, val_size])
    # Set different transform for validation set
    val_dataset.dataset.transform = val_transform
    
    # DataLoader - Increase batch size to 16
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"\n[Dataset Size]")
    print(f"  Training set: {len(train_dataset)}")
    print(f"  Validation set: {len(val_dataset)}")
    
    # -----------------------------
    # 5. Create model & (optional) freeze part of Backbone
    # -----------------------------
    # +1 because we need to include "new_individual" class
    num_classes = len(full_animal_dataset.label_to_idx)
    model = AnimalReIDModel(num_classes=num_classes, confidence_threshold=0.5)  # Lower confidence threshold
    
    # Fully freeze the first few layers of backbone
    freeze_backbone_layers(model, unfreeze_blocks=2)  # Unfreeze last two layers, increase trainable parameters
    
    print(f"\n[Model Information]")
    print(f"  Swin Backbone output feature dimension: {model.feature_dim}")
    print(f"  Total number of classes: {num_classes}")
    
    # -----------------------------
    # 6. Training
    # -----------------------------
    train_epochs = 30
    train_model(model, train_loader, val_loader, num_classes=num_classes, num_epochs=train_epochs)

if __name__ == '__main__':
    main()
