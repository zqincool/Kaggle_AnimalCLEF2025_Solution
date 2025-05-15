import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import timm
import numpy as np
from tqdm import tqdm
import pandas as pd
from torchvision import transforms
from wildlife_datasets.datasets import AnimalCLEF2025

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
        # Get all sea turtle identities
        # ---------------------
        sea_turtle_metadata = self.dataset.dataset.metadata[self.dataset.dataset.metadata['dataset'] == 'SeaTurtleID2022']
        
        # Valid identities (excluding NaN)
        valid_labels = [label for label in sea_turtle_metadata['identity'].unique() if pd.notna(label)]
        # Add "new_individual" as label for unknown individuals
        valid_labels.append('new_individual')
        
        # Sort alphabetically
        valid_labels = sorted(valid_labels)
        self.label_to_idx = {label: idx for idx, label in enumerate(valid_labels)}
        
        # Print label mapping information
        print(f"\n标签映射信息:")
        print(f"总标签数: {len(self.label_to_idx)}")
        print(f"标签范围: 0 - {len(self.label_to_idx)-1}")
        print("前5个标签映射示例:")
        for label, idx in list(self.label_to_idx.items())[:5]:
            print(f"  {label} -> {idx}")
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]  # returns (PIL_image, identity_str or nan)
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
    def __init__(self, num_classes, confidence_threshold=0.3):
        super(AnimalReIDModel, self).__init__()
        # Use Swin Transformer as backbone
        self.backbone = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=True,
            num_classes=0,      # Don't let Swin output classification
            global_pool='avg'   # Let it do global average pooling inside forward()
        )
        
        self.feature_dim = self.backbone.num_features
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        
        # Confidence threshold: if above threshold, classify as known individual, otherwise as new_individual
        self.confidence_threshold = confidence_threshold
        print(f"\n模型置信度阈值: {self.confidence_threshold}")

    def forward(self, x):
        features = self.backbone(x)   # [B, feature_dim]
        features = self.dropout(features)
        out = self.classifier(features)  # [B, num_classes]
        return out
    
    def predict(self, x):
        """Prediction function with new_individual detection logic"""
        outputs = self.forward(x)             # [B, num_classes]
        probs = torch.softmax(outputs, dim=1) # [B, num_classes]
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

def train_model(model, train_loader, val_loader, num_classes, num_epochs=20):
    """
    Change training epochs to 10 to give model more time to converge
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    
    # Can use CosineAnnealingLR, or keep learning rate fixed (comment out next two lines to fix)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Enable cudnn benchmark
    torch.backends.cudnn.benchmark = True
    
    best_val_loss = float('inf')
    best_model_path = 'confidence0.3.pth'
    
    for epoch in range(num_epochs):
        # -----------------------------
        # Training phase
        # -----------------------------
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_labels = []
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} (Train)'):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)               # [B, num_classes]
            loss = criterion(outputs, labels)     # CrossEntropy
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Use predict() to calculate predicted labels, note this includes new_individual detection
            predicted = model.predict(images)
            train_predictions.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        # -----------------------------
        # Validation phase
        # -----------------------------
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
        
        # -----------------------------
        # Calculate loss and balanced accuracy
        # -----------------------------
        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        
        train_predictions = torch.tensor(train_predictions)
        train_labels      = torch.tensor(train_labels)
        val_predictions   = torch.tensor(val_predictions)
        val_labels        = torch.tensor(val_labels)
        
        train_bal_acc = calculate_balanced_accuracy(train_predictions, train_labels, num_classes)
        val_bal_acc   = calculate_balanced_accuracy(val_predictions, val_labels, num_classes)
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}] Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Balanced Acc: {train_bal_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f}, Balanced Acc: {val_bal_acc:.2f}%")
        
        # -----------------------------
        # Save best model
        # -----------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  [*] Best model saved at epoch {epoch+1} (val_loss={val_loss:.4f}, val_bal_acc={val_bal_acc:.2f}%)")
        
        # Update learning rate
        scheduler.step()
    
    print("\nTraining completed!")

def main():
    # -----------------------------
    # 0. Check CUDA
    # -----------------------------
    if not torch.cuda.is_available():
        print("[错误] CUDA不可用！请检查环境。")
        return
    
    torch.cuda.empty_cache()
    print("CUDA可用，开始训练...")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"可用显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Set random seed
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # -----------------------------
    # 1. Define data augmentation
    #   (Simplified: removed RandomRotation, ColorJitter and other strong augmentations)
    # -----------------------------
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # -----------------------------
    # 2. Load original AnimalCLEF2025 data
    # -----------------------------
    root = '.'
    dataset = AnimalCLEF2025(root, transform=None, load_label=True)
    print("\n[数据集基本信息]")
    print(f"  总样本数: {len(dataset)}")
    print("  metadata columns:", dataset.metadata.columns)
    print("  不同split:", dataset.metadata['split'].unique())
    
    # -----------------------------
    # 3. Select sea turtle data & database subset
    # -----------------------------
    sea_turtle_mask = (dataset.metadata['dataset'] == 'SeaTurtleID2022')
    db_mask = (dataset.metadata['split'] == 'database')
    combined_mask = sea_turtle_mask & db_mask
    print(f"\n[筛选海龟database子集], 总数量: {combined_mask.sum()}")
    
    # Create Subset
    indices = [i for i, valid in enumerate(combined_mask) if valid]
    sea_turtle_subset = torch.utils.data.Subset(dataset, indices)
    
    # -----------------------------
    # 4. Build dataset (with label mapping)
    # -----------------------------
    full_sea_turtle_dataset = AnimalDataset(sea_turtle_subset, transform=train_transform)
    
    # Split into train and validation sets (80:20)
    train_size = int(0.8 * len(full_sea_turtle_dataset))
    val_size   = len(full_sea_turtle_dataset) - train_size
    
    train_dataset, val_dataset = random_split(full_sea_turtle_dataset, [train_size, val_size])
    # Set different transform for validation set
    val_dataset.dataset.transform = val_transform
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"\n[数据集大小]")
    print(f"  训练集: {len(train_dataset)}")
    print(f"  验证集: {len(val_dataset)}")
    
    # -----------------------------
    # 5. Create model & (optional) freeze part of Backbone
    # -----------------------------
    # +1 because we need to include "new_individual" class
    num_classes = len(full_sea_turtle_dataset.label_to_idx)
    model = AnimalReIDModel(num_classes=num_classes, confidence_threshold=0.3)
    
    # (Optional) Freeze first 2 stages, only fine-tune last 2 stages:
    freeze_backbone = False  # If you have very little data, you can try True
    if freeze_backbone:
        freeze_backbone_layers(model, unfreeze_blocks=2)
    
    print(f"\n[模型信息]")
    print(f"  Swin Backbone输出特征维度: {model.feature_dim}")
    print(f"  总类别数: {num_classes}")
    
    # -----------------------------
    # 6. Training
    # -----------------------------
    train_epochs = 10  # Train for more epochs
    train_model(model, train_loader, val_loader, num_classes=num_classes, num_epochs=train_epochs)

if __name__ == '__main__':
    main()
