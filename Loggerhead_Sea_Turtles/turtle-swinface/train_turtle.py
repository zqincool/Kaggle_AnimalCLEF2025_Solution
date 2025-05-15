import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import timm
import numpy as np
from tqdm import tqdm
import math   
import pandas as pd
from torchvision import transforms
from wildlife_datasets.datasets import AnimalCLEF2025
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from timm.loss import LabelSmoothingCrossEntropy
import torch.nn.functional as F

import sys
sys.path.append('/content/drive/MyDrive/swin_face_turtle/swinface')  # 修改为你的 SwinFace 路径

from model import build_model
import torch.nn as nn

# freeze swin backbone
# def freeze_backbone_layers(model, unfreeze_blocks=4):
#     """
#     Freeze the first few layers of Swin, keeping only the last unfreeze_blocks trainable.
#     For swin_base_patch4_window7_224, there are 4 stages in layers:
#         model.backbone.layers[0]
#         model.backbone.layers[1]
#         model.backbone.layers[2]
#         model.backbone.layers[3]
#     Usually the later stages contain higher-level semantic information, so we try to fine-tune only the last 2 stages first.
#     """
#     # freeze all parameters
#     for param in model.backbone.parameters():
#         param.requires_grad = False
    
#     # unfreeze_blocks stages
#     total_stages = len(model.backbone.layers)  # generally 4 stages
#     for stage_idx in range(total_stages - unfreeze_blocks, total_stages):
#         for param in model.backbone.layers[stage_idx].parameters():
#             param.requires_grad = True

def freeze_backbone_layers(model, unfreeze_blocks=2):
    """
    冻结 Swin Transformer 的前几个 stage，仅训练最后 unfreeze_blocks 个 stage。
    同时打印哪些 stage 被冻结 / 解冻。
    """
    try:
        stages = model.backbone.backbone.layers  # 注意：双层 backbone
    except AttributeError:
        print("[错误] 未找到 model.backbone.backbone.layers，请检查模型结构。")
        return

    total_stages = len(stages)
    print(f"[冻结设置] Swin 一共 {total_stages} 个 stage，解冻最后 {unfreeze_blocks} 个：")

    for i, stage in enumerate(stages):
        if i >= total_stages - unfreeze_blocks:
            for param in stage.parameters():
                param.requires_grad = True
            print(f"  ✅ Stage {i}: 解冻")
        else:
            for param in stage.parameters():
                param.requires_grad = False
            print(f"  ❄️  Stage {i}: 冻结")

    # 统计 summary
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[参数统计] 可训练参数数量: {trainable_params:,} / {total_params:,} ({trainable_params / total_params * 100:.2f}%)")




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
# def calculate_balanced_accuracy(predictions, labels, num_classes):
#     mask = labels != num_classes  # 排除 unknown 类
#     predictions = predictions[mask]
#     labels = labels[mask]

#     class_correct = torch.zeros(num_classes)
#     class_total = torch.zeros(num_classes)

#     for i in range(num_classes):
#         class_mask = (labels == i)
#         class_total[i] = class_mask.sum().item()
#         if class_total[i] > 0:
#             class_correct[i] = (predictions[class_mask] == labels[class_mask]).sum().item()

#     valid_classes = class_total > 0
#     if valid_classes.sum() == 0:
#         return 0.0

#     class_acc = class_correct[valid_classes] / class_total[valid_classes]
#     balanced_acc = class_acc.mean().item() * 100
#     return balanced_acc

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confidence_histogram(val_loader, model, device):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            max_probs = torch.max(probs, dim=1).values
            all_probs.extend(max_probs.cpu().numpy())
    plt.hist(all_probs, bins=30)
    plt.title("Validation Max Softmax Confidence")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig("confidence_hist.png")
    plt.close()
def plot_confusion_matrix(predictions, labels, num_classes):
    cm = confusion_matrix(labels, predictions, labels=list(range(num_classes)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix (Validation Set)")
    plt.savefig("confusion_matrix.png")
    plt.close()

def train_model(model, train_loader, val_loader, num_classes, num_epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion_ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    torch.backends.cudnn.benchmark = True

    best_val_loss = float('inf')
    best_model_path = 'confidence0.3.pth'

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            features = model.backbone(images)   # 或者 model.extract_features(images)
            # ArcFace
            arc_logits = model.arcface(features, labels)
            # 分类头
            cls_logits = model.classifier(features)
            loss_arc = model.arcface(arc_logits, labels)
            loss_ce  = model.ce_smooth(cls_logits, labels)
            loss     = loss_arc + loss_ce
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_predictions, val_labels = [], []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} (Val)'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = model.predict(images)
                val_predictions.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        # --- Metrics ---
        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)

        train_preds = torch.tensor(train_predictions)
        train_lbls  = torch.tensor(train_labels)
        val_preds   = torch.tensor(val_predictions)
        val_lbls    = torch.tensor(val_labels)

        # Balanced Acc (unknown counted as wrong)
        train_bal_acc = calculate_balanced_accuracy(train_preds, train_lbls, num_classes)
        val_bal_acc   = calculate_balanced_accuracy(val_preds, val_lbls, num_classes)
        # Plain Acc (unknown counted as wrong)
        train_acc = (train_preds == train_lbls).float().mean().item() * 100
        val_acc   = (val_preds   == val_lbls ).float().mean().item() * 100

        unknown_ratio = (val_preds == -1).sum().item() / len(val_preds)
        print(f"  Unknown prediction rate: {unknown_ratio * 100:.2f}%")

        print(f"\nEpoch [{epoch+1}/{num_epochs}] Summary:")
        print(f"  Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.2f}%, "
              f"Train Bal Acc: {train_bal_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f}, "
              f"Val   Acc: {val_acc:.2f}%, "
              f"Val   Bal Acc: {val_bal_acc:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  [*] Best model saved at epoch {epoch+1} "
                  f"(val_loss={val_loss:.4f}, val_bal_acc={val_bal_acc:.2f}%)")

        #scheduler.step()

    print("\nTraining completed!")




class SwinFaceCfg:
    network = "swin_t"
    fam_kernel_size = 3
    fam_in_chans = 2112
    fam_conv_shared = False
    fam_conv_mode = "split"
    fam_channel_attention = "CBAM"
    fam_spatial_attention = None
    fam_pooling = "max"
    fam_la_num_list = [2 for _ in range(11)]
    fam_feature = "all"
    fam = "3x3_2112_F_s_C_N_max"
    embedding_size = 1024

class SwinFaceModelWrapper(nn.Module):
    def __init__(self, num_classes, confidence_threshold=0.2, pretrained_path= None):
        super().__init__()
        self.cfg = SwinFaceCfg()
        self.backbone = build_model(self.cfg)
        self.confidence_threshold = confidence_threshold
        self.classifier = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.6)
        # —— 使用自定义 ArcMarginProduct
        self.arcface = ArcFaceLoss(
            embedding_size=1024,
            num_classes=num_classes,
            scale=64.0,
            margin=0.5,
            easy_margin=False
        )
        self.ce_smooth = LabelSmoothingCrossEntropy(smoothing=0.1)
        # === 加载预训练权重 ===
        if pretrained_path is not None:
            print(f"[预训练] 加载 backbone 权重: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location='cpu')
            self.backbone.load_state_dict(state_dict, strict=False)  # strict=False 允许 partial match

        # === 获取模型期望输入尺寸 ===
        H, W = 112, 112
        dummy_input = torch.randn(1, 3, H, W)
        self.backbone.eval()  
        # === 自动推断 feature_dim ===
        with torch.no_grad():
            output_dict = self.backbone(dummy_input)
            features = output_dict['cls_feat']
            if isinstance(features, tuple):
                features = features[0]
            self.feature_dim = features.reshape(1, -1).size(1)

        # —— ArcFace 头
        self.arcface = ArcFaceLoss(embedding_size=1024,
                                   num_classes=num_classes,
                                   margin=0.5, scale=64)  # 示例 margin/scale
        # 最后分类头（如果还想保留普通全连接）

        self.fc1 = nn.Linear(self.feature_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(1024, num_classes)
        # self.classifier = nn.Linear(self.feature_dim, num_classes)
        print(f"\n[SwinFace] 模型初始化完成，输入尺寸: {H}x{W}，feature_dim: {self.feature_dim}，置信度阈值: {self.confidence_threshold}")

    # def forward(self, x):
    #     output_dict = self.backbone(x)
    #     features = output_dict['cls_feat']
    #     if isinstance(features, tuple):
    #         features = features[0]
    #     features = features.reshape(features.size(0), -1)
    #     x = self.fc1(features)
    #     x = self.bn1(x)
    #     x = self.dropout(x)
    #     logits = self.classifier(x)
    #     return logits


    # def predict(self, x):
    #     logits = self.forward(x)
    #     probs = torch.softmax(logits, dim=1)
    #     max_probs, preds = torch.max(probs, dim=1)
    #     preds[max_probs < self.confidence_threshold] = -1
    #     return preds
    def forward(self, x, labels=None):
        x_dict = self.backbone(x)
        feat = x_dict['cls_feat']
        if isinstance(feat, tuple): feat = feat[0]
        feat = feat.reshape(feat.size(0), -1)
        feat = self.dropout(feat)

        # 如果给了 labels，就同时输出 ArcFace logits
        if labels is not None:
            arc_logits = self.arcface(feat, labels)
        else:
            arc_logits = None

        cls_logits = self.classifier(feat)
        return feat, arc_logits, cls_logits
    # def forward(self, x):
    #     features = self.backbone(x)['cls_feat']
    #     # … reshape/dropout/fc1/BN/dropout …
    #     embeddings = features_flat  # e.g. after fc1+bn
    #     cls_logits = self.classifier(embeddings)
    #     return embeddings, cls_logits

    def predict(self, x):
        _, _, logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        maxp, pred = probs.max(1)
        pred[maxp < self.confidence_threshold] = -1
        return pred

    # def predict(self, x):
    #     logits = self.forward(x)
    #     probs = torch.softmax(logits, dim=1)
    #     max_probs, preds = torch.max(probs, dim=1)
    #     if self.confidence_threshold is not None:
    #         preds = preds.clone()  # 防止 in-place 错误
    #         preds[max_probs < self.confidence_threshold] = -1
    #     return preds

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


class ArcMarginProduct(nn.Module):
    """
    Implementation of ArcFace margin:
      cos(θ + m)
    Inputs:
      in_features: embedding size
      out_features: number of classes
      s: scale factor
      m: angular margin
    """
    def __init__(self, in_features, out_features, s=64.0, m=0.5, easy_margin=False):
        super().__init__()
        self.in_feats  = in_features
        self.out_feats = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.thresh = math.cos(math.pi - m)
        self.mm     = math.sin(math.pi - m) * m
        self.easy_margin = easy_margin

    def forward(self, embeddings, labels):
        # L2 normalize
        normalized_emb = F.normalize(embeddings, p=2, dim=1)
        normalized_w   = F.normalize(self.weight,    p=2, dim=1)
        # cosine = batch_size x num_classes
        cosine = F.linear(normalized_emb, normalized_w)
        sine   = torch.sqrt(1.0 - torch.clamp(cosine**2, 0, 1))
        # cos(theta + m) = cosθ * cos m − sinθ * sin m
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            # avoid phi < 0
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # ensure numeric stability
            phi = torch.where(cosine > self.thresh, phi, cosine - self.mm)

        # one-hot encode labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1,1), 1.0)

        # combine: only replace target logits with phi
        output = one_hot * phi + (1.0 - one_hot) * cosine
        return output * self.s

class ArcFaceLoss(nn.Module):
    def __init__(self, embedding_size, num_classes, scale=64.0, margin=0.5, easy_margin=False):
        super().__init__()
        self.margin = ArcMarginProduct(embedding_size, num_classes, s=scale, m=margin, easy_margin=easy_margin)
        self.ce     = nn.CrossEntropyLoss()

    def forward(self, embeddings, labels):
        logits = self.margin(embeddings, labels)
        loss   = self.ce(logits, labels)
        return loss, logits
# def main():
#     # -----------------------------
#     # 0. Check CUDA
#     # -----------------------------
#     if not torch.cuda.is_available():
#         print("[错误] CUDA不可用！请检查环境。")
#         return
    
#     torch.cuda.empty_cache()
#     print("CUDA可用，开始训练...")
#     print(f"GPU: {torch.cuda.get_device_name(0)}")
#     print(f"可用显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
#     # Set random seed
#     # seed = 42
#     # torch.manual_seed(seed)
#     # torch.cuda.manual_seed_all(seed)
#     # np.random.seed(seed)
    
#     # -----------------------------
#     # 1. Define data augmentation
#     #   (Simplified: removed RandomRotation, ColorJitter and other strong augmentations)
#     # -----------------------------
#     train_transform = transforms.Compose([
#         transforms.Resize((112, 112)) ,
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])
#     val_transform = transforms.Compose([
#         transforms.Resize((112, 112)),
#         # transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])
    
#     # -----------------------------
#     # 2. Load original AnimalCLEF2025 data
#     # -----------------------------
#     root = '.'
#     dataset = AnimalCLEF2025(root, transform=None, load_label=True)
#     print("\n[数据集基本信息]")
#     print(f"  总样本数: {len(dataset)}")
#     print("  metadata columns:", dataset.metadata.columns)
#     print("  不同split:", dataset.metadata['split'].unique())
    
#     # -----------------------------
#     # 3. Select sea turtle data & database subset
#     # -----------------------------
#     sea_turtle_mask = (dataset.metadata['dataset'] == 'SeaTurtleID2022')
#     db_mask = (dataset.metadata['split'] == 'database')
#     combined_mask = sea_turtle_mask & db_mask
#     print(f"\n[筛选海龟database子集], 总数量: {combined_mask.sum()}")
    
#     # Create Subset
#     indices = [i for i, valid in enumerate(combined_mask) if valid]
#     sea_turtle_subset = torch.utils.data.Subset(dataset, indices)
    
#     # -----------------------------
#     # 4. Build dataset (with label mapping)
#     # -----------------------------
#     full_sea_turtle_dataset = AnimalDataset(sea_turtle_subset, transform=train_transform)
    
#     # Split into train and validation sets (80:20)
#     train_size = int(0.8 * len(full_sea_turtle_dataset))
#     val_size   = len(full_sea_turtle_dataset) - train_size
    
#     train_dataset, val_dataset = random_split(full_sea_turtle_dataset, [train_size, val_size])
#     # Set different transform for validation set
#     val_dataset.dataset.transform = val_transform
    
#     # DataLoader
#     train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,  num_workers=4, pin_memory=True)
#     val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    
#     print(f"\n[数据集大小]")
#     print(f"  训练集: {len(train_dataset)}")
#     print(f"  验证集: {len(val_dataset)}")
    
#     # -----------------------------
#     # 5. Create model & (optional) freeze part of Backbone
#     # -----------------------------
#     # +1 because we need to include "new_individual" class
#     num_classes = len(full_sea_turtle_dataset.label_to_idx)
#     print("总类别数:", len(full_sea_turtle_dataset.label_to_idx))
#     print("所有类别标签:", list(full_sea_turtle_dataset.label_to_idx.keys())[:10], "...") 
#     pretrained_path = "/content/drive/MyDrive/swin_face_turtle/checkpoint_step_79999_gpu_0.pt" 
#     model = SwinFaceModelWrapper(num_classes=num_classes, confidence_threshold=0.3, pretrained_path=pretrained_path)
    
#     # (Optional) Freeze first 2 stages, only fine-tune last 2 stages:
#     freeze_backbone = True  # If you have very little data, you can try True
#     if freeze_backbone:
#         freeze_backbone_layers(model, unfreeze_blocks=4)
    
#     print(f"\n[模型信息]")
#     print(f"  Swin Backbone输出特征维度: {model.feature_dim}")
#     print(f"  总类别数: {num_classes}")
    
#     # -----------------------------
#     # 6. Training
#     # -----------------------------
#     train_epochs = 25  # Train for more epochs
#     train_model(model, train_loader, val_loader, num_classes=num_classes, num_epochs=train_epochs)
def train_one_fold(model, train_loader, val_loader, num_classes, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train(); tloss=0
        for imgs, labs in tqdm(train_loader, desc=f"Train Epoch {ep+1}/{epochs}"):
            imgs, labs = imgs.to(device), labs.to(device)
            opt.zero_grad(); out=model(imgs)
            loss=crit(out,labs); loss.backward(); opt.step()
            tloss+=loss.item()
        model.eval(); vpreds, vlabs=[],[]
        with torch.no_grad():
            for imgs, labs in val_loader:
                imgs, labs = imgs.to(device), labs.to(device)
                p = model.predict(imgs)
                vpreds+=p.cpu().tolist(); vlabs+=labs.cpu().tolist()
        bal = calculate_balanced_accuracy(torch.tensor(vpreds), torch.tensor(vlabs), num_classes)
        print(f"Fold Val Balanced Acc after Epoch {ep+1}: {bal:.2f}%")
# Balanced Accuracy
def calculate_balanced_accuracy(preds, labels, num_classes):
    correct = torch.zeros(num_classes)
    total   = torch.zeros(num_classes)
    for i in range(num_classes):
        mask = (labels==i)
        total[i] = mask.sum().item()
        if total[i]>0:
            correct[i] = (preds[mask]==labels[mask]).sum().item()
    valid = total>0
    return (correct[valid]/total[valid]).mean().item()*100

def main():
    # 1) 数据增强 & 读数据
    train_tf = transforms.Compose([
        transforms.Resize((112,112)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((112,112)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    ds = AnimalCLEF2025('.', transform=None, load_label=True)
    idxs = [i for i, m in enumerate(ds.metadata.dataset=='SeaTurtleID2022') if m]
    subset = torch.utils.data.Subset(ds, idxs)
    full = AnimalDataset(subset, transform=train_tf)
    num_classes = len(full.label_to_idx)

    # 2) KFold 分 5 折
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(range(len(full))), 1):
        print(f"\n=== Fold {fold} ===")

        # 3) 构造每折的 train/val 子集
        tr_ds = Subset(full, tr_idx)
        va_ds = Subset(full, va_idx)
        va_ds.transform = val_tf

        tr_loader = DataLoader(tr_ds, batch_size=16, shuffle=True,  num_workers=4, pin_memory=True)
        va_loader = DataLoader(va_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

        # 4) 实例化新模型
        model = SwinFaceModelWrapper(num_classes=num_classes, confidence_threshold=0.3)
        # （可选）冻结前几层
        freeze_backbone_layers(model, unfreeze_blocks=1)

        # 5) 调用你原本的 train_model
        #    train_model(model, train_loader, val_loader, num_classes, num_epochs)
        train_model(
            model,
            train_loader=tr_loader,
            val_loader=va_loader,
            num_classes=num_classes,
            num_epochs=20
        )

if __name__ == '__main__':
    main()
