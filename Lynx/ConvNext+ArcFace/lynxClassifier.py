"""
ConvNeXt + ArcFace Individual Recognition Script
================================================
* Using ConvNeXt as backbone network + ArcFace loss function
* Features: P×K sampling, multiple data augmentations, OneCycleLR scheduler
* Designed for Lynx individual recognition task
"""

import os, math, random, warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
import timm, numpy as np, pandas as pd
from wildlife_datasets.datasets import AnimalCLEF2025
from utils import calculate_metrics

# ------------------ P×K Balanced Sampler ------------------
class PKSampler(torch.utils.data.Sampler):
    def __init__(self, labels, P, K):
        self.labels = np.array(labels)
        self.P, self.K = P, K
        self.unique = np.unique(self.labels)
        self.id2idx = {l: np.where(self.labels == l)[0] for l in self.unique}
    def __iter__(self):
        idx = []
        ids = np.random.permutation(self.unique)
        for i in range(0, len(ids), self.P):
            batch_ids = ids[i:i+self.P]
            for lab in batch_ids:
                pos = self.id2idx[lab]
                choice = np.random.choice(pos, self.K, replace=len(pos) < self.K)
                idx.extend(choice.tolist())
        return iter(idx)
    def __len__(self):
        return len(self.labels)

# ------------------ Dataset ------------------
class LynxDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset, self.transform = subset, transform
        metas = subset.dataset.metadata
        ids = metas[metas['dataset']=='LynxID2025']['identity'].dropna().unique().tolist()
        ids.sort(); ids.append('new_individual')
        self.id2idx = {lab:i for i,lab in enumerate(ids)}
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        img, lab = self.subset[idx]
        if self.transform: img = self.transform(img)
        lab = 'new_individual' if pd.isna(lab) else lab
        lab = lab if lab in self.id2idx else 'new_individual'
        return img, torch.tensor(self.id2idx[lab], dtype=torch.long)

# ------------------ ArcFace Layer ------------------
class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, scale=64.0, margin=0.3):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s, self.m = scale, margin
        self.cos_m, self.sin_m = math.cos(margin), math.sin(margin)
        self.th, self.mm = math.cos(math.pi - margin), math.sin(math.pi - margin)*margin
        
    def forward(self, emb, labels=None):
        w_norm = F.normalize(self.weight, p=2, dim=1)
        cos = F.linear(emb, w_norm)
        if labels is None:
            return cos * self.s
        sine = torch.sqrt((1.0 - cos.pow(2)).clamp(0,1))
        phi  = cos * self.cos_m - sine * self.sin_m
        phi  = torch.where(cos > self.th, phi, cos - self.mm)
        logits = cos.clone()
        logits[torch.arange(len(labels)), labels] = phi[torch.arange(len(labels)), labels]
        return logits * self.s

# ------------------ ConvNeXt + ArcFace ------------------
class ConvNextArcFace(nn.Module):
    def __init__(self, backbone_name='convnext_base', num_classes=0, emb_size=512, s=64.0, m=0.3):
        super().__init__()
        # 主干网络
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, global_pool='avg')
        feat_dim = self.backbone.num_features
        # BNNeck + 投影
        self.bnneck = nn.BatchNorm1d(feat_dim)
        self.bnneck.bias.requires_grad_(False)
        self.proj = nn.Linear(feat_dim, emb_size, bias=False)
        self.bn_emb = nn.BatchNorm1d(emb_size)
        self.bn_emb.bias.requires_grad_(False)
        # ArcFace
        self.arcface = ArcFace(emb_size, num_classes, s, m)
    
    def forward(self, x, labels=None):
        feats = self.backbone(x)
        feats = self.bnneck(feats)
        emb = self.bn_emb(self.proj(feats))
        emb = F.normalize(emb, p=2, dim=1)
        logits = self.arcface(emb, labels)
        return logits, emb

# ------------------ Train & Eval ------------------
def train_and_evaluate(P=8, K=4, batch_size=32, epochs=30, patience=8, conf_thr=0.3, accum_steps=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # transforms
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.3,0.3,0.3,0.15),
        transforms.RandomAffine(0, translate=(0.1,0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        transforms.RandomErasing(p=0.3)
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    # dataset
    ds = AnimalCLEF2025('.', transform=None, load_label=True)
    mask = (ds.metadata['dataset']=='LynxID2025') & (ds.metadata['split']=='database')
    subset = Subset(ds, np.where(mask)[0])
    full = LynxDataset(subset, transform=train_tf)
    n_train = int(0.7*len(full))
    train_set, val_set = torch.utils.data.random_split(full, [n_train, len(full)-n_train], generator=torch.Generator().manual_seed(42))
    val_set.dataset.transform = val_tf
    # loaders
    train_labels = [l.item() for _,l in train_set]
    sampler = PKSampler(train_labels, P, K)
    train_loader = DataLoader(train_set, batch_size=P*K, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # 创建模型
    print("Using ConvNeXt + ArcFace model")
    num_cls = len(full.id2idx)
    model = ConvNextArcFace('convnext_base', num_cls).to(device)
    
    # Freeze first 50% of layers
    total = len(list(model.backbone.parameters()))
    for i, (_, p) in enumerate(model.backbone.named_parameters()):
        p.requires_grad = (i/total) > 0.5
    
    # Optimizer configuration
    optim_groups = [
        {'params': model.backbone.parameters(), 'lr': 1e-5},
        {'params': list(model.proj.parameters()) + 
                  list(model.bn_emb.parameters()) + 
                  list(model.arcface.parameters()), 'lr': 5e-4}
    ]
    optimizer = optim.AdamW(optim_groups, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[1e-4,6e-4], 
                                            steps_per_epoch=len(train_loader), 
                                            epochs=epochs, pct_start=0.2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_score, wait = 0.0, 0
    model_save_path = 'convnext_arcface_confidence03.pth'
    
    for ep in range(1, epochs+1):
        # ---------- train ----------
        model.train(); tloss=0
        optimizer.zero_grad()  # Zero gradients at the beginning of epoch
        for i, (imgs,labs) in enumerate(tqdm(train_loader, desc=f'Train {ep}/{epochs}')):
            imgs,labs = imgs.to(device), labs.to(device)
            logits,_ = model(imgs, labs)
            loss = criterion(logits,labs) / accum_steps  # Normalize loss
            loss.backward()
            tloss += loss.item() * accum_steps  # Denormalize for reporting
            
            # Update weights after accumulation steps or at the end
            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
        tloss /= len(train_loader)
        # ---------- val ----------
        model.eval(); vloss, preds, gts = 0, [], []
        with torch.no_grad():
            for imgs,labs in tqdm(val_loader, desc='Val'):
                imgs,labs = imgs.to(device), labs.to(device)
                logits,_ = model(imgs)
                vloss += criterion(logits,labs).item()
                prob = F.softmax(logits,1); mx,pred = prob.max(1)
                pred[mx<conf_thr] = -1
                preds.append(pred.cpu().numpy()); gts.append(labs.cpu().numpy())
        vloss /= len(val_loader); preds=np.concatenate(preds); gts=np.concatenate(gts)
        known_mask = gts>=0
        acc = (preds==gts).mean()
        def safe_metric(y_true,y_pred,mask):
            has_unknown=(~mask).sum()>0
            try:
                baks = calculate_metrics(y_true[mask],y_pred[mask],np.ones(mask.sum(),bool))[0] if mask.sum()>0 else 0
            except: baks=0
            try:
                baus = calculate_metrics(y_true[~mask],y_pred[~mask],np.ones((~mask).sum(),bool))[0] if has_unknown else 0
            except: baus=0
            final=math.sqrt(baks*baus) if has_unknown and baus>0 else baks
            return baks,baus,final
        baks,baus,final = safe_metric(gts,preds,known_mask)
        print(f'Epoch{ep}: TLoss={tloss:.4f} VLoss={vloss:.4f} Acc={acc:.3%} BAKS={baks:.4f} BAUS={baus:.4f} Final={final:.4f}')
        if final>best_score:
            best_score,wait = final,0
            torch.save(model.state_dict(), model_save_path)
            print(f'  * Best model saved: {model_save_path}')
        else:
            wait+=1; print(f'  No improvement {wait}/{patience}')
            if wait>=patience: break
    
    model.load_state_dict(torch.load(model_save_path))
    print('Training completed, best score:', best_score)
    return model

# ------------------ main ------------------
if __name__=='__main__':
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        print('Using GPU:', torch.cuda.get_device_name(0))
    
    train_and_evaluate()
