import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from data import get_transformations, load_dataset
from model import load_model
from tqdm import tqdm

# python train.py 

def parse_args():
    parser = argparse.ArgumentParser(description='Train a ResNet model on AnimalDataset')
    parser.add_argument('--root', type=str, default='/root/autodl-tmp', help='Root directory of images')
    parser.add_argument('--csv', type=str, default='/root/autodl-tmp/metadata.csv', help='CSV file path')
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50', 'resnet101'], help='Model name')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of DataLoader workers')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='Validation set ratio (0-1)')
    parser.add_argument('--save-freq', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout probability for the final layer')
    return parser.parse_args()


def train():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'
    transform = get_transformations()
    _, dataset_database, _, _ = load_dataset(args.root, args.csv, transform)
    # Encode identity column as integer labels
    identities = sorted(dataset_database.data['identity'].unique())
    id2idx = {k: i for i, k in enumerate(identities)}
    dataset_database.data['label_idx'] = dataset_database.data['identity'].map(id2idx)
    # Split train/val set
    label_counts = dataset_database.data['label_idx'].value_counts()
    can_stratify = (label_counts >= 2).all()
    if can_stratify:
        train_idx, val_idx = train_test_split(
            range(len(dataset_database)),
            test_size=args.val_ratio,
            stratify=dataset_database.data['label_idx'],
            random_state=42
        )
    else:
        print('Warning: Some classes have less than 2 samples, using random split without stratify.')
        train_idx, val_idx = train_test_split(
            range(len(dataset_database)),
            test_size=args.val_ratio,
            random_state=42
        )
    def collate_fn(batch):
        images = torch.stack([item['image'] for item in batch])
        labels = torch.tensor([id2idx[item['label']] for item in batch], dtype=torch.long)
        return images, labels
    train_subset = torch.utils.data.Subset(dataset_database, train_idx)
    val_subset = torch.utils.data.Subset(dataset_database, val_idx)
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    num_classes = len(identities)
    # Fix torchvision pretrained weights warning
    model = load_model(name=args.model, num_classes=num_classes, pretrained=args.pretrained, device=device, dropout_p=args.dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model.train()

    os.makedirs('checkpoints', exist_ok=True)
    best_val_acc = 0.0
    best_ckpt_path = 'checkpoints/best_checkpoint.pth'
    for epoch in range(args.epochs):
        running_loss = 0.0
        train_iter = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]')
        for images, labels in train_iter:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            train_iter.set_postfix(loss=loss.item())
        avg_loss = running_loss / len(train_subset)
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_iter = tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Val]')
        with torch.no_grad():
            for images, labels in val_iter:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                val_iter.set_postfix(loss=loss.item())
        avg_val_loss = val_loss / len(val_subset)
        val_acc = correct / total if total > 0 else 0
        print(f'Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val Acc: {val_acc:.4f}')
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs:
            ckpt_path = f'checkpoints/checkpoint_epoch_{epoch+1:02d}.pth'
            torch.save(model.state_dict(), ckpt_path)
            print(f'Checkpoint saved to {ckpt_path}')
        # Save best accuracy model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_ckpt_path)
            print(f'Best checkpoint updated at epoch {epoch+1} with val_acc={val_acc:.4f}')
        model.train()

if __name__ == '__main__':
    train() 