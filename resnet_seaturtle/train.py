import torch
import torch.nn as nn
import torch.optim as optim
from net import SimpleCNN
from dataloader.dataloader import get_transforms, SeaTurtleDataset
import argparse
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
import pandas as pd
import os

def custom_collate(batch):
    images, labels, paths = zip(*batch)
    images = torch.stack(images, 0)
    labels = torch.tensor(labels)
    return images, labels, paths

def evaluate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        loop = tqdm(val_loader, desc="Validation", leave=False)
        for inputs, labels, paths in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total if total > 0 else 0
    return acc

def train(model, train_loader, val_loader, criterion, optimizer, device, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch [{epoch+1}/{epochs}]")
        for i, (inputs, labels, paths) in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loop.set_postfix(loss=running_loss/(i+1))
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")
        val_acc = evaluate(model, val_loader, device)
        print(f"Validation Accuracy: {val_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='dataset root directory')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--val_split', type=float, default=0.2, help='validation split ratio')
    args = parser.parse_args()

    device = torch.device("cuda")
    transform = get_transforms(args.img_size)
    full_dataset = SeaTurtleDataset(args.data_dir, transform=transform)
    num_classes = len(full_dataset.class_to_idx)
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Export split csv files
    train_csv = pd.DataFrame([full_dataset.csv_records[i] for i in train_dataset.indices])
    train_csv.to_csv('train_split.csv', index=False)
    val_csv = pd.DataFrame([full_dataset.csv_records[i] for i in val_dataset.indices])
    val_csv.to_csv('val_split.csv', index=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate)
    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train(model, train_loader, val_loader, criterion, optimizer, device, epochs=args.epochs)
    torch.save(model.state_dict(), 'cnn_model.pth')
    print('Training finished. Model saved as cnn_model.pth')

    model.eval()
    idx_to_class = {v: k for k, v in full_dataset.class_to_idx.items()}
    unknown_names = full_dataset.unknown_folder_names if hasattr(full_dataset, 'unknown_folder_names') else ['unknown', 'gan_unknown']
    results = []
    with torch.no_grad():
        for images, labels, paths in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            for path, pred in zip(paths, preds.cpu().numpy()):
                image_id = os.path.basename(path)
                class_name = idx_to_class[pred]
                if class_name in unknown_names:
                    identity = 'new_individual'
                else:
                    identity = f'SeaTurtleID2022_{class_name}'
                results.append({'image_id': image_id, 'identity': identity})
    submission_df = pd.DataFrame(results)
    submission_df.to_csv('submission.csv', index=False)
    print('Submission file saved as submission.csv')

# python train.py --data_dir /root/autodl-tmp/images/SeaTurtleID2022/database/turtles-data/data/images --epoch=20 --batch_size=64 --lr=0.001 --val_split=0.3
# python train.py --data_dir /root/autodl-tmp/padded_images/images/SeaTurtleID2022/database/turtles-data/data/images --epoch=25 --batch_size=64 --lr=0.001 --val_split=0.3