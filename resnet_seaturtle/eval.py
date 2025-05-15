import torch
import torch.nn as nn
from net import SimpleCNN
from dataloader.dataloader import get_dataloader
import argparse


def evaluate(model, eval_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in eval_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total if total > 0 else 0
    print(f'Accuracy on eval set: {acc:.2f}%')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='eval dataset root directory')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--model_path', type=str, default='cnn_model.pth')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_loader, class_to_idx = get_dataloader(args.data_dir, batch_size=args.batch_size, shuffle=False, img_size=args.img_size, num_workers=args.num_workers)
    num_classes = len(class_to_idx)
    model = SimpleCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    eval_loader = None
    print("Please fill in the dataloader section before running evaluation.")
