import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import pandas as pd
import argparse
from net import SimpleCNN
from dataloader.dataloader import get_transforms

def get_class_mapping_from_dir(train_dir):
    # Use subfolder names as class names, sorted alphabetically
    class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
    idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}
    return class_to_idx, idx_to_class

class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        for fname in os.listdir(root_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.samples.append(os.path.join(root_dir, fname))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_path

def main(args):
    # Get class mapping
    class_to_idx, idx_to_class = get_class_mapping_from_dir(args.train_dir)
    num_classes = len(class_to_idx)
    unknown_names = ['unknown', 'gan_unknown']

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Load test set
    transform = get_transforms(args.img_size)
    test_dataset = TestDataset(args.test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    results = []
    with torch.no_grad():
        for images, paths in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            for path, pred in zip(paths, preds.cpu().numpy()):
                image_id = os.path.basename(path)
                class_name = idx_to_class.get(pred, 'unknown')
                if class_name in unknown_names:
                    identity = 'new_individual'
                else:
                    identity = f'SeaTurtleID2022_{class_name}'
                results.append({'image_id': image_id, 'identity': identity})
    submission_df = pd.DataFrame(results)
    submission_df.to_csv(args.output_csv, index=False)
    print(f'Submission file saved as {args.output_csv}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, required=True, help='Test set root directory')
    parser.add_argument('--train_dir', type=str, required=True, help='Train set root directory (used to infer class names and number)')
    parser.add_argument('--model_path', type=str, default='cnn_model.pth', help='Path to trained model checkpoint')
    parser.add_argument('--output_csv', type=str, default='submission.csv', help='Output submission file')
    parser.add_argument('--img_size', type=int, default=32, help='Image size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    args = parser.parse_args()
    main(args) 

# python predict_test.py --test_dir /root/autodl-tmp/images/SeaTurtleID2022/query/images --train_dir /root/autodl-tmp/padded_images/images/SeaTurtleID2022/database/turtles-data/data/images --model_path cnn_model.pth --output_csv submission.csv --img_size 32 --batch_size 32