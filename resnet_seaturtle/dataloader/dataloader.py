import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

def get_transforms(img_size=32):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class SeaTurtleDataset(Dataset):
    def __init__(self, root_dir, transform=None, unknown_folder_names=['unknown', 'gan_unknown']):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.unknown_folder_names = unknown_folder_names
        self.csv_records = []
        self._find_classes_and_samples()

    def _find_classes_and_samples(self):
        # Each subfolder is a class, including 'unknown' or 'gan_unknown' if present
        classes = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        classes.sort()
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        for cls_name in classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(cls_dir, fname)
                    self.samples.append((img_path, self.class_to_idx[cls_name]))
                    self.csv_records.append({
                        'image_id': fname,
                        'identity': cls_name,
                        'dataset': 'trainval',
                        'new_identity': cls_name in self.unknown_folder_names
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, img_path

def get_dataloader(data_dir, batch_size=32, shuffle=True, img_size=32, num_workers=2):
    transform = get_transforms(img_size)
    dataset = SeaTurtleDataset(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader, dataset.class_to_idx 