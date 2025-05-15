import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class AnimalDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, row['path'])
        image = Image.open(img_path).convert('RGB')
        label = row['identity'] if 'identity' in row else -1
        image_id = row['image_id']
        if self.transform:
            image = self.transform(image)
        return {
            'image': image,
            'label': label,
            'image_id': image_id,
            'path': img_path
        }

def get_transformations():
    transform_display = T.Compose([
        T.Resize((384, 384)),
    ])
    transform = T.Compose([
        *transform_display.transforms,
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print("transform")
    return transform

def load_dataset(root, csv_file, transform):
    # Read CSV, split by 'split' column if present, otherwise use all data
    df = pd.read_csv(csv_file)
    if 'split' in df.columns:
        df_database = df[df['split'] == 'database']
        df_query = df[df['split'] == 'query']
    else:
        df_database = df
        df_query = df
    # Save temporary csv for Dataset
    tmp_db_csv = os.path.join(root, '_tmp_database.csv')
    tmp_q_csv = os.path.join(root, '_tmp_query.csv')
    df_database.to_csv(tmp_db_csv, index=False)
    df_query.to_csv(tmp_q_csv, index=False)
    dataset_database = AnimalDataset(tmp_db_csv, root, transform=transform)
    dataset_query = AnimalDataset(tmp_q_csv, root, transform=transform)
    n_query = len(dataset_query)
    print("load database and query from csv")
    return None, dataset_database, dataset_query, n_query 