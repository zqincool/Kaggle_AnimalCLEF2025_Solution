import argparse
import torch
import numpy as np
from model import load_model
from data import get_transformations, load_dataset
from wildlife_tools.features import DeepFeatures
from torch.utils.data import Dataset

def extract_features(model, device, dataset_database, dataset_query):
    extractor = DeepFeatures(model, device=device, batch_size=32, num_workers=0)
    features_database = extractor(dataset_database)
    features_query = extractor(dataset_query)
    print("extract features")
    return features_database, features_query

class FeatureDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
    def __len__(self):
        return len(self.base_dataset)
    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        return item['image'], item['label']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data-root', type=str, required=True, help='Root directory of images')
    parser.add_argument('--csv', type=str, required=True, help='CSV file path')
    parser.add_argument('--output-db', type=str, required=True, help='Output .npy for database features')
    parser.add_argument('--output-query', type=str, required=True, help='Output .npy for query features')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--arch', type=str, default='resnet50', help='Model architecture')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    transform = get_transformations()
    _, dataset_database, dataset_query, n_query = load_dataset(args.data_root, args.csv, transform)
    identities = sorted(dataset_database.base_dataset.data['identity'].unique()) if hasattr(dataset_database, 'base_dataset') else sorted(dataset_database.data['identity'].unique())
    num_classes = len(identities)
    model = load_model(name=args.arch, num_classes=num_classes, pretrained=False, device=device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # Wrap as FeatureDataset
    dataset_database = FeatureDataset(dataset_database)
    dataset_query = FeatureDataset(dataset_query)

    features_database, features_query = extract_features(model, device, dataset_database, dataset_query)
    np.save(args.output_db, features_database)
    np.save(args.output_query, features_query)
    print(f"Saved database features to {args.output_db}")
    print(f"Saved query features to {args.output_query}")


#    python features.py --model checkpoints/checkpoint_epoch_30.pth --data-root /root/autodl-tmp --csv /root/autodl-tmp/metadata.csv --output-db features_database.npy --output-query features_query.npy