"""
This file has been split into several modules:
- cnn/data.py
- cnn/model.py
- cnn/features.py
- cnn/similarity.py
- cnn/predict.py
- cnn/submission.py
- cnn/main.py (main entry)
Please use main.py as the main entry point.
"""

import os
import numpy as np
import pandas as pd
import timm
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from wildlife_datasets.datasets import AnimalCLEF2025
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity

# Create a submission file
def create_sample_submission(dataset_query, predictions, file_name='sample_submission.csv'):
    df = pd.DataFrame({
        'image_id': dataset_query.metadata['image_id'],
        'identity': predictions
    })
    df.to_csv(file_name, index=False)

def get_transformations():
    # Define a transformation that resizes the images to 384x384 pixels
    transform_display = T.Compose([
        T.Resize((384, 384)),
    ])

    # Define a transformation pipeline for preprocessing images for the model
    # This includes resizing, normalization, and converting to tensor format
    # The normalization values are based on the ImageNet dataset
    transform = T.Compose([
        # Resize((384, 384)),
        *transform_display.transforms,
        # Convert images to tensor format
        T.ToTensor(),
        # Normalize the images using ImageNet statistics
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Check that this step is being done
    print("transform")

    return transform


# Validate image paths before processing

def load_dataset(root, transform):
# Load the AnimalCLEF2025 dataset with the specified transformations
    dataset = AnimalCLEF2025(root, transform=transform, load_label=True)
    print("load dataset")

    # Plot a sample grid of the data
    # dataset.plot_grid()

    # Load pretrained MegaDescriptor to extract features from all images 
    dataset_database = dataset.get_subset(dataset.metadata['split'] == 'database')
    dataset_query = dataset.get_subset(dataset.metadata['split'] == 'query')
    n_query = len(dataset_query)

    print("load database and query")

    return dataset, dataset_database, dataset_query, n_query



def load_model(name, device):
    # Load the model with pretrained weights
    # The model is set to output features instead of classification scores (num_classes=0)
    # The model is moved to the specified device (GPU or CPU)

    model = timm.create_model(name, num_classes=0, pretrained=True).to(device)
    
    print("load model")

    return model

def extract_features(model, device, dataset_database, dataset_query):
    # Use batch processing for feature extraction
    extractor = DeepFeatures(model, device=device, batch_size=32, num_workers=0)
    features_database = extractor(dataset_database)
    features_query = extractor(dataset_query)
    print("extract features")

    return  features_database, features_query

def compute_similarity(features_query, features_database, n_query):
    # Use cosine to compute similarity 
    similarity = CosineSimilarity()(features_query, features_database)
    pred_idx = similarity.argsort(axis=1)[:, -1]
    pred_scores = similarity[range(n_query), pred_idx]
    print("compute similarity")

    return pred_idx, pred_scores

def perform_predictions(pred_idx, pred_scores, dataset_database):
        # Perform predictions by assigning labels based on similarity scores
        new_individual = 'new_individual'
        threshold = 0.6
        labels = dataset_database.labels_string
        predictions = labels[pred_idx]
        predictions[pred_scores < threshold] = new_individual

        print("perform predictions")

        return predictions

def main(): 
     # Specify the root where the data is stored
    root = 'data/animal-clef-2025'
    # transform
    transform = get_transformations()
    # Load the dataset
    dataset, dataset_database, dataset_query, n_query = load_dataset(root, transform)
    # Load the model
    name = 'hf-hub:BVRA/MegaDescriptor-L-384'
    device = 'cuda'
    model = load_model(name, device)
    # Extract features from the database and query datasets
    features_database, features_query = extract_features(model, device, dataset_database, dataset_query)
    # Compute similarity scores between query and database features
    pred_idx, pred_scores = compute_similarity(features_query, features_database, n_query)
    labels = dataset_database.labels_string
    # Perform predictions based on similarity scores
    predictions = perform_predictions(pred_idx, pred_scores, dataset_database)
    
    # Create a sample submission file with the predictions
    create_sample_submission(dataset_query, predictions, file_name='sample_submission.csv')
    print("create sample submission")

    # Check if the submission file is created successfully
    print(os.path.exists('sample_submission.csv'))  # Should print True if the file exists
    print("Reach end of file")

if __name__ == "__main__":
    main()
    