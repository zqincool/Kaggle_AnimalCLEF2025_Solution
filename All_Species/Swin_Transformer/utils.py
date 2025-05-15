import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
from glob import glob

def load_image(image_path):
    """
    Load image and return PIL Image object
    Args:
        image_path: path to the image
    Returns:
        PIL Image object
    """
    try:
        # Open image using PIL
        image = Image.open(image_path).convert('RGB')
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        # Return a black image as fallback
        return Image.new('RGB', (224, 224), color='black')

def load_sea_turtle_data(data_root, is_train=True):
    """
    Load sea turtle dataset
    
    Args:
        data_root: root directory of the dataset, e.g. 'images/SeaTurtleID2022/database'
        is_train: whether this is training data
    
    Returns:
        image_paths: list of image paths
        labels: corresponding labels list
        label_to_id: dictionary mapping labels to IDs
        id_to_label: dictionary mapping IDs to labels
    """
    # Read metadata file
    metadata = pd.read_csv('metadata.csv')
    
    # Filter sea turtle data based on identity column
    sea_turtle_metadata = metadata[metadata['identity'].str.contains('SeaTurtle', na=False)]
    
    # Only use data from database
    database_metadata = sea_turtle_metadata[sea_turtle_metadata['path'].str.contains('database', na=False)]
    
    # Print data statistics
    print("\n海龟数据统计：")
    print("总样本数:", len(database_metadata))
    print("唯一个体数:", len(database_metadata['identity'].unique()))
    
    # Get all individual IDs
    unique_identities = sorted(database_metadata['identity'].unique())
    label_to_id = {identity: idx for idx, identity in enumerate(unique_identities)}
    id_to_label = {idx: identity for identity, idx in label_to_id.items()}  # Add reverse mapping
    
    # Get image paths and labels
    image_paths = []
    labels = []
    for _, row in database_metadata.iterrows():
        # Process path
        path = row['path']
        image_path = os.path.join(os.getcwd(), path)
        
        # Print debug information
        # print(f"Trying to load image: {image_path}")
        
        if os.path.exists(image_path):
            image_paths.append(image_path)
            labels.append(label_to_id[row['identity']])
        else:
            print(f"警告：找不到图像文件 {image_path}")
    
    print(f"\n成功加载图像数: {len(image_paths)}")
    return image_paths, labels, label_to_id, id_to_label

def calculate_metrics(y_true, y_pred, known_mask):
    """Calculate BAKS and BAUS metrics"""
    # Separate known and unknown samples
    y_true_known = y_true[known_mask]
    y_pred_known = y_pred[known_mask]
    y_true_unknown = y_true[~known_mask]
    y_pred_unknown = y_pred[~known_mask]
    
    # Calculate BAKS
    baks = balanced_accuracy_score(y_true_known, y_pred_known)
    
    # Calculate BAUS
    baus = balanced_accuracy_score(y_true_unknown, y_pred_unknown)
    
    # Calculate final score
    final_score = np.sqrt(baks * baus)
    
    return baks, baus, final_score

def create_sample_submission(predictions, image_ids):
    """Create submission file"""
    submission = pd.DataFrame({
        'image_id': image_ids,
        'prediction': predictions
    })
    return submission

def get_species_from_path(image_path):
    """Extract species information from image path"""
    if 'SeaTurtleID2022' in image_path:
        return 'SeaTurtle'
    elif 'SalamanderID2025' in image_path:
        return 'Salamander'
    elif 'LynxID2025' in image_path:
        return 'Lynx'
    else:
        raise ValueError(f"Unknown species in path: {image_path}") 