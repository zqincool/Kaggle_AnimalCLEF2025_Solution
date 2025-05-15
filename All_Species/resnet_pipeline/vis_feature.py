import os
import pandas as pd
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

csv_path = '/root/autodl-tmp/metadata.csv'
img_col = 'path' 
img_root = '/root/autodl-tmp'
features_path = 'features_database.npy' 
output_dir = 'vis'
num_samples = 100
output_fig = os.path.join(output_dir, 'tsne_features.png')

os.makedirs(output_dir, exist_ok=True)


df = pd.read_csv(csv_path)
print("df shape:", df.shape)

features = np.load(features_path)
print("features shape:", features.shape)

df = df.iloc[:features.shape[0]].copy()
df['full_path'] = df[img_col].apply(lambda x: os.path.join(img_root, x))

mask = df['full_path'].apply(os.path.exists)
df_exist = df[mask].reset_index(drop=True)
features_exist = features[mask.values]

sample_indices = random.sample(range(len(df_exist)), min(num_samples, len(df_exist)))
sampled_df = df_exist.iloc[sample_indices].reset_index(drop=True)
sampled_features = features_exist[sample_indices]

perplexity = min(30, len(sampled_features) - 1)
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
features_2d = tsne.fit_transform(sampled_features)

plt.figure(figsize=(8, 6))
ax = plt.gca()
ax.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.2)

for i, row in sampled_df.iterrows():
    img_path = row['full_path']
    x, y = features_2d[i]
    try:
        img = Image.open(img_path).convert('RGB')
        img.thumbnail((32, 32))
        imagebox = OffsetImage(np.array(img), zoom=1)
        ab = AnnotationBbox(imagebox, (x, y), frameon=False)
        ax.add_artist(ab)
    except Exception as e:
        print(f"Error loading {img_path}: {e}")

plt.title('t-SNE of Features with Sample Images')
plt.savefig(output_fig, dpi=200)
plt.close()
print(f"t-SNE plot with images saved to {output_fig}")