# make_submission_retrieval.py
import os
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image

from wildlife_datasets.datasets import AnimalCLEF2025
from my_dataset import AnimalDataset
from model import convnext_tiny as create_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str,
                        default='/content/drive/MyDrive/ConvNext')
    parser.add_argument('--weights', type=str,
                        default='weights/best_model.pth',
                        help='finetune 后的分类头权重')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='余弦相似度阈值；低于此则判 new_individual')
    return parser.parse_args()

class QueryDataset(Dataset):
    def __init__(self, df, root, transform):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = int(row['image_id'])
        path   = os.path.join(self.root, row['path'])
        img    = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, img_id

def extract_features(model, loader, device):
    feats, ids = [], []
    model.eval()
    with torch.no_grad():
        for images, img_ids in loader:
            images = images.to(device)
            # 1) 用 forward_features 拿 convnext 的中间特征
            feat = model.forward_features(images)
            # 2) flatten 如果必要
            if feat.ndim > 2:
                feat = feat.flatten(1)
            # 3) L2 归一化
            feat = torch.nn.functional.normalize(feat, dim=1)
            feats.append(feat.cpu())
            ids.extend([int(x) for x in img_ids])
    return torch.cat(feats, 0), ids


def main():
    args = parse_args()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 1) 读取 metadata
    meta = pd.read_csv(os.path.join(args.data_root, 'metadata.csv'))

    img_size = 224
    tfm = transforms.Compose([
    transforms.Resize(int(img_size*1.143)),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])    
    # 2) 重建 label_to_idx
    db_meta = meta[meta['split']=='database']
    full_clef = AnimalCLEF2025(args.data_root, transform=None, load_label=True)
    db_subset = Subset(full_clef, db_meta.index.tolist())
    train_ds  = AnimalDataset(db_subset, transform=tfm)
    label_to_idx = train_ds.label_to_idx
    idx_to_label = {v:k for k,v in label_to_idx.items()}

    # 3) 定义 transform
    img_size = 224
    tfm = transforms.Compose([
        transforms.Resize(int(img_size*1.143)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    # 4) DataLoader for database & query
    db_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    query_meta = meta[meta['split']=='query'][['image_id','path']]
    query_loader = DataLoader(
        QueryDataset(query_meta, args.data_root, transform=tfm),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 5) 加载模型并提取特征
    model = create_model(num_classes=len(label_to_idx)).to(DEVICE)
    ckpt = torch.load(os.path.join(args.data_root, args.weights), map_location=DEVICE)
    model.load_state_dict(ckpt); model.eval()

    db_feats, db_ids = extract_features(model, db_loader, DEVICE)
    q_feats,  q_ids  = extract_features(model, query_loader, DEVICE)

    # 6) 计算余弦相似度并检索
    # sim: [N_q, N_db]
    sim = q_feats @ db_feats.t()    
    k = 5
    topk_sim, topk_idx = sim.topk(k, dim=1, largest=True, sorted=True)

    results = []
    new_idx = label_to_idx['new_individual']

    import matplotlib.pyplot as plt

    # 假设你已经有了 q_feats, db_feats
    import sklearn.metrics.pairwise as pw
    sim = pw.cosine_similarity(q_feats, db_feats)
    top_sim = sim.max(axis=1)
# 如果是 torch.Tensor，先转成 numpy：
    if isinstance(top_sim, torch.Tensor):
        sim_vals = top_sim.cpu().numpy()
    else:
        sim_vals = top_sim

    plt.figure(figsize=(8,6))
    plt.hist(sim_vals, bins=50)
    plt.title("Query → Database Top-1 Cosine Similarity")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig("sim_hist.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 已保存相似度直方图到 sim_hist.png")

    for qid, sims_row, idxs_row in zip(q_ids, topk_sim, topk_idx):
        # 如果最邻近的相似度都低于阈值，就当 unknown
        if sims_row[0].item() < args.threshold:
            pred_identity = 'new_individual'
        else:
            # 拿这 k 个 neighbor 的 label_idx 做一次多数投票
            neighbor_labels = []
            for db_i in idxs_row:
                # train_ds[db_i] 会返回 (img, label_idx_tensor)
                _, lbl = train_ds[db_i]
                neighbor_labels.append(lbl.item())
            # 统计出现次数最多的那个 label_idx
            from collections import Counter
            most_common = Counter(neighbor_labels).most_common(1)[0][0]
            pred_identity = idx_to_label[most_common]
        results.append((qid, pred_identity))

    # 7) 写 submission
    sub = pd.DataFrame(results, columns=['image_id','identity'])
    sub.to_csv('sample_submission.csv', index=False)
    print("Done, wrote sample_submission.csv")

if __name__ == '__main__':
    main()
