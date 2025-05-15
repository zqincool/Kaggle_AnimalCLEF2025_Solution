from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd
import random

class TripletDataset(Dataset):
    def __init__(self, base_dataset):
        """
        base_dataset: 一个 AnimalDataset 实例
        """
        self.base = base_dataset
        # 构造 label -> [indices]
        labels = [lbl.item() for _, lbl in self.base]
        self.label_to_indices = {}
        for idx, lab in enumerate(labels):
            self.label_to_indices.setdefault(lab, []).append(idx)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img_a, lab_a = self.base[idx]
        lab_a = lab_a.item()

        # positive
        pos_list = self.label_to_indices[lab_a]
        pos_idx = idx
        while pos_idx == idx:
            pos_idx = random.choice(pos_list)
        img_p, _ = self.base[pos_idx]

        # negative
        neg_lab = lab_a
        while neg_lab == lab_a:
            neg_lab = random.choice(list(self.label_to_indices.keys()))
        neg_idx = random.choice(self.label_to_indices[neg_lab])
        img_n, _ = self.base[neg_idx]

        return img_a, img_p, img_n

class AnimalDataset(Dataset):
    def __init__(self, subset_dataset, transform=None):
        """
        subset_dataset: torch.utils.data.Subset (已经筛过 split/database)
        transform: torchvision.transforms
        """
        self.dataset = subset_dataset
        self.transform = transform

        # 1) 拿到完整的 metadata
        md = self.dataset.dataset.metadata

        # 2) 如果是 Subset，就只保留那些 indices 对应的行
        if hasattr(self.dataset, 'indices'):
            md = md.iloc[self.dataset.indices]

        # 3) 从这部分 metadata 中提取所有非空的 identity
        valid_labels = [lab for lab in md['identity'].unique() if pd.notna(lab)]

        # 4) 加上 new_individual
        valid_labels.append('new_individual')

        # 5) 排序并生成映射
        valid_labels = sorted(valid_labels)
        self.label_to_idx = {lab: idx for idx, lab in enumerate(valid_labels)}

        # （可选）打印检查
        print(f"[AnimalDataset] 子集共 {len(md)} 张图，检测到 {len(valid_labels)} 类")
        print("前 5 类示例：", valid_labels[:5])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        label = 'new_individual' if pd.isna(label) else label
        if label not in self.label_to_idx:
            label = 'new_individual'
        label_idx = self.label_to_idx[label]
        return image, torch.tensor(label_idx, dtype=torch.long)

class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels