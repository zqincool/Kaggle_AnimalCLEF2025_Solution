import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from wildlife_datasets.datasets import AnimalCLEF2025
# 导入Swin Transformer模型
from swinTransformerClassifier import AnimalDataset, AnimalReIDModel
import torch.nn as nn

class CustomAnimalReIDModel(AnimalReIDModel):
    def __init__(self, num_classes, confidence_threshold=0.3):
        super().__init__(num_classes, confidence_threshold)
        # 修改模型结构以匹配保存的权重
        self.feature_dim = self.backbone.num_features
        
        # 保持与训练时完全相同的结构
        self.dropout1 = nn.Dropout(0.3)  # 改回0.3
        self.bn1 = nn.BatchNorm1d(self.feature_dim)  # 使用完整的特征维度
        self.fc1 = nn.Linear(self.feature_dim, self.feature_dim // 2)
        self.dropout2 = nn.Dropout(0.3)  # 添加第二个dropout
        self.bn2 = nn.BatchNorm1d(self.feature_dim // 2)
        self.classifier = nn.Linear(self.feature_dim // 2, num_classes)
        
        self.confidence_threshold = confidence_threshold
        print(f"\n模型置信度阈值: {self.confidence_threshold}")
    
    def forward(self, x):
        features = self.backbone(x)
        features = self.bn1(features)
        features = self.dropout1(features)
        features = torch.relu(self.fc1(features))
        features = self.bn2(features)
        features = self.dropout2(features)
        out = self.classifier(features)
        return out

def load_model_weights(model, state_dict):
    """
    安全地加载模型权重，处理键值不匹配的情况
    """
    # 创建新的state_dict，只包含当前模型需要的键
    new_state_dict = {}
    
    # 需要加载的键
    needed_keys = [
        'backbone',  # backbone的所有参数
        'dropout1.weight', 'dropout1.bias',
        'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var',
        'fc1.weight', 'fc1.bias',
        'dropout2.weight', 'dropout2.bias',
        'bn2.weight', 'bn2.bias', 'bn2.running_mean', 'bn2.running_var',
        'classifier.weight', 'classifier.bias'
    ]
    
    # 遍历原始state_dict
    for k, v in state_dict.items():
        # 检查是否是backbone的参数
        if k.startswith('backbone.'):
            new_state_dict[k] = v
        # 检查是否是其他需要的参数
        elif any(k.endswith(key) for key in needed_keys):
            new_state_dict[k] = v
    
    # 加载新的state_dict
    model.load_state_dict(new_state_dict, strict=False)
    return model

def predict_all_query_images():
    """
    针对所有动物图像的预测函数，使用训练好的Swin Transformer模型
    包括：猞猁(Lynx)、蝾螈(Salamander)和海龟(Sea Turtle)
    """
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据集
    root = '.'
    dataset = AnimalCLEF2025(root, transform=None, load_label=True)
    
    # 创建测试转换
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # 选择数据库子集来创建标签映射
    db_mask = (dataset.metadata['split'] == 'database')
    db_indices = np.where(db_mask)[0].tolist()
    db_subset = torch.utils.data.Subset(dataset, db_indices)
    
    # 创建数据集以获取标签映射
    temp_dataset = AnimalDataset(db_subset, transform=None)
    id2idx = temp_dataset.label_to_idx
    idx2id = {idx: label for label, idx in id2idx.items()}
    
    # 选择查询子集
    query_mask = (dataset.metadata['split'] == 'query')
    query_indices = np.where(query_mask)[0].tolist()
    query_subset = torch.utils.data.Subset(dataset, query_indices)
    
    # 类别数量
    num_classes = len(id2idx)
    print(f"类别数量: {num_classes}")
    
    # 创建模型
    print("使用Swin Transformer模型进行预测")
    model = CustomAnimalReIDModel(num_classes=num_classes, confidence_threshold=0.3).to(device)
    
    # 加载训练模型
    model_path = 'swin_final_ep20.pth'
    try:
        # 使用weights_only=True来安全加载模型权重
        state_dict = torch.load(model_path, weights_only=True)
        # 使用自定义的加载函数
        model = load_model_weights(model, state_dict)
        print(f"模型加载成功: {model_path}")
    except Exception as e:
        print(f"模型加载错误: {e}")
        return
    
    model.eval()
    
    # 获取查询图像ID
    image_ids = []
    for i in query_indices:
        image_id = dataset.metadata.iloc[i]['image_id']
        image_ids.append(image_id)
    
    # 进行预测
    predictions = []
    print(f"为{len(query_subset)}张图像生成预测结果...")
    
    for i, (image, _) in enumerate(tqdm(query_subset)):
        if test_transform:
            image = test_transform(image)
        
        # 添加批次维度
        image = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            # 使用模型的predict方法进行预测
            predicted = model.predict(image)
            
            # 将预测结果转换为标签
            if predicted.item() == -1:  # -1 表示 new_individual
                predictions.append("new_individual")
            else:
                label = idx2id[predicted.item()]
                predictions.append(label)
    
    # 创建提交数据帧
    submission = pd.DataFrame({
        'image_id': image_ids,
        'identity': predictions
    })
    
    # 按image_id排序
    submission = submission.sort_values('image_id')
    
    # 保存提交文件
    output_file = 'all_species_submission.csv'
    submission.to_csv(output_file, index=False)
    print(f"\n提交文件已保存到: {output_file}")
    
    # 打印预测类别分布
    class_counts = submission['identity'].value_counts()
    print("\n预测分布:")
    print(f"new_individual: {class_counts.get('new_individual', 0)}")
    print(f"已知个体: {len(submission) - class_counts.get('new_individual', 0)}")
    
    # 打印样本预测
    print("\n样本预测:")
    print(submission.head(10))
    
    return submission

if __name__ == '__main__':
    # 添加异常处理，便于调试
    try:
        predict_all_query_images()
    except Exception as e:
        print(f"运行中出现错误: {e}")
        import traceback
        traceback.print_exc() 