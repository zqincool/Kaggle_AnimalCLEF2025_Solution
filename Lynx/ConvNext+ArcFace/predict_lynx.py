import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from wildlife_datasets.datasets import AnimalCLEF2025
# 导入必要的类
from LynxClassifier import LynxDataset, ConvNextArcFace

def predict_lynx_query_images():
    """
    针对猞猁(Lynx)图像的预测函数，使用训练好的ConvNeXt+ArcFace模型
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
    db_mask = (dataset.metadata['dataset'] == 'LynxID2025') & (dataset.metadata['split'] == 'database')
    db_indices = np.where(db_mask)[0].tolist()
    db_subset = torch.utils.data.Subset(dataset, db_indices)
    
    # 创建Lynx数据集以获取id2idx映射
    temp_dataset = LynxDataset(db_subset, transform=None)
    id2idx = temp_dataset.id2idx
    idx2id = {idx: label for label, idx in id2idx.items()}
    
    # 选择查询子集（仅限Lynx）
    query_mask = (dataset.metadata['dataset'] == 'LynxID2025') & (dataset.metadata['split'] == 'query')
    query_indices = np.where(query_mask)[0].tolist()
    query_subset = torch.utils.data.Subset(dataset, query_indices)
    
    # 类别数量
    num_classes = len(id2idx)
    print(f"类别数量: {num_classes}")
    
    # 创建模型
    print("使用ConvNeXt + ArcFace模型进行预测")
    model = ConvNextArcFace('convnext_base', num_classes).to(device)
    
    # 加载训练模型
    model_path = 'best_convnext_arcface.pth'
    try:
        model.load_state_dict(torch.load(model_path))
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
    
    # 设置置信度阈值
    conf_threshold = 0.3  # 与训练时保持一致
    
    # 进行预测
    predictions = []
    
    print(f"为{len(query_subset)}张图像生成预测结果...")
    for i, (image, _) in enumerate(tqdm(query_subset)):
        if test_transform:
            image = test_transform(image)
        
        # 添加批次维度
        image = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            # 获取模型预测
            logits, _ = model(image)
            probs = torch.softmax(logits, dim=1)
            max_prob, predicted_idx = torch.max(probs, dim=1)
            
            # 如果低于置信度阈值，分类为new_individual
            if max_prob.item() < conf_threshold:
                predictions.append("new_individual")
            else:
                # 将索引转换为标签
                label = idx2id[predicted_idx.item()]
                predictions.append(label)
    
    # 创建提交数据帧
    submission = pd.DataFrame({
        'image_id': image_ids,
        'identity': predictions
    })
    
    # 保存提交文件
    output_file = 'lynx_submission.csv'
    submission.to_csv(output_file, index=False)
    print(f"提交文件已保存到: {output_file}")
    
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
        predict_lynx_query_images()
    except Exception as e:
        print(f"运行中出现错误: {e}")
        import traceback
        traceback.print_exc() 