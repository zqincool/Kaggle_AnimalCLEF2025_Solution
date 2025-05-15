import os
import argparse
import torch.nn as nn     
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import pandas as pd
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from wildlife_datasets.datasets import AnimalCLEF2025
from my_dataset import MyDataSet
from model import convnext_tiny as create_model
from utils import read_split_data, create_lr_scheduler, get_params_groups, train_one_epoch, evaluate
from my_dataset import AnimalDataset, TripletDataset
from torchvision.datasets import ImageFolder
from sklearn.metrics import balanced_accuracy_score

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    # full_clef = AnimalCLEF2025(args.data_path, transform=None, load_label=True)
    # print("所有 dataset 名称：", full_clef.metadata['dataset'].unique())
    # print("各个 dataset 样本数：")
    # print(full_clef.metadata['dataset'].value_counts())
    # # 2) 筛出 SeaTurtleID2022 数据库 split
    # md   = full_clef.metadata
    # mask = (md['dataset'] == 'LynxID2025') & (md['split'] == 'database')
    # indices = mask[mask].index.tolist()
    # n_ids = md[mask]['identity'].nunique()
    # print(f"LynxID2025 在 database split 下共有 {n_ids} 个不同的 identity。")
    # clef_subset = Subset(full_clef, indices)

    # # 3) 包装成我们自定义的 AnimalDataset 并传入 train 的 transform
    # wrapped = AnimalDataset(clef_subset, transform=data_transform["train"])

    # # 4) 按 80/20 划分 train/val
    # train_size = int(0.8 * len(wrapped))
    # val_size   = len(wrapped) - train_size
    # images_root = os.path.join(args.data_path, "images")
    # full_ds = ImageFolder(root=images_root, transform=None)

    # # 自动拿类别数
    # num_classes = len(full_ds.classes)
    # print(f"Found {num_classes} classes: {full_ds.classes}")

    # # 8:2 随机拆分
    # total = len(full_ds)
    # train_size = int(0.8 * total)
    # test_size  = total - train_size
    # train_dataset, val_dataset = random_split(
    #     full_ds,
    #     [train_size, test_size],
    #     generator=torch.Generator().manual_seed(42)
    # )

    # # 分别赋予不同的 transform
    # train_dataset.dataset.transform = data_transform["train"]
    # val_dataset  .dataset.transform = data_transform["val"]
    # ────────────────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────────────────────

    # 1) 加载原始 AnimalCLEF2025
    full_clef = AnimalCLEF2025(args.data_path, transform=None, load_label=True)
    print("所有 dataset 名称：", full_clef.metadata['dataset'].unique())

    # 2) 筛“database” split 下的所有 species
    md   = full_clef.metadata
    mask = (md['dataset'] == 'LynxID2025') & (md['split'] == 'database')
    indices = mask[mask].index.tolist()
    print(f"LynxID2025 的 database split 样本数: {len(indices)}")
    subset_clef = Subset(full_clef, indices)

    subset_clef = Subset(full_clef, indices)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))


    # 3) 包装并传入 train transform
    wrapped = AnimalDataset(subset_clef, transform=data_transform["train"])
    triplet_ds = TripletDataset(wrapped)
    triplet_loader = DataLoader(
        triplet_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=True
    )
    # 4) 8:2 拆 train/val
    total      = len(wrapped)
    train_size = int(0.8 * total)
    val_size   = total - train_size
    train_dataset, val_dataset = random_split(
        wrapped,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    val_dataset.dataset.transform = data_transform["val"]

    # 5) 自动拿类别数
    num_classes = len(wrapped.label_to_idx)
    print(f"自动计算得到的总类别数: {num_classes}")

    # 4) 随机抽几个样本看下 label 是否合理
    for i in [0, len(wrapped)//2, -1]:
        img, lbl = wrapped[i]
        print(f"wrapped[{i}] = 图像张量 {tuple(img.shape)}，标签索引 {lbl}")



    # train_dataset, val_dataset = random_split(wrapped, [train_size, val_size])
    # print("Train size =", len(train_dataset), " Val size =", len(val_dataset))

    # # 5) 把 val 子集的 transform 换成验证时的那一套
    # val_dataset.dataset.transform = data_transform["val"]

    # num_classes = len(wrapped.label_to_idx)
    # print(f"自动计算得到的总类别数: {num_classes}")


    print(f'Using {nw} dataloader workers per process')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw)

    model = create_model(num_classes=num_classes).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    # pg = [p for p in model.parameters() if p.requires_grad]
    pg = get_params_groups(model, weight_decay=args.wd)
    # optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    triplet_criterion = nn.TripletMarginLoss(margin=1.0)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=1)

    best_acc = 0.
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            lr_scheduler=lr_scheduler
        )

        # validate
        # val_loss, val_acc = evaluate(model=model,
        #                              data_loader=val_loader,
        #                              device=device,
        #                              epoch=epoch)
        new_idx = wrapped.label_to_idx['new_individual']
        val_loss, val_acc = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            epoch=epoch,
            threshold=args.threshold,
            new_class_idx=new_idx
        )
        print(f"Epoch {epoch+1}/{args.epochs} Summary:")
        print(f"  Train   — loss: {train_loss:.4f}, acc: {train_acc*100:.2f}%")
        print(f"  Validate— loss: {val_loss:.4f}, acc: {val_acc*100:.2f}%")
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        if best_acc < val_acc:
            torch.save(model.state_dict(), "./weights/best_model.pth")
            best_acc = val_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=5e-2)
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='置信度阈值：max_prob < threshold 则判为 new_individual')

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="/content/drive/MyDrive/ConvNext")

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='/content/drive/MyDrive/ConvNext/convnext_tiny_1k_224_ema.pth',
                        help='initial weights path')
    # 是否冻结head以外所有权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)