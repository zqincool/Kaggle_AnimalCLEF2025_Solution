
from backbones.swin_transformer import SwinTransformer
import torch.nn as nn

def build_model(cfg):
    model = SwinBackbone(cfg)
    return model

class SwinBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = SwinTransformer(embed_dim=128,
                                        depths=[2, 2, 18, 2],
                                        num_heads=[4, 8, 16, 32],
                                        window_size=7,
                                        drop_path_rate=0.3)
        
    def forward(self, x):
        feat = self.backbone(x)  # [B, 512]
        return {'cls_feat': feat}
