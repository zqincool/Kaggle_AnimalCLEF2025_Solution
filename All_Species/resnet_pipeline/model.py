import torchvision.models as models
import torch.nn as nn

def load_model(name='resnet50', num_classes=1000, pretrained=True, device='cpu', dropout_p=0.5):
    # Compatible with torchvision 0.13+, use weights argument
    if name == 'resnet50':
        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
        else:
            weights = None
        model = models.resnet50(weights=weights)
    elif name == 'resnet101':
        if pretrained:
            weights = models.ResNet101_Weights.DEFAULT
        else:
            weights = None
        model = models.resnet101(weights=weights)
    else:
        raise ValueError('Unsupported model name: {}'.format(name))
    # Add Dropout before the final fully connected layer
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(model.fc.in_features, num_classes)
    )
    model = model.to(device)
    return model 