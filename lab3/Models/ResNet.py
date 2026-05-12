import torch
import torch.nn as nn
import torchvision.models as models


class ResNetCNN(nn.Module):
    def __init__(self, freeze_backbone = True):
        super(ResNetCNN, self).__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        for param in resnet.parameters():
            param.requires_grad = not freeze_backbone

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)


    def forward(self, images):
        features = self.resnet(images)
        features = features.reshape(features.size(0), -1)

        return features