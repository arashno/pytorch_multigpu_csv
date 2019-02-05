import os

import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

class model(nn.Module):
    def __init__(self, architecture, num_classes, use_pretrained = False):
        super(model, self).__init__()
        self.inner_model = models.__dict__[architecture](pretrained = use_pretrained)
        self.head = None
        if architecture.startswith('resnet'):
            in_feats = self.inner_model.fc.in_features
            self.inner_model.fc = nn.Linear(in_feats, num_classes)
            self.head = self.inner_model.fc
        elif architecture.startswith('inception'):
            in_feats = self.inner_model.fc.in_features
            self.inner_model.fc = nn.Linear(in_feats, num_classes)
            self.head = self.inner_model.fc
        if architecture.startswith('densenet'):
            in_feats = self.inner_model.classifier.in_features
            self.inner_model.classifier = nn.Linear(in_feats, num_classes)
            self.head = self.inner_model.classifier
        if architecture.startswith('vgg'):
            in_feats = self.inner_model.classifier._modules['6'].in_features
            self.inner_model.classifier._modules['6'] = nn.Linear(in_feats, num_classes)
            self.head = self.inner_model.classifier._modules['6']
        if architecture.startswith('alexnet'):
            in_feats = self.inner_model.classifier._modules['6'].in_features
            self.inner_model.classifier._modules['6'] = nn.Linear(in_feats, num_classes)
            self.head = self.inner_model.classifier._modules['6']

    def forward(self, x):
        return self.inner_model.forward(x)

    def freeze(self):
        for child in self.inner_model.children():
            for param in child.parameters():
                param.requires_grad = False
        for param in self.head.parameters():
            param.requires_grad = True

    def unfreeze(self):
        for child in self.inner_model.children():
            for param in child.parameters():
                param.requires_grad = True
