import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class HP_VGG16(nn.Module):

    def __init__(self):
        super(HP_VGG16, self).__init__()
        trunk_net = torchvision.models.vgg16_bn(pretrained=True)
        for p in trunk_net.parameters():
            p.requires_grad = True

        self.trunk_net = trunk_net

        self.bn1 = nn.BatchNorm1d(1000)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1000, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, input):
        x = self.trunk_net(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x


class HP_RES50(nn.Module):

    def __init__(self):
        super(HP_RES50, self).__init__()
        trunk_net = torchvision.models.resnet50(pretrained=True)
        for p in trunk_net.parameters():
            p.requires_grad = True

        self.trunk_net = trunk_net
        self.bn1 = nn.BatchNorm1d(1000)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, input):
        x = self.trunk_net(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x


def hp_net_vgg16():
    model = HP_VGG16()
    return model


def hp_net_res50():
    model = HP_RES50()
    return model
