import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

image = torch.zeros((1, 3, 600, 800)).float()
image_size = (600, 800)
labels = torch.LongTensor([6, 8])


backbone = torchvision.models.vgg16(pretrained=True)
feat = list(backbone.features)
req_features = feat[:30]
image_feature = nn.Sequential(*req_features)

backbone_out =
