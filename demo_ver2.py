import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from collections import OrderedDict
import os
import numpy as np
import pandas as pd
import pathlib

from typing import Tuple, List, Dict, Optional, Union

from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork



# make rand input
imgs = []
targets = []

for i in range(8):
    imgs.append(torch.rand(3, 500, 800)) # (C, H, W)
    trg = {}
    trg['boxes'] = torch.rand(50, 4)
    trg['caps'] = torch.rand(50, 17)
    trg['caps_len'] = torch.randint(6, 17, (50,))

    targets.append(trg)

original_image_sizes: List[Tuple[int, int]] = []
for img in imgs:
    val = img.shape[-2:]
    assert len(val) == 2
    original_image_sizes.append((val[0], val[1]))


# transform parameters
min_size=300
max_size=720
image_mean = [0.485, 0.456, 0.406]
image_std = [0.229, 0.224, 0.225]

transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

images, targets = transform(imgs, targets)




# backbone
BACKBONE_PRETRAINED = False
backbone = resnet_fpn_backbone('resnet50', BACKBONE_PRETRAINED)

features = backbone(images.tensors)
if isinstance(features, torch.Tensor):
    features = OrderedDict([('0', features)])

features = list(features.values())

grid_sizes = list([feature_map.shape[-2:] for feature_map in features])
image_size = images.tensors.shape[-2:]
dtype, device = features[0].dtype, features[0].device
strides = [
    [torch.tensor(image_size[0] // g[0], dtype=torch.int64, device=device),
     torch.tensor(image_size[1] // g[1], dtype=torch.int64, device=device)] for
    g in grid_sizes]



features = list(features.values())
#########
dummy_img = torch.zeros((1, 3, 800, 800)).float()
model = torchvision.models.vgg16(pretrained=True)
feat = list(model.features)
req_features = feat[:30]
faster_rcnn_feature = nn.Sequential(*req_features)
sample_output = faster_rcnn_feature(dummy_img)
# rpn

# Define Anchor geenerator
anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)


rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

# Define rpn_head
out_channels = backbone.out_channels
rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])


rpn_pre_nms_top_n_train = 2000
rpn_pre_nms_top_n_test = 1000
rpn_post_nms_top_n_train = 2000
rpn_post_nms_top_n_test = 1000

rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train,
                         testing=rpn_pre_nms_top_n_test)
rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train,
                          testing=rpn_post_nms_top_n_test)
rpn_fg_iou_thresh = 0.7
rpn_bg_iou_thresh = 0.3
rpn_batch_size_per_image = 256 # proportion of positive anchors in a mini-batch during trainingof the RPN
rpn_positive_fraction = 0.5
rpn_nms_thresh = 0.7



rpn = RegionProposalNetwork(
    rpn_anchor_generator, rpn_head,
    rpn_fg_iou_thresh, rpn_bg_iou_thresh,
    rpn_batch_size_per_image, rpn_positive_fraction,
    rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

proposals, proposal_losses = rpn(images, features, targets)


# roi_heads

from torchvision.models.detection.roi_heads import RoIHeads

from torchvision.ops import MultiScaleRoIAlign
box_roi_pool = MultiScaleRoIAlign(
    featmap_names=['0', '1', '2', '3'],
    output_size=7,
    sampling_ratio=2)



class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


resolution = box_roi_pool.output_size[0]
representation_size = 1024
box_head = TwoMLPHead(
    out_channels * resolution ** 2,
    representation_size)



class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


representation_size = 1024
box_predictor = FastRCNNPredictor(
    representation_size,
    2)


box_roi_pool = None
box_head = None
box_predictor = None
box_score_thresh = 0.05
box_nms_thresh = 0.5
box_detections_per_img = 100
box_fg_iou_thresh = 0.5
box_bg_iou_thresh = 0.5
box_batch_size_per_image = 512
box_positive_fraction = 0.25
bbox_reg_weights = None

