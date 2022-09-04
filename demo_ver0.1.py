import numpy as np
import torch
import os
import pathlib
import pandas as pd
import json
import random

from torch.utils.data.dataset import Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn

from model.densecap import densecap_resnet50_fpn
from data_loader import DenseCapDataset, DataLoaderPFG
from evaluate import  quantity_check

from apex import amp


def seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(False)
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# backbone
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

backbone = resnet_fpn_backbone('resnet50', pretrained=True)

# RPN

return_features = False
# Caption parameters
box_describer = None
feat_size = None
hidden_size = None
max_len = None
emb_size = None
rnn_num_layers = None
vocab_size = None
fusion_type = 'init_inject'

# transform parameters
min_size = 300
max_size = 720
image_mean = None
image_std = None

# RPN parameters
rpn_anchor_generator = None
rpn_head = None
rpn_pre_nms_top_n_train = 2000
rpn_pre_nms_top_n_test = 1000
rpn_post_nms_top_n_train = 2000
rpn_post_nms_top_n_test = 1000
rpn_nms_thresh = 0.7
rpn_fg_iou_thresh = 0.7
rpn_bg_iou_thresh = 0.3
rpn_batch_size_per_image = 256
rpn_positive_fraction = 0.5

# Box parameters
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


out_channels = backbone.out_channels

# define anchor_generator
from torchvision.models.detection.rpn import AnchorGenerator

anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
rpn_anchor_generator_ = AnchorGenerator(anchor_sizes, aspect_ratios)


from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork

rpn_head_ = RPNHead(out_channels, rpn_anchor_generator_.num_anchors_per_location()[0])

rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train,
                         testing=rpn_pre_nms_top_n_test)
rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train,
                          testing=rpn_post_nms_top_n_test)

rpn = RegionProposalNetwork(
    rpn_anchor_generator_, rpn_head_,
    rpn_fg_iou_thresh, rpn_bg_iou_thresh,
    rpn_batch_size_per_image, rpn_positive_fraction,
    rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)


from torchvision.ops import MultiScaleRoIAlign
box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=7,
                sampling_ratio=2)


import torch.nn as nn
import torch.nn.functional as F

class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models
    Arguments:
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
representation_size = 4096
box_head = TwoMLPHead(
    out_channels * resolution ** 2,
    representation_size)


class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.
    Arguments:
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

representation_size = 4096
box_predictor = FastRCNNPredictor(
    representation_size, 2)



from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LanguageModel(nn.Module):

    def __init__(self, feat_size, hidden_size, max_len, emb_size, rnn_num_layers, vocab_size,
                 fusion_type='init_inject'):
        super(LanguageModel, self).__init__()

        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.emb_size = emb_size
        self.rnn_num_layers = rnn_num_layers
        self.vocab_size = vocab_size
        self.fusion_type = fusion_type
        self.special_idx = {
            '<pad>': 0,
            '<bos>': 1,
            '<eos>': 2
        }

        self.embedding_layer = nn.Embedding(vocab_size, emb_size)

        self.lstm = nn.LSTM(input_size=emb_size, hidden_size=hidden_size, num_layers=rnn_num_layers, batch_first=True)

        self.feature_project_layer = nn.Sequential(
            nn.Linear(feat_size, emb_size),
            nn.ReLU()
        )

    def init_hidden(self, batch_size, device):

        h0 = torch.zeros(self.rnn_num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.rnn_num_layers, batch_size, self.hidden_size).to(device)

        return h0, c0


    def forward(self, feat, cap_gt, cap_lens):
        """
        :param feat: (batch_size, feat_size)
        :param cap_gt: (batch_size, max_len)
        :param cap_lens: (batch_size,)
        :return: predicts (batch_size, max_len, vocab_size)
        """
        batch_size = feat.shape[0]
        device = feat.device

        word_emb = self.embedding_layer(cap_gt)  # (batch_size, max_len, embed_size)
        feat_emb = self.feature_project_layer(feat)  # (batch_size, embed_size)

        h0, c0 = self.init_hidden(batch_size, device)

        packed = pack_padded_sequence(word_emb,
                                             lengths=cap_lens.to('cpu'),
                                             batch_first=True,
                                             enforce_sorted=False)

        _, (h0, c0) = self.lstm(feat_emb.unsqueeze(1), (h0, c0))  # first input projected feat emb to rnn

        lstm_outputs, _ = self.lstm(packed, (h0, c0))

        outputs, _ = pad_packed_sequence(lstm_outputs, batch_first=True,
                                            total_length=self.max_len)
        predicts = self.fc_layer(outputs)

        return predicts


    def sample(self, feat):

        batch_size = feat.shape[0]
        device = feat.device

        feat_emb = self.feature_project_layer(feat)

        predicts = torch.ones(batch_size, self.max_len+1, dtype=torch.long).to(device) * self.special_idx['<pad>']
        predicts[:, 0] = torch.ones(batch_size, dtype=torch.long).to(device) * self.special_idx['<bos>']
        keep = torch.arange(batch_size,)

        h, c = self.init_hidden(batch_size, device)

        _, (h, c) = self.lstm(feat_emb.unsqueeze(1),
                             (h, c))  # first input projected feat emb to rnn

        for i in range(self.max_len):
            word_emb = self.embedding_layer(predicts[keep, i])  # (valid_batch_size, embed_size)

            _, (h, c) = self.lstm(word_emb.unsqueeze(1), (h, c))  # (num_layers, valid_batch_size, hidden_size)

            rnn_output = h[-1]

            pred = self.fc_layer(rnn_output)  # (valid_batch_size, vocab_size)
            predicts[keep, i + 1] = pred.log_softmax(dim=-1).argmax(dim=-1)

            non_stop = predicts[keep, i+1] != self.special_idx['<eos>']
            keep = keep[non_stop]  # update unfinished indices

            if keep.nelement() == 0:  # stop if all finished
                break
            else:
                h = h[:, non_stop, :]
                c = c[:, non_stop, :]

        return predicts





class BoxDescriber(nn.Module):

    def __init__(self, feat_size, hidden_size, max_len, emb_size, rnn_num_layers, vocab_size,
                 fusion_type='init_inject', pad_idx=0, start_idx=1, end_idx=2):

        assert fusion_type in {'init_inject', 'merge'}, "only init_inject and merge is supported"

        super(BoxDescriber, self).__init__()

        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.emb_size = emb_size
        self.rnn_num_layers = rnn_num_layers
        self.vocab_size = vocab_size
        self.fusion_type = fusion_type
        self.special_idx = {
            '<pad>':pad_idx,
            '<bos>':start_idx,
            '<eos>':end_idx
        }

        self.embedding_layer = nn.Embedding(vocab_size, emb_size)

        self.rnn = nn.LSTM(input_size=emb_size, hidden_size=hidden_size, num_layers=rnn_num_layers, batch_first=True)

        self.feature_project_layer = nn.Sequential(
            nn.Linear(feat_size, emb_size),
            nn.ReLU()
        )

        self.fc_layer = nn.Linear(hidden_size, vocab_size)

    def init_hidden(self, batch_size, device):

        h0 = torch.zeros(self.rnn_num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.rnn_num_layers, batch_size, self.hidden_size).to(device)

        return h0, c0

    def forward_train(self, feat, cap_gt, cap_lens):
        """
        :param feat: (batch_size, feat_size)
        :param cap_gt: (batch_size, max_len)
        :param cap_lens: (batch_size,)
        :return: predicts (batch_size, max_len, vocab_size)
        """

        batch_size = feat.shape[0]
        device = feat.device

        word_emb = self.embedding_layer(cap_gt)  # (batch_size, max_len, embed_size)
        feat_emb = self.feature_project_layer(feat)  # (batch_size, embed_size)

        h0, c0 = self.init_hidden(batch_size, device)
        if self.fusion_type == 'init_inject':
            _, (h0, c0) = self.rnn(feat_emb.unsqueeze(1), (h0, c0))  # first input projected feat emb to rnn

        rnn_input_pps = pack_padded_sequence(word_emb, lengths=cap_lens.to('cpu'), batch_first=True, enforce_sorted=False)

        rnn_output_pps, _ = self.rnn(rnn_input_pps, (h0, c0))

        rnn_output, _ = pad_packed_sequence(rnn_output_pps, batch_first=True, total_length=self.max_len)

        if self.fusion_type == 'merge':
            feat_emb = feat_emb[:, None, :].expand(batch_size, self.max_len, self.emb_size)
            rnn_output = torch.cat([rnn_output, feat_emb], dim=-1)  # (batch_size, max_len, hidden_size + emb_size)

        predicts = self.fc_layer(rnn_output)

        return predicts

    def forward_test(self, feat):
        """Greedy inference for the sake of speed
        :param feat: (batch_size, feat_size)
        :return: predicts (batch_size, max_len)
        """
        batch_size = feat.shape[0]
        device = feat.device

        feat_emb = self.feature_project_layer(feat)  # (batch_size, embed_size)

        predicts = torch.ones(batch_size, self.max_len+1, dtype=torch.long).to(device) * self.special_idx['<pad>']
        predicts[:, 0] = torch.ones(batch_size, dtype=torch.long).to(device) * self.special_idx['<bos>']
        keep = torch.arange(batch_size,)  # keep track of unfinished sequences

        h, c = self.init_hidden(batch_size, device)
        if self.fusion_type == 'init_inject':
            _, (h, c) = self.rnn(feat_emb.unsqueeze(1), (h, c))  # first input projected feat emb to rnn

        for i in range(self.max_len):
            word_emb = self.embedding_layer(predicts[keep, i])  # (valid_batch_size, embed_size)

            _, (h, c) = self.rnn(word_emb.unsqueeze(1), (h, c))  # (num_layers, valid_batch_size, hidden_size)

            if self.fusion_type == 'init_inject':
                rnn_output = h[-1]
            else: # merge
                rnn_output = torch.cat([h[-1], feat_emb[keep]], dim=-1)  # (valid_batch_size, hidden_size + emb_size)

            pred = self.fc_layer(rnn_output)  # (valid_batch_size, vocab_size)

            predicts[keep, i+1] = pred.log_softmax(dim=-1).argmax(dim=-1)

            non_stop = predicts[keep, i+1] != self.special_idx['<eos>']
            keep = keep[non_stop]  # update unfinished indices
            if keep.nelement() == 0:  # stop if all finished
                break
            else:
                h = h[:, non_stop, :]
                c = c[:, non_stop, :]

        return predicts

    def forward(self, feat, cap_gt=None, cap_lens=None):

        if isinstance(cap_gt, list) and isinstance(cap_lens, list):
            cap_gt = torch.cat(cap_gt, dim=0)
            cap_lens = torch.cat(cap_lens, dim=0)
            assert feat.shape[0] == cap_gt.shape[0] and feat.shape[0] == cap_lens.shape[0]

        if self.training:
            assert cap_gt is not None and cap_lens is not None, "cap_gt and cap_lens should not be None during training"
            cap_gt = cap_gt[:, :-1]  # '<eos>' does not include in input
            cap_lens = torch.clamp(cap_lens - 1, min=0)
            return self.forward_train(feat, cap_gt, cap_lens)
        else:
            return self.forward_test(feat)


representation_size = 4096 if feat_size is None else feat_size
box_describer = BoxDescriber(representation_size, hidden_size, max_len,
                             emb_size, rnn_num_layers, vocab_size, fusion_type)



from torchvision.ops import boxes as box_ops
from torchvision.models.detection import _utils as det_utils


def detect_loss(class_logits, box_regression, labels, regression_targets):
    """
    Computes the loss for detection part.
    Arguments:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)
    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, -1, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss



def caption_loss(caption_predicts, caption_gt, caption_length):
    """
    Computes the loss for caption part.
    Arguments:
        caption_predicts (Tensor)
        caption_gt (Tensor or list[Tensor])
        caption_length (Tensor or list[Tensor])
        caption_loss (Tensor)
    """

    if isinstance(caption_gt, list) and isinstance(caption_length, list):
        caption_gt = torch.cat(caption_gt, dim=0)  # (batch_size, max_len+1)
        caption_length = torch.cat(caption_length, dim=0) # (batch_size, )
        assert caption_predicts.shape[0] == caption_gt.shape[0] and caption_predicts.shape[0] == caption_length.shape[0]

    # '<bos>' is not considered
    caption_length = torch.clamp(caption_length-1, min=0)

    predict_pps = pack_padded_sequence(caption_predicts, caption_length.to('cpu'), batch_first=True, enforce_sorted=False)

    target_pps = pack_padded_sequence(caption_gt[:, 1:], caption_length.to('cpu'), batch_first=True, enforce_sorted=False)

    return F.cross_entropy(predict_pps.data, target_pps.data)


class DenseCapRoIHeads(nn.Module):

    def __init__(self,
                 box_describer,
                 box_roi_pool,
                 box_head,
                 box_predictor,
                 # Faster R-CNN training
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 bbox_reg_weights,
                 # Faster R-CNN inference
                 score_thresh,
                 nms_thresh,
                 detections_per_img,
                 # Whether return features during testing
                 return_features=False,
                 ):

        super(DenseCapRoIHeads, self).__init__()

        self.return_features = return_features
        self.box_similarity = box_ops.box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=False)

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image,
            positive_fraction)

        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor
        self.box_describer = box_describer

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):

        matched_idxs = []
        labels = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):  # 每张图片循环

            if gt_boxes_in_image.numel() == 0:
                # Background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
            else:

                device = proposals_in_image.device
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                # iou (Tensor[N, M]): the NxM matrix containing the IoU values for every element in boxes1 and boxes2

                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]

                # Label background (below the low threshold)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = torch.tensor(0, device = device)

                # Label ignore proposals (between low and high thresholds)
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = torch.tensor(-1, device=device)  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def subsample(self, labels):

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
                zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def select_training_samples(self, proposals, targets):
        """
        proposals: (List[Tensor[N, 4]])
        targets (List[Dict])
        """
        assert targets is not None
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_captions = [t["caps"] for t in targets]
        gt_captions_length = [t["caps_len"] for t in targets]
        gt_labels = [torch.ones((t["boxes"].shape[0],), dtype=torch.int64, device=device) for t in
                     targets]  # generate labels LongTensor(1)

        # append ground-truth bboxes to propos
        # List[2*N,4],一个list是一张图片
        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]  # (M,) 0~P-1
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]  # before (P,) / after (M,) 0~N-1

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])
            gt_captions[img_id] = gt_captions[img_id][matched_idxs[img_id]]  # before (N, ) / after (M, )
            gt_captions_length[img_id] = gt_captions_length[img_id][matched_idxs[img_id]]

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)

        return proposals, matched_idxs, gt_captions, gt_captions_length, labels, regression_targets

    def postprocess_detections(self, logits, box_regression, caption_predicts, proposals, image_shapes,
                               box_features, return_features):
        device = logits.device
        num_classes = logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)
        pred_caption_list = caption_predicts.split(boxes_per_image, 0)
        if return_features:
            pred_box_features_list = box_features.split(boxes_per_image, 0)
        else:
            pred_box_features_list = None

        all_boxes = []
        all_scores = []
        all_labels = []
        all_captions = []
        all_box_features = []
        remove_inds_list = []
        keep_list = []
        for boxes, scores, captions, image_shape in zip(pred_boxes_list, pred_scores_list, pred_caption_list,
                                                        image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            remove_inds_list.append(inds)
            boxes, scores, captions, labels = boxes[inds], scores[inds], captions[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, captions, labels = boxes[keep], scores[keep], captions[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            keep_list.append(keep)
            boxes, scores, captions, labels = boxes[keep], scores[keep], captions[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_captions.append(captions)
            all_labels.append(labels)

        if return_features:
            for inds, keep, box_features in zip(remove_inds_list, keep_list, pred_box_features_list):
                all_box_features.append(box_features[inds[keep]//(num_classes-1)])

        return all_boxes, all_scores, all_captions, all_box_features

    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """

        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
                assert t["caps"].dtype == torch.int64, 'target caps must of int64 (torch.long) type'
                assert t["caps_len"].dtype == torch.int64, 'target caps_len must of int64 (torch.long) type'

        if self.training:
            proposals, matched_idxs, caption_gt, caption_length, labels, regression_targets = \
                self.select_training_samples(proposals, targets)
        else:
            labels = None
            matched_idxs = None
            caption_gt = None
            caption_length = None
            regression_targets = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        logits, box_regression = self.box_predictor(box_features)

        if self.training:
            # labels 到这里应该是有0和（1，class-1），0代表背景，其余代表类别，需要剔除背景，然后进行描述(List[Tensor])
            # 也需要滤除对应的caption和caption_length
            keep_ids = [label>0 for label in labels]
            boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
            box_features = box_features.split(boxes_per_image, 0)
            box_features_gt = []
            for i in range(len(keep_ids)):
                box_features_gt.append(box_features[i][keep_ids[i]])
                caption_gt[i] = caption_gt[i][keep_ids[i]]
                caption_length[i] = caption_length[i][keep_ids[i]]
            box_features = torch.cat(box_features_gt, 0)

        caption_predicts = self.box_describer(box_features, caption_gt, caption_length)

        result, losses = [], {}
        if self.training:
            loss_classifier, loss_box_reg = detect_loss(logits, box_regression, labels, regression_targets)
            loss_caption = caption_loss(caption_predicts, caption_gt, caption_length)

            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg,
                "loss_caption": loss_caption
            }
        else:
            boxes, scores, caption_predicts, feats = self.postprocess_detections(logits, box_regression,
                                                                                 caption_predicts, proposals,
                                                                                 image_shapes, box_features,
                                                                                 self.return_features)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "caps": caption_predicts[i],
                        "scores": scores[i],
                    }
                )
                if self.return_features:
                    result[-1]['feats'] = feats[i]

        return result, losses




roi_heads = DenseCapRoIHeads(
    # Caption
    box_describer,
    # Box
    box_roi_pool, box_head, box_predictor,
    box_fg_iou_thresh, box_bg_iou_thresh,
    box_batch_size_per_image, box_positive_fraction,
    bbox_reg_weights,
    box_score_thresh, box_nms_thresh, box_detections_per_img,
    # Whether return features during testing
    return_features)


from torchvision.models.detection.transform import GeneralizedRCNNTransform

if image_mean is None:
    image_mean = [0.485, 0.456, 0.406]
if image_std is None:
    image_std = [0.229, 0.224, 0.225]
transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)



