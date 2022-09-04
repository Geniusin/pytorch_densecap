import numpy as np
import torch
import os
import pathlib
import pandas as pd
import json

from torch.utils.data.dataset import Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn

from model.densecap import densecap_resnet50_fpn
from data_loader import DenseCapDataset, DataLoaderPFG
from evaluate import  quantity_check

from apex import amp

torch.backends.cudnn.benchmark = True
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def set_args():

    args = dict()

    args['backbone_pretrained'] = True
    args['return_features'] = False

    # Caption parameters
    args['feat_size'] = 4096
    args['hidden_size'] = 512
    args['max_len'] = 16
    args['emb_size'] = 512
    args['rnn_num_layers'] = 1
    # args['vocab_size'] = 10629
    args['vocab_size'] = 60484
    args['fusion_type'] = 'init_inject'

    # Training Settings
    args['detect_loss_weight'] = 1.
    args['caption_loss_weight'] = 1.
    args['lr'] = 1e-4
    args['caption_lr'] = 1e-3
    args['weight_decay'] = 0.
    args['batch_size'] = 8
    args['use_pretrain_fasterrcnn'] = True
    args['box_detections_per_img'] = 50

    if not os.path.exists(os.path.join(CONFIG_PATH, MODEL_NAME)):
        os.mkdir(os.path.join(CONFIG_PATH, MODEL_NAME))
    with open(os.path.join(CONFIG_PATH, MODEL_NAME, 'config.json'), 'w') as f:
        json.dump(args, f, indent=2)

    return args


def save_model(model, optimizer, amp_, results_on_val, iter_counter, flag=None):

    state = {'model': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'amp': amp_.state_dict(),
             'results_on_val': results_on_val,
             'iterations': iter_counter}
    if isinstance(flag, str):
        filename = os.path.join('./result', '{}_{}.pth.tar'.format(MODEL_NAME, flag))
    else:
        filename = os.path.join('result', '{}.pth.tar'.format(MODEL_NAME))
    print('Saving checkpoint to {}'.format(filename))
    torch.save(state, filename)



args = set_args()

model = densecap_resnet50_fpn(backbone_pretrained=args['backbone_pretrained'],
                              feat_size=args['feat_size'],
                              hidden_size=args['hidden_size'],
                              max_len=args['max_len'],
                              emb_size=args['emb_size'],
                              rnn_num_layers=args['rnn_num_layers'],
                              vocab_size=args['vocab_size'],
                              fusion_type=args['fusion_type'],
                              box_detections_per_img=args['box_detections_per_img'])


if args['use_pretrain_fasterrcnn']:
    model.backbone.load_state_dict(fasterrcnn_resnet50_fpn(pretrained=True).backbone.state_dict(), strict=False)
    model.rpn.load_state_dict(fasterrcnn_resnet50_fpn(pretrained=True).rpn.state_dict(), strict=False)

model.to(device)

optimizer = torch.optim.Adam([{'params': (para for name, para in model.named_parameters()
                                if para.requires_grad and 'box_describer' not in name)},
                              {'params': (para for para in model.roi_heads.box_describer.parameters()
                                          if para.requires_grad), 'lr': args['caption_lr']}],
                              lr=args['lr'], weight_decay=args['weight_decay'])

'''
apex : pytorch에서 쉽게 분산학습과 mixed precision을 사용할수 있게 해주는 Nvidia툴
mixed precision이란 처리속도를 높이기 위해 FP16(16bit Floating Point)연산과 FP32(32bit Floating Point)를
섞어서 학습하는 방법

opt_level

O0 : 기존 FP32버전(AMP적용 x)
O1 가장 기본으로 사용하는 opt_level, Tensor Core에서 사용하는 연산은 FP16으로 계산하고,
정확한 계산이 필요한 경우 FP32로 계산하는 옵션
'''

opt_level = 'O1'

model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

model.roi_heads.box_roi_pool.forward = \
        amp.half_function(model.roi_heads.box_roi_pool.forward)

####################
# with data loader

MAX_EPOCHS = 10
USE_TB = True
CONFIG_PATH = './model_params'
MODEL_NAME = 'train_all_val_all_bz_2_epoch_10_inject_init'

IMG_DIR_ROOT = pathlib.Path("C:/datasets/visual_genome")
VG_DATA_PATH = IMG_DIR_ROOT / 'VG-regions-lite.h5'
LOOK_UP_TABLES_PATH = IMG_DIR_ROOT / 'VG-regions-dicts-lite.pkl'

MAX_TRAIN_IMAGE = -1  # if -1, use all images in train set
MAX_VAL_IMAGE = -1


train_set = DenseCapDataset(str(IMG_DIR_ROOT), str(VG_DATA_PATH), str(LOOK_UP_TABLES_PATH),
                            dataset_type='train')
val_set = DenseCapDataset(str(IMG_DIR_ROOT), str(VG_DATA_PATH), str(LOOK_UP_TABLES_PATH),
                          dataset_type='val')
idx_to_token = train_set.look_up_tables['idx_to_token']

if MAX_TRAIN_IMAGE > 0:
    train_set = Subset(train_set, range(MAX_TRAIN_IMAGE))
if MAX_VAL_IMAGE > 0:
    val_set = Subset(val_set, range(MAX_VAL_IMAGE))

train_loader = DataLoaderPFG(train_set, batch_size=args['batch_size'],
                             shuffle=True, num_workers=2,
                             pin_memory=True,
                             collate_fn=DenseCapDataset.collate_fn)

best_map = 0.

# use tensorboard to track the loss
if USE_TB:
    writer = SummaryWriter()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

history = list()
for epoch in range(MAX_EPOCHS):
    iter_counter = 0
    for batch, (img, targets, info) in enumerate(train_loader):
        start.record()

        img = [img_tensor.to(device) for img_tensor in img]
        targets = [{k: v.to(device) for k, v in target.items()} for target in
                   targets]

        model.train()
        losses = model(img, targets)

        detect_loss = losses['loss_objectness'] + losses['loss_rpn_box_reg'] + \
                      losses['loss_classifier'] + losses['loss_box_reg']
        caption_loss = losses['loss_caption']

        total_loss = args['detect_loss_weight'] * detect_loss + args[
            'caption_loss_weight'] * caption_loss

        # record loss
        if USE_TB:
            writer.add_scalar('batch_loss/total', total_loss.item(),
                              iter_counter)
            writer.add_scalar('batch_loss/detect_loss', detect_loss.item(),
                              iter_counter)
            writer.add_scalar('batch_loss/caption_loss', caption_loss.item(),
                              iter_counter)

            writer.add_scalar('details/loss_objectness',
                              losses['loss_objectness'].item(), iter_counter)
            writer.add_scalar('details/loss_rpn_box_reg',
                              losses['loss_rpn_box_reg'].item(), iter_counter)
            writer.add_scalar('details/loss_classifier',
                              losses['loss_classifier'].item(), iter_counter)
            writer.add_scalar('details/loss_box_reg',
                              losses['loss_box_reg'].item(), iter_counter)

        if iter_counter % (len(train_set) / (args['batch_size'] * 16)) == 0:
            print("[{}][{}]\ntotal_loss {:.3f}".format(epoch + 1, batch,
                                                       total_loss.item()))


        optimizer.zero_grad()
        # total_loss.backward()
        # apex backward
        with amp.scale_loss(total_loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        if iter_counter > 0 and iter_counter % 1000 == 0:
            try:
                results = quantity_check(model, val_set, idx_to_token, device,
                                         max_iter=-1, verbose=True) # quantity_check 확인 필요
                if results['map'] > best_map:
                    best_map = results['map']
                    save_model(model, optimizer, amp, results, iter_counter)

                if USE_TB:
                    writer.add_scalar('metric/map', results['map'],
                                      iter_counter)
                    writer.add_scalar('metric/det_map', results['detmap'],
                                      iter_counter)

            except AssertionError as e:
                print('[INFO]: evaluation failed at epoch {}'.format(epoch))
                print(e)

        end.record()
        torch.cuda.synchronize()

        if iter_counter % 10 == 0:
            history.append([epoch + 1, iter_counter, total_loss.item(),
                            detect_loss.item(),
                            caption_loss.item(),
                            scaled_loss.item()])

            expected_running_time = (start.elapsed_time(end) / 1000) * \
                                    (len(train_loader) - iter_counter)

            print(f'expected training end time : '
                  f'epoch: {epoch+1} | {MAX_EPOCHS} / '
                  f'{expected_running_time // 3600}h / '
                  f'{(expected_running_time % 3600) // 60}m / '
                  f'{(expected_running_time % 3600) % 60:.1f}s')

        print(f'epoch : {epoch+1} / {MAX_EPOCHS}, iter_cnt = {iter_counter} / '
              f'{len(train_loader)}, total_loss : {total_loss.item():.4f}')

        iter_counter += 1

    save_model(model, optimizer, amp, results, iter_counter, flag='end')
    torch.save(model.state_dict(), f'./result/model_220818{epoch + 1}epoch.pth')

if USE_TB:
    writer.close()


df = pd.DataFrame(history, columns=['epoch', 'iter', 'total_loss', 'detect_loss', 'caption_loss', 'scaled_loss'])
df.to_csv(f'./loss_22.08.18-1015.csv', sep=',', index=False)
