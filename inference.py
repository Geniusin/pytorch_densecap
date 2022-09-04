import torch
import os
import pathlib
import json
import matplotlib.pyplot as plt
import h5py
from PIL import Image
from matplotlib.patches import Rectangle
from tqdm import tqdm
import pickle
import numpy as np

import torchvision.transforms as transforms
from model.densecap import densecap_resnet50_fpn

def img_to_tensor(img_list):

    assert isinstance(img_list, list) and len(img_list) > 0

    img_tensors = []

    for img_path in img_list:

        img = Image.open(img_path).convert("RGB")

        img_tensors.append(transforms.ToTensor()(img))

    return img_tensors


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
    args['vocab_size'] = 60484
    args['fusion_type'] = 'init_inject'

    # Training Settings
    args['detect_loss_weight'] = 1.
    args['caption_loss_weight'] = 1.
    args['lr'] = 1e-4
    args['caption_lr'] = 1e-3
    args['weight_decay'] = 0.
    args['batch_size'] = 4
    args['use_pretrain_fasterrcnn'] = True
    args['box_detections_per_img'] = 50

    return args



##########################


def get_image_path(root):

    img_list = []
    dir_images = root / 'images'

    for folder in dir_images.glob('*'):
        for img in folder.glob('*'):
            img_list.append(str(img))

    return img_list

def load_model(model_args):

    model = densecap_resnet50_fpn(
        backbone_pretrained=model_args['backbone_pretrained'],
        return_features=True,
        feat_size=model_args['feat_size'],
        hidden_size=model_args['hidden_size'],
        max_len=model_args['max_len'],
        emb_size=model_args['emb_size'],
        rnn_num_layers=model_args['rnn_num_layers'],
        vocab_size=model_args['vocab_size'],
        fusion_type=model_args['fusion_type'],
        box_detections_per_img=100)

    checkpoint = torch.load(
        './result/train_all_val_all_bz_2_epoch_10_inject_init.pth.tar')
    model.load_state_dict(checkpoint['model'])

    print('[INFO]: correspond performance on val set:')
    for k, v in checkpoint['results_on_val'].items():
        if not isinstance(v, dict):
            print('        {}: {:.3f}'.format(k, v))

    return model

def img_to_tensor(img_list):

    assert isinstance(img_list, list) and len(img_list) > 0

    img_tensors = []

    for img_path in img_list:

        img = Image.open(img_path).convert("RGB")

        img_tensors.append(transforms.ToTensor()(img))

    return img_tensors



def describe_images(model, img_list, device, model_args):

    all_results = []

    with torch.no_grad():
        model.to(device)
        model.eval()

        for i in tqdm(range(0, len(img_list), 1)):
            image_tensors = img_to_tensor(img_list[i:i + 1])
            input_ = [t.to(device) for t in image_tensors]

            results = model(input_)

            all_results.extend(
                [{k: v.cpu() for k, v in r.items()} for r in results])

    return all_results


def visualize_result(image_file_path, result):

    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)

    assert isinstance(result, list)

    img = Image.open(image_file_path)
    plt.imshow(img)
    ax = plt.gca()
    for r in result:
        ax.add_patch(Rectangle((r['box'][0], r['box'][1]),
                               r['box'][2]-r['box'][0],
                               r['box'][3]-r['box'][1],
                               fill=False,
                               edgecolor='red',
                               linewidth=3))
        ax.text(r['box'][0], r['box'][1], r['cap'], style='italic', bbox={'facecolor':'white', 'alpha':0.7, 'pad':10})
    plt.tick_params(labelbottom='off', labelleft='off')
    plt.show()

def save_results_to_file(img_list, all_results, model_args, root):
    pkl_path = root / 'VG-regions-dicts-lite.pkl'

    with open(os.path.join(pkl_path), 'rb') as f:
        look_up_tables = pickle.load(f)

    idx_to_token = look_up_tables['idx_to_token']

    results_dict = {}

    total_box = sum(len(r['boxes']) for r in all_results)
    start_idx = 0
    img_idx = 0
    h = h5py.File(os.path.join('./result', 'box_feats.h5'), 'w')
    h.create_dataset('feats', (total_box, all_results[0]['feats'].shape[1]), dtype=np.float32)
    h.create_dataset('boxes', (total_box, 4), dtype=np.float32)
    h.create_dataset('start_idx', (len(img_list),), dtype=np.long)
    h.create_dataset('end_idx', (len(img_list),), dtype=np.long)

    for img_path, results in zip(img_list, all_results):


        print('[Result] ==== {} ====='.format(img_path))

        results_dict[img_path] = []
        for box, cap, score in zip(results['boxes'], results['caps'], results['scores']):

            r = {
                'box': [round(c, 2) for c in box.tolist()],
                'score': round(score.item(), 2),
                'cap': ' '.join(idx_to_token[idx] for idx in cap.tolist()
                                if idx_to_token[idx] not in ['<pad>', '<bos>', '<eos>'])
            }

            if r['score'] > 0.9:
                print('        SCORE {}  BOX {}'.format(r['score'], r['box']))
                print('        CAP {}\n'.format(r['cap']))

            results_dict[img_path].append(r)


        box_num = len(results['boxes'])
        h['feats'][start_idx: start_idx+box_num] = results['feats'].cpu().numpy()
        h['boxes'][start_idx: start_idx+box_num] = results['boxes'].cpu().numpy()
        h['start_idx'][img_idx] = start_idx
        h['end_idx'][img_idx] = start_idx + box_num - 1
        start_idx += box_num
        img_idx += 1

    h.close()
    # save order of img to a txt
    if len(img_list) > 1:
        with open(os.path.join('./result', 'feat_img_mappings.txt'), 'w') as f:
            for img_path in img_list:
                f.writelines(os.path.split(img_path)[1] + '\n')

    if not os.path.exists('./result'):
        os.mkdir('./result')
    with open(os.path.join('./result', 'result.json'), 'w') as f:
        json.dump(results_dict, f, indent=2)


    print('[INFO] result save to {}'.format(os.path.join('./result', 'result.json')))
    print('[INFO] feats save to {}'.format(os.path.join('./result', 'box_feats.h5')))
    print('[INFO] order save to {}'.format(os.path.join('./result', 'feat_img_mappings.txt')))


def validate_box_feat(model, all_results, device):

    with torch.no_grad():

        box_describer = model.roi_heads.box_describer
        box_describer.to(device)
        box_describer.eval()

        print('[INFO] start validating box features...')
        for results in tqdm(all_results):

            captions = box_describer(results['feats'].to(device))

            assert (captions.cpu() == results['caps']).all().item(), 'caption mismatch'


    print('[INFO] validate box feat done, no problem')


###########
model_args = set_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = pathlib.Path('C:/datasets/visual_genome')

img_list = get_image_path(root)

model = load_model(model_args)

all_results = describe_images(model, img_list, device, model_args)

save_results_to_file(img_list, all_results, model_args, root)

validate_box_feat(model, all_results, device)

###############

img_list = []
dir_images = root / 'images'

for folder in dir_images.glob('*'):
    for img in folder.glob('*'):
        img_list.append(str(img))

        break
    break

all_results = []

with torch.no_grad():
    model.to(device)
    model.eval()

    for i in tqdm(range(0, len(img_list), 1)):
        image_tensors = img_to_tensor(img_list[i:i + 1])
        input_ = [t.to(device) for t in image_tensors]

        results = model(input_)

        all_results.extend(
            [{k: v.cpu() for k, v in r.items()} for r in results])


pkl_path = root / 'VG-regions-dicts-lite.pkl'

with open(os.path.join(pkl_path), 'rb') as f:
    look_up_tables = pickle.load(f)

idx_to_token = look_up_tables['idx_to_token']

results_dict = {}
results_dict[img_list[0]] = []
for box, cap, score in zip(results[0]['boxes'], results[0]['caps'], results[0]['scores']):

    r = {
        'box': [round(c, 2) for c in box.tolist()],
        'score': round(score.item(), 2),
        'cap': ' '.join(idx_to_token[idx] for idx in cap.tolist()
                        if idx_to_token[idx] not in ['<pad>', '<bos>', '<eos>'])
    }

    if r['score'] > 0.9:
        print('        SCORE {}  BOX {}'.format(r['score'], r['box']))
        print('        CAP {}\n'.format(r['cap']))

    results_dict[img_list[0]].append(r)


fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

img = Image.open(img_list[0])
cnt = 0
plt.imshow(img)
ax = plt.gca()
for i, r in enumerate(results_dict[img_list[0]]):
    ax.add_patch(Rectangle((r['box'][0], r['box'][1]),
                           r['box'][2]-r['box'][0],
                           r['box'][3]-r['box'][1],
                           fill=False,
                           edgecolor='red'))
    ax.text(r['box'][0], r['box'][1], r['cap'], style='italic', bbox={'facecolor':'white', 'alpha':0.7, 'pad':10})

    if cnt >= 5:
        break
    cnt += 1
plt.tick_params(labelbottom='off', labelleft='off')
plt.show()