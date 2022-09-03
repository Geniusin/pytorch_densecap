import pathlib
import json
import h5py
import string
import numpy as np
import queue
from threading import Thread, Lock
import os
import re
from collections import Counter
import pickle
from typing import Tuple, List

def encode_splits(data, split_data):
    """ Encode splits by mappings """
    id_to_split = {}
    for split, idxs in split_data.items():
        for idx in idxs:
            id_to_split[idx] = split

    split_dict = {k:list() for k in split_data.keys()}

    for i, img in enumerate(data):
        split_dict[id_to_split[img['id']]].append(i)

    return split_dict


def filter_images(data, split_data):
    """ Keep only images that are in some split and have some captions """
    all_split_ids = set()
    for split_name, ids in split_data.items():
        all_split_ids.update(ids)
    new_data = []
    for img in data:
        keep = img['id'] in all_split_ids and len(img['regions']) > 0
        if keep:
            new_data.append(img)
    return new_data


# Vocab 체크 필요
def build_vocab(data, min_token_instances, verbose=True):
    """ Builds a set that contains the vocab. Filters infrequent tokens. """
    token_counter = Counter()
    for img in data:
        for region in img['regions']:
            if region['tokens'] is not None:
                token_counter.update(region['tokens'])
    vocab = set()
    for token, count in token_counter.items():
        if count >= min_token_instances:
            vocab.add(token)

    if verbose:
        print('Keeping {} / {} tokens with enough instances'.format(len(vocab), len(token_counter)))

    vocab = list(vocab)
    vocab = sorted(vocab, key=lambda token: token_counter[token], reverse=True)
    if len(vocab) < len(token_counter):
        vocab = ['<pad>', '<bos>', '<eos>', '<unk>'] + vocab
        if verbose:
            print('adding special <pad> <bos> <eos> <unk> token.')
    else:
        vocab = ['<pad>', '<bos>', '<eos>'] + vocab
        if verbose:
            print('adding special <pad> <bos> <eos> token.')

    return vocab


def build_vocab_dict(vocab):
    token_to_idx, idx_to_token = {}, {}
    next_idx = 0  # 0-indexed

    for token in vocab:
        token_to_idx[token] = next_idx
        idx_to_token[next_idx] = token
        next_idx = next_idx + 1

    return token_to_idx, idx_to_token


def words_preprocess(phrase):
    """ preprocess a sentence: lowercase, clean up weird chars, remove punctuation """
    translator = str.maketrans('', '', string.punctuation)
    replacements = {
        u'½': u'half',
        u'—': u'-',
        u'™': u'',
        u'¢': u'cent',
        u'ç': u'c',
        u'û': u'u',
        u'é': u'e',
        u'°': u' degree',
        u'è': u'e',
        u'…': u'',
    }

    for k, v in replacements.items():
        phrase = phrase.replace(k, v)
    return str(phrase).lower().translate(translator).split()


def split_filter_captions(data, max_token_length, tokens_type, verbose=True):
    """
    Modifies data in-place by adding a 'tokens' field to each region.
    If the region's label is too long, 'tokens' will be None; otherwise
    it will be a list of strings.
    Splits by space when tokens_type = "words", or lists all chars when "chars"
    """
    captions_kept = 0
    captions_removed = 0
    for i, img in enumerate(data):
        if verbose and (i + 1) % 2000 == 0:
            print('Splitting tokens in image {} / {}'.format(i + 1, len(data)))
        img_kept, img_removed = 0, 0
        for region in img['regions']:

            # create tokens array
            if tokens_type == 'words':
                tokens = words_preprocess(region['phrase'])  # 此处分词
            elif tokens_type == 'chars':
                tokens = list(region['label'])
            else:
                assert False, 'tokens_type must be "words" or "chars"'

            # filter by length
            if max_token_length > 0 and len(tokens) <= max_token_length:
                region['tokens'] = tokens
                captions_kept += 1
                img_kept += 1
            else:
                region['tokens'] = None
                captions_removed += 1
                img_removed += 1

        if img_kept == 0:
            print('kept {}, removed {}'.format(img_kept, img_removed))
            assert False, 'DANGER, some image has no valid regions. Not super sure this doesnt cause bugs. Think about more if it comes up'

    if verbose:
        print('Keeping {} captions'.format(captions_kept))
        print('Skipped {} captions for being too long'.format(captions_removed))


def encode_caption(tokens, token_to_idx, max_token_length):
    encoded = np.ones(max_token_length+2, dtype=np.int64) * token_to_idx['<pad>']
    encoded[0] = token_to_idx['<bos>']
    encoded[len(tokens)+1] = token_to_idx['<eos>']

    for i, token in enumerate(tokens):

        if token in token_to_idx:
            encoded[i+1] = token_to_idx[token]
        else:
            encoded[i+1] = token_to_idx['<unk>']

    return encoded


def encode_captions(data, token_to_idx, max_token_length):
    encoded_list = []
    lengths = []
    for img in data:
        for region in img['regions']:
            tokens = region['tokens']
            if tokens is None: continue
            tokens_encoded = encode_caption(tokens, token_to_idx, max_token_length)
            encoded_list.append(tokens_encoded)
            lengths.append(len(tokens)+2)
    return np.vstack(encoded_list), np.asarray(lengths, dtype=np.int64)  # in pytorch np.int64 is torch.long



def encode_boxes(data, image_data, all_image_ids):
    all_boxes = []
    for i, img in enumerate(data):

        img_info = image_data[all_image_ids.index(img['id'])]
        assert img['id'] == img_info['id'], 'id mismatch'

        for region in img['regions']:
            if region['tokens'] is None:
                continue

            x1, y1 = int(region['x']), int(region['y'])
            if x1 < 1: x1 = 1
            if y1 < 1: y1 = 1
            x2, y2 = x1 + int(region['width']), y1 + int(region['height'])


            # sanity check
            try:
                assert x1 <= img_info['width'], 'invalid x1 coordinate {} > {} in image_id:{} box_id:{}'.format(x1, img_info['width'], img['id'], region['id'])
                assert y1 <= img_info['height'], 'invalid y1 coordinate {} > {} in image_id:{} box_id:{}'.format(y1, img_info['height'], img['id'], region['id'])
                assert x2 <= img_info['width'], 'invalid x2 coordinate {} > {} in image_id:{} box_id:{}'.format(x2, img_info['width'], img['id'], region['id'])
                assert y2 <= img_info['height'], 'invalid y2 coordinate {} > {} in image_id:{} box_id:{}'.format(y2, img_info['height'], img['id'], region['id'])
            except AssertionError as e:
                print(e)

                print('orignal bbox coordinate ', (x1, y1, x2, y2))
                # clamp to image


                if x1 > img_info['width']: x1 = (img_info['width'] - 1) - region['width']
                if y1 > img_info['height']: y1 = (img_info['height'] - 1) - region['height']
                if x2 > img_info['width']: x2 = img_info['width'] - 1
                if y2 > img_info['height']: y2 = img_info['height'] - 1
                print('clamped bbox coordinate ', (x1, y1, x2, y2))

                # if x1 >= x2: x1 = int(x2 - 2)
                # print(f'changed bbox : {(x1, y1, x2, y2)}'), print(type(x1),
                #                                                    type(y1),
                #                                                    type(x2),
                #                                                    type(y2))
                # if y1 >= y2: y1 = int(y2 - 2)
                # print(f'changed bbox : {(x1, y1, x2, y2)}'), print(type(x1),
                #                                                    type(y1),
                #                                                    type(x2),
                #                                                    type(y2))

            # 잘못된 bbox 확인
            box = np.asarray([x1, y1, x2, y2], dtype=np.int32)
            degenerate_boxes = box[2:] <= box[:2]
            if degenerate_boxes.any():
                if box[0] <= box[2]:
                    box[0] = box[2] - 1
                if box[1] <= box[3]:
                    box[1] = box[3] - 1
                # print the first degenerate box
                # bb_idx = np.where(degenerate_boxes.any())[0]
                # degen_bb: List[float] = box[bb_idx].tolist()
            degenerate_boxes = box[2:] <= box[:2]
            if degenerate_boxes.any():
                print(box)

            all_boxes.append(box)
    return np.vstack(all_boxes)


def build_img_idx_to_box_idxs(data):
    img_idx = 0
    box_idx = 0
    num_images = len(data)
    img_to_first_box = np.zeros(num_images, dtype=np.int32)
    img_to_last_box = np.zeros(num_images, dtype=np.int32)
    for img in data:
        img_to_first_box[img_idx] = box_idx
        for region in img['regions']:
            if region['tokens'] is None: continue
            box_idx += 1
        img_to_last_box[img_idx] = box_idx - 1  # -1 to make these inclusive limits 闭集
        img_idx += 1

    return img_to_first_box, img_to_last_box


def build_filename_dict(data):
    # First make sure all filenames
    filenames_list = ['%d.jpg' % img['id'] for img in data]
    assert len(filenames_list) == len(set(filenames_list))

    next_idx = 0
    filename_to_idx, idx_to_filename = {}, {}
    for img in data:
        filename = '%d.jpg' % img['id']
        filename_to_idx[filename] = next_idx
        idx_to_filename[next_idx] = filename
        next_idx += 1
    return filename_to_idx, idx_to_filename


def build_directory_dict(data, image_data, all_image_ids):

    idx_to_directory = dict()

    next_idx = 0
    for img in data:

        img_info = image_data[all_image_ids.index(img['id'])]
        assert img['id'] == img_info['id'], 'id mismatch'

        # idx_to_directory[next_idx] = re.search('(VG.*)/(.*.jpg)$', img_info['url']).group(1)
        idx_to_directory[next_idx] = re.search('(VG.*)', img_info['url']).group(1).split('/')[0]
        next_idx += 1

    return idx_to_directory


def encode_filenames(data, filename_to_idx):
    filename_idxs = []
    for img in data:
        filename = '%d.jpg' % img['id']
        idx = filename_to_idx[filename]
        for region in img['regions']:
            if region['tokens'] is None: continue
            filename_idxs.append(idx)
    return np.asarray(filename_idxs, dtype=np.int32)



path_root = pathlib.Path("C:/datasets/visual_genome")

path_image_data = path_root / "image_data.json"
path_region_descriptions = path_root / "region_descriptions.json"
path_split_data = path_root / 'info' / 'densecap_splits.json'

min_token_instances = 1
max_token_length = 15
tokens_type = 'words'
pickle_output = path_root / 'VG-regions-dicts-lite.pkl'
h5_output = path_root / 'VG-regions-lite.h5'

with open(path_image_data, 'r') as f:
    image_data = json.load(f)

with open(path_region_descriptions, 'r') as f:
    data = json.load(f)

with open(path_split_data, 'r') as f:
    split_data = json.load(f)

all_image_ids = [image_data[i]['id'] for i in range(len(image_data))]

print(f'There are {len(data)} images total')


data = filter_images(data, split_data)
print(f'After filtering for splits there are {len(data)} images')

split = encode_splits(data, split_data)

with h5py.File(h5_output, 'w') as f:

    split_filter_captions(data, max_token_length, tokens_type)

    vocab = build_vocab(data, min_token_instances)  # vocab is a list()
    token_to_idx, idx_to_token = build_vocab_dict(vocab)

    captions_matrix, lengths_vector = encode_captions(data, token_to_idx,
                                                      max_token_length)
    f.create_dataset('captions', data=captions_matrix)
    f.create_dataset('lengths', data=lengths_vector)

    boxes_matrix = encode_boxes(data, image_data, all_image_ids)
    f.create_dataset('boxes', data=boxes_matrix)

    img_to_first_box, img_to_last_box = build_img_idx_to_box_idxs(data)
    f.create_dataset('img_to_first_box', data=img_to_first_box)
    f.create_dataset('img_to_last_box', data=img_to_last_box)

    filename_to_idx, idx_to_filename = build_filename_dict(data)
    idx_to_directory = build_directory_dict(data, image_data, all_image_ids)
    box_to_img = encode_filenames(data, filename_to_idx)
    f.create_dataset('box_to_img', data=box_to_img)

    pickle_struct = {
    'token_to_idx': token_to_idx,
    'idx_to_token': idx_to_token,
    'filename_to_idx': filename_to_idx,
    'idx_to_filename': idx_to_filename,
    'idx_to_directory': idx_to_directory,
    'split': split,
    }

    with open(pickle_output, 'wb') as f_2:
        pickle.dump(pickle_struct, f_2)