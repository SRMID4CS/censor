import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import operator
import torch

import logging
logger = logging.getLogger(__name__)

class Flickr30kCap(Dataset):
    def __init__(self, root, split='train', transform=None, tokenizer=None, max_length=40, train_all=False, modality='img+txt', name='Flickr30k'):
        self.root = root
        self.split = split
        self.transform = transform
        self.modality = modality
        self.name = name

        if train_all:
            split = 'train_all'

        anno = pd.read_csv(os.path.join(root, f'{split}.csv'), delimiter='|')

        self.images = anno['image_name']
        self.captions = [str(i) for i in anno[' comment'].values]

        self.tokenizer = tokenizer
        self.max_length = max_length

        self.n_images = len(set(self.images))
        self.iid_to_cls = {}
        
        # For classification tasks (_img, _txt), create simple image-to-class mapping
        if modality in ['img', 'txt']:
            unique_images = list(set(self.images))
            self.iid_to_cls = {img: idx for idx, img in enumerate(unique_images)}
            logger.info(f'[LOAD] [{self.name}] Created {len(self.iid_to_cls)} image-to-class mappings!')
    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(os.path.join(self.root, 'flickr30k_images', image_path)).convert('RGB')
        caption = self.captions[index]
        indice = index

        if self.transform is not None:
            image = self.transform(image)
        
        if self.modality == 'img+txt':
            x_input = image
            if self.tokenizer is not None:
                target = self.tokenizer(caption, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")['input_ids'][0]
            else:
                target = caption
        elif self.modality == 'img' or self.modality == 'txt':
            # Use simple image-to-class mapping for classification tasks
            if image_path not in self.iid_to_cls:
                target = 0  # Default target
            else:
                target = self.iid_to_cls[image_path]
            if self.modality == 'img':
                x_input = image
            else:
                if self.tokenizer is not None:
                    x_input = self.tokenizer(caption, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")['input_ids'][0]
                else:
                    x_input = caption
        else:
            raise ValueError(f'Unknown modality {self.modality}')

        return x_input, target

    def reduce_samples(self, num_samples=1000):
        if num_samples > len(self):
            raise ValueError(f"num_samples ({num_samples}) cannot be greater than the dataset size ({len(self)}).")
        
        # Use negative indexing to get the last num_samples items (consistent with COCO)
        sampled = np.arange(0, num_samples)

        self.images = list(operator.itemgetter(*sampled)(self.images))
        self.captions = list(operator.itemgetter(*sampled)(self.captions))
        self.n_images = len(set(self.images))
        logger.info(f'[LOAD] [FLICKR] Reduced dataset to {num_samples} samples!')

    
    def __len__(self):
        return len(self.images)

def fetch_flickr30k(args, root, transforms, tokenizer, modality='img+txt'):
    logger.info('[LOAD] [FLICKR] Fetching dataset!')
    
    if modality == 'img':
        name = 'Flickr30k_img'
    elif modality == 'txt':
        name = 'Flickr30k_txt'
    else:
        name = 'Flickr30k'
    
    # configure arguments for dataset
    dataset_args = {'root': root, 'transform': transforms[0], "split": "train", "tokenizer": tokenizer, "max_length": args.seq_len, "train_all": args.flickr_train_all, 'modality': modality, 'name': name}

    # create dataset instance
    raw_train = Flickr30kCap(**dataset_args)
    if args.reduce_samples >0 :
        raw_train.reduce_samples(args.reduce_samples)
    if args.reduce_samples_seg_scale > 0:
        raw_train.reduce_samples(num_samples=int(len(raw_train) * args.reduce_samples_seg_scale))
    if modality == 'img' or modality == 'txt':
        raw_train.task = 'cls'
    else:
        raw_train.task = 'img+txt'
    raw_train.modality = modality
    raw_train.name = name


    test_args = dataset_args.copy()
    test_args['transform'] = transforms[1]
    test_args['split'] = 'test'
    test_args['train_all'] = False

    raw_test = Flickr30kCap(**test_args)
    if args.reduce_test_samples >0 and args.reduce_test_samples < len(raw_test):
        raw_test.reduce_samples(args.reduce_test_samples)
    if modality == 'img' or modality == 'txt':
        raw_test.task = 'cls'
    else:
        raw_test.task = 'img+txt'
    raw_test.modality = modality
    raw_test.name = name
    
    logger.info('[LOAD] [FLICKR] ...fetched dataset!')

    args.in_channels = 3
    args.num_classes = None
    
    return raw_train, raw_test, args
