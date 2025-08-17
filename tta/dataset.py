import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from dataset.random_erasing import RandomErasing
from dataset.search_dataset import search_train_dataset, search_test_dataset


import os
import random
from random import randint, shuffle
from random import random as rand
import numpy as np
from PIL import Image

from torch.utils.data import Dataset

from dataset.utils import pre_caption, read_json_to_list



class search_tta_dataset:
    def __init__(self, config, transform):
        self.image_root = config['image_root']
        self.transform = transform
        self.max_words = config['max_words']
        self.eda_p = config['eda_p']

        self.be_hard = config.get('be_hard', False)
        self.be_pose_img = config.get('be_pose_img', False)
        print('tta dataset -->    be_hard:', self.be_hard, '    be_pose_img:', self.be_pose_img)

        ann_file = config['tta_file']
        self.ann = []
        for f in ann_file:
            anns = read_json_to_list(f)
            for item in anns:
                self.ann.append(item)

        self.img_ids = {}
        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

            if self.be_hard:
                img_id = ann['hard_i_id']
                if img_id not in self.img_ids.keys():
                    self.img_ids[img_id] = n
                    n += 1
        print('image ids:', n)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        img_id = ann['image_id']

        cap = ann['caption']
        caption = pre_caption(cap, self.max_words)

        if self.be_hard:
            hard_caption = pre_caption(ann['hard_c'], self.max_words)
        else:
            hard_caption = {}

        caption_eda = pre_caption(cap, self.max_words, True, self.eda_p)

        if self.be_pose_img:
            pose_path = os.path.join(self.image_root, 'pose/' + ann['image'])
            pose = Image.open(pose_path).convert('RGB')
            pose = self.transform(pose)
        else:
            pose = {}

        if self.be_hard:
            hard_path = os.path.join(self.image_root, 'train/' + ann['hard_i'])
            hard_i = Image.open(hard_path).convert('RGB')
            hard_i= self.transform(hard_i)
            if self.be_pose_img:
                hard_pose_path = os.path.join(self.image_root, 'pose/train/' + ann['hard_i'])
                hard_i_pose = Image.open(hard_pose_path).convert('RGB')
                hard_i_pose = self.transform(hard_i_pose)
            else:
                hard_i_pose = {}
        else:
            hard_i = {}
            hard_i_pose = {}

        return image, caption, caption_eda, self.img_ids[img_id], pose, hard_i, hard_i_pose, hard_caption



def create_tta_dataset(config, tta=False):

    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    tta_transform = transforms.Compose([
        transforms.Resize((config['h'], config['w']), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        RandomErasing(probability=config['erasing_p'], mean=[0.0, 0.0, 0.0])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((config['h'], config['w']), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    test_dataset = search_test_dataset(config, test_transform)

    tta_dataset = search_tta_dataset(config, tta_transform)

    return tta_dataset, test_dataset


def create_tta_loader(datasets, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, bs, n_worker, is_train, collate_fn in zip(datasets, batch_size, num_workers, is_trains, collate_fns):
        if is_train:
            shuffle = True
            drop_last = True
        else:
            shuffle = False
            drop_last = False

        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)

    if len(loaders) <= 1:
        print(f"### be careful: func create_loader returns a list length of {len(loaders)}")

    return loaders
