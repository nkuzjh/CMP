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



def create_test_dataset(config):

    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    # tta_transform = transforms.Compose([
    #     transforms.Resize((config['h'], config['w']), interpolation=InterpolationMode.BICUBIC),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     normalize,
    #     RandomErasing(probability=config['erasing_p'], mean=[0.0, 0.0, 0.0])
    # ])

    test_transform = transforms.Compose([
        transforms.Resize((config['h'], config['w']), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    test_dataset = search_test_dataset(config, test_transform)

    # tta_dataset = search_tta_dataset(config, tta_transform)

    # return tta_dataset, test_dataset
    return test_dataset


def create_test_loader(datasets, batch_size, num_workers, is_trains, collate_fns):
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



class search_tta_dataset(Dataset):
    def __init__(self, config, tta_transform, sims_matrix_t2i, image_embeds, text_embeds, text_atts, recall_types, ss_idxs_list, uncertaintys_list, proba_top1_sim_list, proba_inversed_sim_list):
        # ann_file = config['tta_file']
        # self.transform = transform
        # self.image_root = config.get('image_root_tta', config['image_root'])
        # self.max_words = config['max_words']# 56

        # self.ann = read_json_to_list(ann_file)

        # self.be_pose_img = config.get('be_pose_img', False)
        # print('tta dataset -->    be_pose_img:', self.be_pose_img)

        # self.text = []
        # self.image = []
        # self.g_pids = []
        # self.q_pids = []
        # for img_id, ann in enumerate(self.ann):
        #     self.g_pids.append(ann['image_id'])
        #     self.image.append(ann['image'])
        #     for i, caption in enumerate(ann['caption']):
        #         self.q_pids.append(ann['image_id'])
        #         self.text.append(pre_caption(caption, self.max_words))
        self.config = config
        # self.transform = tta_transform
        self.sims_matrix_t2i = sims_matrix_t2i
        self.image_embeds= image_embeds
        self.text_embeds = text_embeds
        self.text_atts = text_atts
        # self.recall_types = recall_types
        # self.ss_idxs_list = ss_idxs_list
        self.uncertaintys_list = uncertaintys_list
        # if config.get('uncertainty_temper_is_learnable', False) == True:
        if 1:
            self.proba_top1_sim_list = proba_top1_sim_list
            self.proba_inversed_sim_list = proba_inversed_sim_list

        if config.get('sample_selection', 'all') == 'top1':
            self.sims_matrix_t2i = sims_matrix_t2i[ss_idxs_list]
            #### self.image_embeds= image_embeds[ss_idxs_list]
            self.text_embeds = text_embeds[ss_idxs_list]
            self.text_atts = text_atts[ss_idxs_list]
            # self.labels = [labels[i] for i in ss_idxs_list]
            # self.recall_types = [recall_types[i] for i in ss_idxs_list]
            self.uncertaintys_list = [uncertaintys_list[i] for i in ss_idxs_list]
            # if config.get('uncertainty_temper_is_learnable', False) == True:
            if 1:
                self.proba_top1_sim_list = [proba_top1_sim_list[i] for i in ss_idxs_list]
                self.proba_inversed_sim_list = [proba_inversed_sim_list[i] for i in ss_idxs_list]

    def __len__(self):
        return len(self.sims_matrix_t2i)

    def __getitem__(self, index):
        # image_path = os.path.join(self.image_root, self.ann[index]['image'])
        # image = Image.open(image_path).convert('RGB')
        # image = self.transform(image)

        # if self.be_pose_img:
        #     pose_path = os.path.join(self.image_root, 'pose/' + self.ann[index]['image'])
        #     pose = Image.open(pose_path).convert('RGB')
        #     pose = self.transform(pose)
        # else:
        #     pose = {}

        # return image, pose, index
        topk_sim, topk_idx = self.sims_matrix_t2i[index].topk(k=self.config['k_tta'], dim=0) #[k_tta]
        encoder_output = self.image_embeds[topk_idx] #[k_tta, 50, 1024]
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long) #[k_tta, 50])
        text_embeds = self.text_embeds[index].repeat(self.config['k_tta'], 1, 1) #k_tta, 56, 768])
        text_atts = self.text_atts[index].repeat(self.config['k_tta'], 1) #k_tta, 56
        uncertainty = self.uncertaintys_list[index]
        proba_top1_sim = self.proba_top1_sim_list[index]
        proba_inversed_sim = self.proba_inversed_sim_list[index]

        return encoder_output, encoder_att, text_embeds, text_atts, uncertainty, proba_top1_sim, proba_inversed_sim


class search_tta_img_aug_idx_dataset(Dataset):
    def __init__(self, config, sims_matrix_t2i, text_embeds, text_atts, ss_idxs_list, uncertaintys_list, proba_top1_sim_list, proba_inversed_sim_list):
        self.config = config

        ## using cosine similarity matrix & text feature & uncertianty without augmentation
        self.sims_matrix_t2i = sims_matrix_t2i
        self.text_embeds = text_embeds
        self.text_atts = text_atts
        self.uncertaintys_list = uncertaintys_list
        # if config.get('uncertainty_temper_is_learnable', False) == True:
        if 1:
            self.proba_top1_sim_list = proba_top1_sim_list
            self.proba_inversed_sim_list = proba_inversed_sim_list

        ## sample selection strategy based on inverse recall probability
        if config.get('sample_selection', 'all') == 'top1':
            self.sims_matrix_t2i = sims_matrix_t2i[ss_idxs_list]
            self.text_embeds = text_embeds[ss_idxs_list]
            self.text_atts = text_atts[ss_idxs_list]
            self.uncertaintys_list = [uncertaintys_list[i] for i in ss_idxs_list]
            # if config.get('uncertainty_temper_is_learnable', False) == True:
            if 1:
                self.proba_top1_sim_list = [proba_top1_sim_list[i] for i in ss_idxs_list]
                self.proba_inversed_sim_list = [proba_inversed_sim_list[i] for i in ss_idxs_list]

    def __len__(self):
        return len(self.sims_matrix_t2i)

    def __getitem__(self, index):
        topk_sim, topk_idx = self.sims_matrix_t2i[index].topk(k=self.config['k_tta'], dim=0) #[k_tta]
        ## 仅返回topk_idx，用于索引aug_img_embeds
        text_embeds = self.text_embeds[index].repeat(self.config['k_tta'], 1, 1) #k_tta, 56, 768])
        text_atts = self.text_atts[index].repeat(self.config['k_tta'], 1) #k_tta, 56
        uncertainty = self.uncertaintys_list[index]
        proba_top1_sim = self.proba_top1_sim_list[index]
        proba_inversed_sim = self.proba_inversed_sim_list[index]

        return topk_idx, text_embeds, text_atts, uncertainty, proba_top1_sim, proba_inversed_sim


def create_tta_dataset(config, sims_matrix_t2i, image_embeds, text_embeds, text_atts, recall_types, ss_idxs_list, uncertaintys_list, proba_top1_sim_list, proba_inversed_sim_list,):

    # normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    # if config.get('is_image_augmentation', False):
    #     tta_transform = transforms.Compose([
    #         transforms.Resize((config['h'], config['w']), interpolation=InterpolationMode.BICUBIC),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #         RandomErasing(probability=config['erasing_p'], mean=[0.0, 0.0, 0.0])
    #     ])
    # else:
    #     tta_transform = transforms.Compose([
    #         transforms.Resize((config['h'], config['w']), interpolation=InterpolationMode.BICUBIC),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])
    tta_transform = None

    if config.get('is_image_augmentation', False):
        tta_dataset = search_tta_img_aug_idx_dataset(config, sims_matrix_t2i, text_embeds, text_atts, ss_idxs_list, uncertaintys_list, proba_top1_sim_list, proba_inversed_sim_list)
    else:
        tta_dataset = search_tta_dataset(config, tta_transform, sims_matrix_t2i, image_embeds, text_embeds, text_atts, recall_types, ss_idxs_list, uncertaintys_list, proba_top1_sim_list, proba_inversed_sim_list)

    return tta_dataset


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



class search_tta_img_aug_dataset(Dataset):
    def __init__(self, config, transform):
        ann_file = config['tta_file']
        self.transform = transform
        self.image_root = config.get('image_root_tta', config['image_root'])
        self.max_words = config['max_words']

        self.ann = read_json_to_list(ann_file)

        self.be_pose_img = config.get('be_pose_img', False)
        print('     tta img aug dataset -->    be_pose_img:', self.be_pose_img)

        self.text = []
        self.image = []
        self.g_pids = []
        self.q_pids = []
        for img_id, ann in enumerate(self.ann):
            self.g_pids.append(ann['image_id'])
            self.image.append(ann['image'])
            for i, caption in enumerate(ann['caption']):
                self.q_pids.append(ann['image_id'])
                self.text.append(pre_caption(caption, self.max_words))
        pass

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.ann[index]['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        if self.be_pose_img:
            pose_path = os.path.join(self.image_root, 'pose/' + self.ann[index]['image'])
            pose = Image.open(pose_path).convert('RGB')
            pose = self.transform(pose)
        else:
            pose = {}

        return image, pose, index


def create_tta_img_aug_dataset(config):

    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    tta_img_aug_transform = transforms.Compose([
        transforms.Resize((config['h'], config['w']), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        RandomErasing(probability=config['erasing_p'], mean=[0.0, 0.0, 0.0])
    ])

    tta_img_aug_dataset = search_tta_img_aug_dataset(config, tta_img_aug_transform)

    return tta_img_aug_dataset


def create_tta_img_aug_loader(datasets, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, bs, n_worker, is_train, collate_fn in zip(datasets, batch_size, num_workers, is_trains, collate_fns):
        if is_train:
            shuffle = True
            drop_last = False
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