#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.utils.data as data

import os
import pickle
import random
import sys

import numpy as np
from scipy.spatial.distance import cdist

class QuickDrawDataset(data.Dataset):
    def __init__(self, split='train', root_dir='/root/code/SAKE/dataset/Sketchy/', classes_file='', class_emb, vocab, transform=None):
        self.transform = transform
        self.train_class = train_class
        self.dicts_class = dicts_class
        self.word2vec = class_emb
        self.vocab = vocab



        self.dir_image = os.path.join(args.data_path, 'QuickDraw_images_final')
        self.dir_sketch = os.path.join(args.data_path, 'QuickDraw_sketches_final')
        self.loader = default_image_loader
        self.fnames_sketch, self.cls_sketch = get_file_list(self.dir_sketch,
                                                            self.train_class, 'sketch')
        self.temp = 0.1  # Similarity temperature
        self.w2v_sim = np.exp(-np.square(cdist(self.word2vec, self.word2vec, 'euclidean')) / self.temp)

    def __getitem__(self, index):
        # Read sketch

        fname = os.path.join(self.dir_sketch, self.cls_sketch[index], self.fnames_sketch[index])
        sketch = self.loader(fname)
        sketch = self.transform(sketch)

        # Target
        label = self.cls_sketch[index]
        lbl = self.dicts_class.get(label)

        # Word 2 Vec (Semantics)
        w2v = torch.FloatTensor(self.word2vec[self.vocab.index(label), :])

        # Negative class
        # Hard negative
        sim = self.w2v_sim[self.vocab.index(label), :]
        possible_classes = [x for x in self.train_class if x != label]
        sim = [sim[self.vocab.index(x)] for x in possible_classes]
        # Similarity to probability
        norm = np.linalg.norm(sim, ord=1)
        sim = sim / norm
        label_neg = np.random.choice(possible_classes, 1, p=sim)[0]
        lbl_neg = self.dicts_class.get(label_neg)

        # Positive image
        # The constraint according to the ECCV 2018
        # fname = os.path.join(self.dir_image, label, (fname.split('/')[-1].split('-')[0]+'.jpg'))
        fname = get_random_file_from_path(os.path.join(self.dir_image, label))
        image = self.transform(self.loader(fname))

        fname = get_random_file_from_path(os.path.join(self.dir_image, label_neg))
        image_neg = self.transform(self.loader(fname))

        return sketch, image, image_neg, w2v, lbl, lbl_neg

    def __len__(self):
        return len(self.fnames_sketch)

    def get_classDict(self):
        return self.train_class

