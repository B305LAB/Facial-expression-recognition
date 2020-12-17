import os
import time
import math
from csv import reader
import pickle

import numpy as np
import cv2
import random
import torch
from torch.utils.data import Dataset

random.seed(0)
np.random.seed(0)

def img_sorted(imgs_n):
    if not imgs_n:
        return imgs_n
    if imgs_n[0].split('_')[0] is 'I':
        imgs_n = sorted(imgs_n)
    else:
        imgs_n.sort(key=lambda x: int(x.split('_')[0]))
    return imgs_n

class AFEWDataset(Dataset):
    def __init__(self, cfg, mode):
        self._cfg = dict(cfg)
        self._mode = str(mode) # 1:train, 2:val, 3:test
        self._preprocess()
        self._statistic()

    def _preprocess(self):

        if self._mode == '1':
            self._anno_path = os.path.join(self._cfg['LABELPATH.TRAIN'], )
            self._faces_dir = self._cfg['FACES_DIR.TRAIN']
            cache_path = self._cfg['CACHE.TRAIN']
        elif self._mode == '2':
            self._anno_path = os.path.join(self._cfg['LABELPATH.VAL'], )
            self._faces_dir = self._cfg['FACES_DIR.VAL']
            cache_path = self._cfg['CACHE.VAL']
        elif self._mode == '3':
            self._anno_path = os.path.join(self._cfg['LABELPATH.TEST'], )
            self._faces_dir = self._cfg['FACES_DIR.TEST']
            cache_path = self._cfg['CACHE.TEST']
        else:
            raise Exception('#mode: {} is WRONG! '
                            .format(self._mode))

        if os.path.exists(cache_path):
            print('loading from {}...'.format(cache_path))
            with open(cache_path, 'rb') as f:
                self.imdb = pickle.load(f)
        else:
            self._make_imdb()
            with open(cache_path, 'wb') as f:
                pickle.dump(self.imdb, f)


    def _make_imdb(self):
        self.imdb = []

        with open(self._anno_path) as f:
            lines = f.readlines()

        lines_iter = reader(lines)
        for line in lines_iter:
            vid_name = line[0]
            emo = line[2]

            vid_path = os.path.join(self._faces_dir, vid_name)
            if not os.path.exists(vid_path):
                continue

            img_names = os.listdir(vid_path)
            n = len(img_names)
            if n < 10:
                continue

            img_names = img_sorted(img_names)

            for i in range(n // 5 - 1):
                imgs_path = []
                for img_name in img_names[i*5 : i*5 + 10]:
                    img_path = os.path.join(vid_path, img_name)
                    imgs_path.append(img_path)

                self.imdb.append([imgs_path, vid_name, emo])

            if n % 5 != 0:
                imgs_path = []
                for img_name in img_names[-10:]:
                    img_path = os.path.join(vid_path, img_name)
                    imgs_path.append(img_path)

                self.imdb.append([imgs_path, vid_name, emo])

        random.shuffle(self.imdb)


    def make_imgs(self, imgs_info):
        imgs_path, vid_name, label = imgs_info
        label = int(label)

        n = len(imgs_path)
        m = 224
        imgs = torch.zeros(n, 3, m, m)
        for i in range(n):
            if vid_name[-1] == '_':
                img = cv2.imread(imgs_path[i], 0)[12:244, 12:244]
            else:
                img = cv2.imread(imgs_path[i], 0)
            img = cv2.resize(img, (m, m))
            img = img[None, :].repeat(3, axis=0).astype(np.float32)
            img = torch.from_numpy(img)

            imgs[i] = img

        # imgs = torch.from_numpy(np.array(imgs))
        label = np.eye(7)[label].astype(np.float32)

        return imgs, vid_name, label

    def _statistic(self):
        classes = {}
        for im in self.imdb:
            imgs_path, vid_name, label = im
            classes[label] = classes.get(label, 0) + 1

        self.statistic = {}
        self.statistic['classes'] = classes



    def __len__(self):
        return len(self.imdb)

    def __getitem__(self, idx):
        imgs_info = self.imdb[idx]
        imgs, vid_name, label = self.make_imgs(imgs_info)

        return (imgs, vid_name, label, idx)


