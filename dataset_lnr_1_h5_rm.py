"""
lnr_1_h5: package the img in h5file to a new h5
lnr_1_h5_rm: based on ep2, but I found ep2 is too slow to train frame-based method, so I rm the event, namely change the train_event_file and test_event_file in ep2 into lnr_1_h5:
self.num_frames = int(len(self.h5_file['aug_pre'].keys())/3)
"""

from audioop import reverse
from types import GetSetDescriptorType
from torch.utils.data import Dataset
import os
import glob
import h5py
import torch
import numpy as np
import os, argparse, sys, math, random, glob, cv2
from torch.utils.data import DataLoader
import commentjson as json

class ELNRainDataset(Dataset):
    def __init__(self, root_dir, mode, idx):
        
        super(ELNRainDataset, self).__init__()
        self.event_videos = []
        self.scene_idx = idx
        self.mode = mode

        if self.mode == "train":
            event_list_filename = "/data/booker/LNRain_v2/preprocess/train_event_file.txt"
        elif self.mode == "test":
            event_list_filename = "/data/booker/LNRain_v2/preprocess/test_event_file.txt"
        else:
            event_list_filename = "/data/booker/LNRain_v2/preprocess/val_event_file.txt"

        with open(event_list_filename) as f:
            self.event_videos = [line.rstrip() for line in f.readlines()]

        self.event_dir = self.event_videos[self.scene_idx]
        self.load_data(self.event_dir, self.mode)
    
    def load_data(self, data_path, mode):
        try:
            self.h5_file = h5py.File(data_path, 'r')
        except OSError as err:
            print("Couldn't open {}: {}".format(data_path, err))
        
        if mode == "train":
            self.num_frames = int(len(self.h5_file['aug_pre'].keys())/3)
        else:
            self.num_frames = self.num_frames = self.h5_file['images'].attrs["num_images"] -1
        self.length = self.num_frames

    def get_frame(self, index, mode="train"):
        
        if mode == "train":
            frame = self.h5_file['aug_pre']['input{:05d}'.format(index)][:]
            gt_frame = self.h5_file['aug_pre']['gt{:05d}'.format(index)][:]
        else:
            frame = self.h5_file['preprocess']['input{:05d}'.format(index)][:]
            gt_frame = self.h5_file['preprocess']['gt{:05d}'.format(index)][:]

        return frame, gt_frame

    def get_gt_frame(self, index):
        return self.h5_file['preprocess']['gt{:05d}'.format(index)][:]
    
    def get_event(self, index):
        return self.h5_file['preprocess']['event{:05d}'.format(index)][:]
    
    def transform_frame(self, frame, gt_frame, event):
        frame = torch.from_numpy(frame)
        gt_frame = torch.from_numpy(gt_frame)
        event = torch.from_numpy(event)

        return frame, gt_frame, event
    
    def __len__(self):
        return self.length 

    def __getitem__(self, index):
        assert 0 <= index < self.__len__(), "index {} out of bounds (0 <= x < {})".format(index, self.__len__())
        
        frame, gt_frame = self.get_frame(index, self.mode)

        # frame, gt_frame, voxel = self.transform_frame(frame, gt_frame, voxel)

        item = {"frame":frame,
                "gt":gt_frame}
            
        return item

class SequenceDataset(Dataset):
    def __init__(self, root_dir, mode, idx, sequence_length = 7, dataset_type='ELNRainDataset', opts = None, step_size = None):
            
        self.mode = mode
        self.L = sequence_length
        self.dataset = eval(dataset_type)(root_dir, mode, idx)
        self.opts = opts
        self.step_size = step_size if step_size is not None else self.L

        assert(self.L > 0)
        assert(self.step_size > 0)

        if self.L >= self.dataset.length:
            self.length = 0
        else:
            self.length = (self.dataset.length - self.L) // self.step_size + 1

    
    def __len__(self):
        return self.length
    
    def __getitem__(self, i):
        """ Returns a list containing synchronized events <-> frame pairs
            [e_{i-L} <-> I_{i-L},
                e_{i-L+1} <-> I_{i-L+1},
            ...,
            e_{i-1} <-> I_{i-1},
            e_i <-> I_i]
        """

        assert(i >= 0)
        assert(i < self.length)

        sequence = []

        k=0
        j = i * self.step_size
        item = self.dataset.__getitem__(j)
        sequence.append(item)
        
        for n in range(self.L - 1):
            k+=1
            item = self.dataset.__getitem__(j + k)
            sequence.append(item)
        
        return sequence