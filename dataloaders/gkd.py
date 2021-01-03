import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.autograd import gradcheck
import cv2
import numpy as np
import random

class DataPrefetcher():
    def __init__(self, data_loader):
        self.stream = torch.cuda.Stream()
        self.data_loader = data_loader
        self.loader = iter(data_loader)

        self.preload()

    def preload(self):
        try:
            inputs, valence_label, arousal_label, _, filename, frame_index_list = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_valence_label = None
            self.next_arousal_label = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = inputs.cuda(non_blocking=True)
            self.next_valence_label = valence_label[1].cuda()
            self.next_arousal_label = arousal_label[1].cuda()

            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        inputs = self.next_input
        next_valence_label = self.next_valence_label
        next_arousal_label = self.next_arousal_label
        self.preload()
        return inputs, next_valence_label, next_arousal_label

    def __len__(self):
        return len(self.data_loader)