import os
from sklearn.model_selection import train_test_split

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
import random
import dlib
import time
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

class VideoDatasetPreloaded(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            dataset (str): Name of dataset. Defaults to 'ucf101'.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
    """

    def __init__(self, dataset='ABAW', label_type="VA_Set", split='Training_Set', clip_len=16, stride=2, add_landmarks=False,\
        mini=False, check_dirty_list=False, triplet_label=False):

        self.root_dir, self.output_dir = Path.db_dir(dataset)
        # self.output_dir = '/public/home/fanyachun/data/ABAW/dataset'
        if split == 'Test_Set':
            image_folder = os.path.join(self.output_dir, label_type, 'Validation_Set', 'image')
            va_label_folder = os.path.join(self.output_dir, "VA_Set", 'Validation_Set', 'label')
            exp_label_folder = os.path.join(self.output_dir, "EXPR_Set", 'Validation_Set', 'label')
        elif split == 'Submission_Set':
            image_folder = os.path.join(self.output_dir, label_type, 'Testing_Set', 'image')
            va_label_folder = os.path.join(self.output_dir, "VA_Set", 'Validation_Set', 'label')
            exp_label_folder = os.path.join(self.output_dir, "EXPR_Set", 'Validation_Set', 'label')
        else:
            image_folder = os.path.join(self.output_dir, label_type, split, 'image')
            va_label_folder = os.path.join(self.output_dir, "VA_Set", split, 'label')
            exp_label_folder = os.path.join(self.output_dir, "EXPR_Set", split, 'label')

        self.clip_len = clip_len
        self.stride = stride
        self.split = split
        self.add_landmarks = add_landmarks
        self.check_dirty_list = check_dirty_list
        self.triplet_label = triplet_label

        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 112
        self.resize_width = 112
        self.is_submission = split == 'Submission_Set'

        # Obtain all the filenames of files inside all the class folders
        # Going through each class folder one at a time
        self.fnames, self.va_labels, self.exp_labels = [], [], [] # [所有video对应image文件夹的路径]，[所有label的txt的路径]
        for video_filename in sorted(os.listdir(image_folder)):
            # if video_filename[0] == 'h' or video_filename[0] == 'g':
            #     continue
            frame_path = os.path.join(image_folder, video_filename)
            if len(os.listdir(frame_path)) < 32:
                continue
            
            self.fnames.append(os.path.join(image_folder, video_filename))
            self.va_labels.append(os.path.join(va_label_folder, video_filename + '.txt'))
            # self.exp_labels.append(os.path.join(exp_label_folder, video_filename + '.txt'))

        assert len(self.va_labels) == len(self.fnames)

        self.preload(clip_len, stride)
        print('Number of {} videos: {:d}'.format(split, len(self.fnames)))
        print('Number of {} samples: {:d}'.format(split, len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        buffer, valence_label, arousal_label = self.data[index]
        image_clip = self.normalize(buffer)
        image_clip = self.to_tensor(buffer)

        return image_clip, valence_label, arousal_label

    def normalize(self, buffer):
        return (buffer - 128) / 128

    def to_tensor(self, buffer):
        return torch.from_numpy(buffer.transpose((3, 0, 1, 2)))

    def preload(self, clip_len, stride):
        print("Start preloading video clips into memory")
        self.data = []
        for i, video_fname in tqdm(enumerate(self.fnames)):
            # 把这个视频的所有label读出来
            va_label_path = self.va_labels[i]
            with open(va_label_path, 'rt') as f:
                lines = f.read().splitlines()

            # 排除非jpg文件
            frame_count = 0
            frame_names_per_video = []
            for each in os.listdir(video_fname):
                if '.jpg' in each:
                    frame_names_per_video.append(each)
                    frame_count += 1

            frames = sorted([os.path.join(video_fname, img) for img in frame_names_per_video], \
                key=lambda x: int(x.split('/')[-1].split('.')[0]) )
            for idx in range(int(clip_len/2*stride), int(frame_count - clip_len/2*stride), 512): # 中间一帧的index，向前找clip_len/2，向后找clip_len/2
                
                mid_time_index = int(frames[idx].split('/')[-1].split(".")[0]) # index转换为真实时间帧号
                clip = frames[idx-int(clip_len/2)*stride : idx+int(clip_len/2)*stride : stride]
                buffer = np.empty((clip_len, self.resize_height, self.resize_width, 3), np.dtype('float32'))

                for i, frame_name in enumerate(clip):
                    frame = np.array(cv2.imread(frame_name)).astype(np.float32)
                    frame = cv2.resize(frame, (self.resize_height, self.resize_width))
                    # Data Augmentation, 先转PIL 再transform。。。
                    frame_PIL = Image.fromarray(np.uint8(frame))
                    frame_PIL = transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05)(frame_PIL)
                    frame = np.asarray(frame_PIL)
                    buffer[i] = frame

                # 这一帧对应的label
                line = lines[mid_time_index]
                valence_label = float(line.split(',')[0])
                arousal_label = float(line.split(',')[1])
                if (valence_label == -5 or arousal_label == -5):
                    print('label not valid: ', str(video_fname.split('/')[-1], str(idx)))
                    is_label_valid = False
                    continue

                self.data.append( (buffer, valence_label, arousal_label) )


class VideoDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            dataset (str): Name of dataset. Defaults to 'ucf101'.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
    """

    def __init__(self, dataset='ABAW', label_type="VA_Set", split='Training_Set', clip_len=16, stride=2, add_landmarks=False,\
        mini=False, check_dirty_list=False, triplet_label=False):

        self.root_dir, self.output_dir = Path.db_dir(dataset)
        # self.output_dir = '/public/home/fanyachun/data/ABAW/dataset'
        if split == 'Test_Set':
            image_folder = os.path.join(self.output_dir, label_type, 'Validation_Set', 'image')
            va_label_folder = os.path.join(self.output_dir, "VA_Set", 'Validation_Set', 'label')
            exp_label_folder = os.path.join(self.output_dir, "EXPR_Set", 'Validation_Set', 'label')
        elif split == 'Submission_Set':
            image_folder = os.path.join(self.output_dir, label_type, 'Testing_Set', 'image')
            va_label_folder = os.path.join(self.output_dir, "VA_Set", 'Validation_Set', 'label')
            exp_label_folder = os.path.join(self.output_dir, "EXPR_Set", 'Validation_Set', 'label')
        else:
            image_folder = os.path.join(self.output_dir, label_type, split, 'image')
            va_label_folder = os.path.join(self.output_dir, "VA_Set", split, 'label')
            exp_label_folder = os.path.join(self.output_dir, "EXPR_Set", split, 'label')

        self.clip_len = clip_len
        self.stride = stride
        self.split = split
        self.add_landmarks = add_landmarks
        self.check_dirty_list = check_dirty_list
        self.triplet_label = triplet_label

        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 112
        self.resize_width = 112
        self.is_submission = split == 'Submission_Set'

        # Obtain all the filenames of files inside all the class folders
        # Going through each class folder one at a time
        self.fnames, self.va_labels, self.exp_labels = [], [], [] # [所有video对应image文件夹的路径]，[所有label的txt的路径]
        for video_filename in sorted(os.listdir(image_folder)):
            # if video_filename[0] == 'h' or video_filename[0] == 'g':
            #     continue
            frame_path = os.path.join(image_folder, video_filename)
            if len(os.listdir(frame_path)) < 32:
                continue
            
            self.fnames.append(os.path.join(image_folder, video_filename))
            self.va_labels.append(os.path.join(va_label_folder, video_filename + '.txt'))
            # self.exp_labels.append(os.path.join(exp_label_folder, video_filename + '.txt'))
            if mini:
                # 在小数据上跑一下
                if len(self.fnames) > 100:
                    break

        assert len(self.va_labels) == len(self.fnames)
        print('Number of {} videos: {:d}'.format(split, len(self.fnames)))

        if check_dirty_list:
            self.dirty_list = self.get_dirty_list()
        else:
            self.dirty_list = {}


    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # Loading
        if self.split == 'Test_Set':
            # 测试集只返回文件名list, 在train中读图片
            # 排除非jpg文件
            file_dir = self.fnames[index]
            fs = []
            for each in os.listdir(file_dir):
                if '.jpg' in each:
                    fs.append(each)
            frames = sorted([os.path.join(file_dir, img) for img in fs], key=lambda x: int(x.split('/')[-1].split('.')[0]) )
            return frames, self.va_labels[index]
        elif self.split == 'Submission_Set':
            file_dir = self.fnames[index]
            fs = []
            for each in os.listdir(file_dir):
                if '.jpg' in each:
                    fs.append(each)
            frames = sorted([os.path.join(file_dir, img) for img in fs], key=lambda x: int(x.split('/')[-1].split('.')[0]) )
            return frames, []
        else:
            image_clip, valence_label, arousal_label, exp_label, center = self.crop(file_dir=self.fnames[index], \
                va_label_path=self.va_labels[index], exp_label_path = None, stride=self.stride, clip_len=self.clip_len)

            image_clip = self.normalize(image_clip)
            image_clip = self.to_tensor(image_clip)

            # avgface_path = self.fnames[index].replace('image', 'avgfaces')
            # avgface_path = os.path.join(avgface_path, 'avgface.png')
            # avgface = cv2.imread(avgface_path, cv2.IMREAD_GRAYSCALE)
            # avgface = np.stack((avgface,)*3, axis=-1)

            # avgface = (avgface - 128) / 128
            # avgface = torch.from_numpy(avgface.transpose(2,0,1)).float()

            return image_clip, valence_label, arousal_label, exp_label, self.fnames[index], center

    def get_dirty_list(self):
        dirty_list = {}
        with open('/media/ubuntu/datasets/ABAW/dataset/VA_Set/Training_Set/dirty_list.txt', 'r') as f:
        # with open('/public/home/fanyachun/data/ABAW/dataset/VA_Set/Training_Set/dirty_list.txt', 'r') as f:
            line = f.readline()
            while line:
                if len(line) < 3:
                    line = f.readline()
                    continue
                video_id = line.split(' ')[0]
                ranges = line.split(' ')[1:]
                dirty_list[video_id] = []
                for r in ranges:
                    if r == '\n': continue  
                    if '-' in r:
                        start = int(r.split('-')[0])
                        end = int(r.split('-')[1])
                    else:
                        start = int(r)
                        end = int(r)
                    if start > end:
                        print("ERROR", video_id, start, end)
                    dirty_list[video_id].append(range(start-8, end+9))
                line = f.readline()
        return dirty_list

    def normalize(self, buffer):
        # if buffer.shape[3] == 4:
        #     for i, frame in enumerate(buffer):
        #         buffer[i] = ( frame - np.array([[[86.0, 93.8, 118.1, 128]]]) ) / 128.0
        # else:
        #     for i, frame in enumerate(buffer):
        #         buffer[i] = ( frame - np.array([[[86.0, 93.8, 118.1]]]) ) / 128.0

        # return (buffer - buffer.mean()) / buffer.var()
        
        return (buffer - 128) / 128

    def to_tensor(self, buffer):
        return torch.from_numpy(buffer.transpose((3, 0, 1, 2)))


    def crop(self, file_dir, va_label_path, exp_label_path, clip_len, stride):
        if self.add_landmarks:
            buffer = np.empty((clip_len, self.resize_height, self.resize_width, 4), np.dtype('float32'))
        else:
            buffer = np.empty((clip_len, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        
        # 排除非jpg文件
        frame_count = 0
        fs = []
        for each in os.listdir(file_dir):
            if '.jpg' in each:
                fs.append(each)
                frame_count += 1

        frames = sorted([os.path.join(file_dir, img) for img in fs], \
            key=lambda x: int(x.split('/')[-1].split('.')[0]) )

        valence_label = None
        arousal_label = None
        exp_label = -1
        # exp_label_onehot = np.zeros(8)
        # exp_label_onehot[7] = 1 # 无效label

        while True:
            '''随机裁一个时间窗口'''
            if not clip_len*stride <= frame_count: # Frame count is too small, or stride is too large?
                idx = random.randint(0, frame_count-1)
            else:
                idx = random.randint(clip_len/2*stride, frame_count - clip_len/2*stride) # 中间一帧的index，向前找clip_len/2，向后找clip_len/2
            mid_time_index = int(frames[idx].split('/')[-1].split(".")[0])
            clip = frames[idx-int(clip_len/2)*stride : idx+int(clip_len/2)*stride : stride]

            if not self.is_submission:
                # 取label
                is_label_valid = True
                with open(va_label_path, 'rt') as f:
                    lines = f.read().splitlines()

                    if not self.triplet_label:
                        line = lines[mid_time_index]
                        valence_label = ( float(line.split(',')[0]) )
                        arousal_label = ( float(line.split(',')[1]) )
                        if (valence_label == -5 or arousal_label == -5):
                            # print('label not valid')
                            is_label_valid = False
                    else:
                        prev_idx = int(max(0, idx - clip_len/2*stride))
                        prev_time_index = int(frames[prev_idx].split('/')[-1].split(".")[0]) # 序列头  
                        next_idx = int(min(len(frames)-1, idx + clip_len/2*stride))
                        next_time_index = int(frames[next_idx].split('/')[-1].split(".")[0]) # 序列尾
                        line1 = lines[prev_time_index]
                        line2 = lines[mid_time_index]
                        line3 = lines[next_time_index]
                        valence_label1 = ( float(line1.split(',')[0]) )
                        arousal_label1 = ( float(line1.split(',')[1]) )
                        valence_label2 = ( float(line2.split(',')[0]) )
                        arousal_label2 = ( float(line2.split(',')[1]) )
                        valence_label3 = ( float(line3.split(',')[0]) )
                        arousal_label3 = ( float(line3.split(',')[1]) )
                        valence_label = (valence_label1, valence_label2, valence_label3)
                        arousal_label = (arousal_label1, arousal_label2, arousal_label3)
                        if (valence_label1 == -5 or arousal_label1 == -5 or
                            valence_label2 == -5 or arousal_label2 == -5 or
                            valence_label3 == -5 or arousal_label3 == -5):
                            print('label not valid')
                            is_label_valid = False

                if not is_label_valid:
                    continue
                
                # if os.path.exists(exp_label_path):
                #     with open(exp_label_path, 'rt') as f:
                #         lines = f.read().splitlines()

                #         if True:
                #             line = lines[mid_time_index]
                #             exp_label = int(line)
                #             # if not exp_label == -1:
                #             #     exp_label_onehot[exp_label] = 1
                #             #     exp_label_onehot[7] = 0 # 有效label
                #         else:
                #             # t1 = int(max(1, mid_time_index - clip_len/2*stride))             # 序列头
                #             # t2 = int(mid_time_index)                                         # 序列中心
                #             # t3 = int(min(len(lines)-1, mid_time_index + clip_len/2*stride))  # 序列尾
                #             line1 = lines[prev_time_index]
                #             line2 = lines[mid_time_index]
                #             line3 = lines[next_time_index]
                #             exp_label1 = int(line1)
                #             exp_label2 = int(line2)
                #             exp_label3 = int(line3)
                #             exp_label = (exp_label1, exp_label2, exp_label3)



            # 检查video和frame是否在dirty list中
            video_id = file_dir.split('/')[-1]
            assert int(frames[idx].split('/')[-1].split(".")[0]) == mid_time_index # something wrong in timeline..?
            frame_id = int(frames[idx].split('/')[-1].split('.')[0])  
            
            if video_id not in self.dirty_list:
                break
            is_valid = True
            for rg in self.dirty_list[video_id]:
                if frame_id in rg:
                    is_valid = False
                    # print(video_id + ":" + str(frame_id) + " not valid")
                    break
            if is_valid: break

        for i, frame_name in enumerate(clip):
            frame = np.array(cv2.imread(frame_name)).astype(np.float32)
            frame = cv2.resize(frame, (self.resize_height, self.resize_width))
            # Data Augmentation, 先转PIL 再transform。。。
            #cv2.imshow('test', frame/255)
            #cv2.waitKey(1000)
            frame_PIL = Image.fromarray(np.uint8(frame))
            frame_PIL = transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05)(frame_PIL)
            #cv2.imshow('test', np.asarray(frame_PIL)/255)
            #cv2.waitKey(1000)
            frame = np.asarray(frame_PIL)
            
            if self.add_landmarks:
                # 读landmarks
                lm_name = frame_name.replace('image', 'landmarks')
                lm_img = np.array(cv2.imread(lm_name, cv2.IMREAD_GRAYSCALE)).astype(np.float32)
                lm_img = cv2.resize(lm_img, (self.resize_height, self.resize_width))
                #cv2.imshow('lm', np.asarray(lm_img)/255)
                #cv2.waitKey(1000)
                lm_img = np.expand_dims(lm_img, 2)
                # 把landmark拼接上
                buffer[i] = np.concatenate((frame, lm_img),2)
            else:
                buffer[i] = frame


        #return buffer, np.array(valence_label).astype(np.float32), np.array(arousal_label).astype(np.float32), mid_time_index
        return buffer, valence_label, arousal_label, -1, mid_time_index

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    train_data = VideoDataset(dataset='affwild2', split='train', clip_len=16)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=1)
    for i, sample in enumerate(train_data):
        inputs = sample[0]
        valence_labels = sample[1]
        arousal_labels = sample[2]
        print(inputs.size())
        print(valence_labels)
        print(arousal_labels)

        if i == 1:
            break