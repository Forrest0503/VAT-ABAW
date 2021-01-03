
import os
import numpy as np
import cv2
import torch
from torch import nn
import random
from threading import Thread
from queue import Queue

from network import resnet3D, transformer_v3

SEED = 23333
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

run_id = 1
resume_epoch = 1000  # Default is 0, change if want to resume
CLIP_LEN = 8
FRAME_STRIDE = 4
STOP_FLAG = 0

modelName = 'Resnet3d'   # Options: C3DVA or R2Plus1D or R3D or CSN Resnet3d
dataset = 'ABAW' 
num_classes = 2
save_dir_root = '/media/ubuntu/bak/VA'
save_dir = os.path.join(save_dir_root, 'experiments', 'run_' + str(run_id))
saveName = modelName + '-' + dataset

def push_into_buffer(buffer):
    img_folder = '/media/ubuntu/datasets/ABAW/dataset/VA_Set/Validation_Set/image/1-30-1280x720/'
    frames = sorted(os.listdir(img_folder))
    print(len(frames))
    for each in frames:
        img = cv2.imread(os.path.join(img_folder, each))
        buffer.put(img)
    STOP_FLAG = 1

def get_frame_data(buffer, frames_list, clip_queue):
    while not STOP_FLAG:
        frames_list.pop(0)
        # read an image from buffer
        img = buffer.get()
        frames_list.append(img[:,:,:,np.newaxis])

        clip = np.concatenate(frames_list, axis=3)
        clip = (clip - 128) / 128
        # clip = clip.transpose((3, 0, 1, 2))
        clip = torch.from_numpy(clip).unsqueeze(0).permute(0, 3, 4, 1, 2).float() #[b, c, CLIP_LEN, h, w]
        clip_queue.put(clip)

img_folder = '/media/ubuntu/datasets/ABAW/dataset/VA_Set/Validation_Set/image/1-30-1280x720/'
frames = sorted(os.listdir(img_folder))

def get_frame_data_ABAW(buffer, frames_list, clip_queue):

    frame_ids = [] # 图片对应的帧号
    for each in frames:
        fid = int(each.split('.')[0])
        frame_ids.append(fid)
    
    for center in ( frame_ids ):
        buffer = np.empty(( CLIP_LEN, 112, 112, 3), np.dtype('float32'))
        clip = list(range(center-int(CLIP_LEN/2)*FRAME_STRIDE, center+int(CLIP_LEN/2)*FRAME_STRIDE, FRAME_STRIDE))
        for i, frame_id in enumerate(clip):
            frame_name = None
            if (frame_id < 1 or frame_id > frame_ids[-1]):
                frame = np.ones((112, 112, 3))*128
            else:
                frame_name = os.path.join(img_folder,str(frame_id).zfill(5)+'.jpg')
                #print(center, frames[i])
                
                if os.path.exists(frame_name):
                    frame = np.array(cv2.imread(frame_name)).astype(np.float32)
                else:
                    frame = np.ones((112, 112, 3))*128
            buffer[i] = frame
        
        buffer = (buffer - 128) / 128
        buffer = buffer.transpose((3, 0, 1, 2))
        inputs = torch.from_numpy(buffer).unsqueeze(0)
        clip_queue.put((center, inputs))

    STOP_FLAG = 1

def inference(clip_queue, model):
    while not STOP_FLAG:
        (center, inputs) = clip_queue.get()
        with torch.no_grad():
            outputs_va, outputs_expr = model(inputs.cuda())
            vector_c_valence = np.array([-0.9, -0.65, -0.4, -0.225, -0.1, -0.025, 0.025, 0.075, 0.125, 0.175, 
            0.225, 0.275, 0.325, 0.375, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
            vector_c_arousal = np.array([-0.9, -0.65, -0.425, -0.275, -0.15, -0.05, 0.025, 0.075, 0.125, 0.175, 
            0.225, 0.275, 0.325, 0.375, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])         
            valence_pred_cls = outputs_va[:,0:20]
            arousal_pred_cls = outputs_va[:,20:40] 
            valence_pred_cls_softmax = nn.Softmax(dim=0)(valence_pred_cls[0])
            arousal_pred_cls_softmax = nn.Softmax(dim=0)(arousal_pred_cls[0])

            valence_pred_reg = np.dot(vector_c_valence, valence_pred_cls_softmax.detach().cpu().numpy().T)
            arousal_pred_reg = np.dot(vector_c_arousal, arousal_pred_cls_softmax.detach().cpu().numpy().T)

            print(center, valence_pred_reg, arousal_pred_reg)
    print('done.')
    
    

def main():

    # load model
    model = resnet3D.resnet3d18(num_classes=400, pretrained=None)
    checkpoint = torch.load(os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar'),
                       map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
    print("Initializing weights from: {}...".format(
        os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)

    print('Start inference...'.format(dataset))

    model.eval()

    buffer = Queue(maxsize=1)
    frame_list = [np.ones((112, 112, 3, 1))*128] * CLIP_LEN
    clip_queue = Queue(maxsize=1)

    # Thread(target=push_into_buffer, args=(buffer,)).start()
    Thread(target=get_frame_data_ABAW, args=(buffer, frame_list, clip_queue)).start()
    Thread(target=inference, args=(clip_queue,model)).start()

    # print('done.')
    


if __name__ == "__main__":
    main()