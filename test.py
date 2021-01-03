import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm
import numpy as np
import cv2

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.autograd import gradcheck
import random

from dataloaders.ABAW import VideoDataset
from network import C3DVA_model, R2Plus1D_model, R2Plus1D_model_ln, R2Plus1D_model_in, R3D_model, CSN_model, \
    Resnet3d_model, r2plus1d, resnet3D, transformer_v3

from criterion import SetVACriterion

SEED = 23333
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

run_id = 10
split = "Test_Set" # Submission_Set Test_Set
resume_epoch = 1000  # Default is 0, change if want to resume
LAYER_SIZES = (1, 2, 2, 1)
useTest = False # See evolution of the test set when training
nTestInterval = 20 # Run on test set every nTestInterval epochs
snapshot = 1500 # Store a model every snapshot epochs
lr = 1e-5
ADD_LANDMARKS = False
CLIP_LEN = 8
FRAME_STRIDE = 4
RESIZE_WIDTH = 112
RESIZE_HEIGHT = 112

STEP_SIZE = 1000
VIS_DATA = True
VIS_GRAD = True

modelName = 'Resnet3d'   # Options: C3DVA or R2Plus1D or R3D or CSN Resnet3d

dataset = 'ABAW' 

if dataset == 'hmdb51':
    num_classes=51
elif dataset == 'ucf101':
    num_classes = 101
elif dataset == 'affwild2' or dataset == 'ABAW':
    num_classes = 2
else:
    print('We only implemented hmdb and ucf datasets.')
    raise NotImplementedError

#save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
save_dir_root = '/media/ubuntu/bak/VA'
# save_dir_root = '/public/home/fanyachun/code/VA_regressor/'

exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]


save_dir = os.path.join(save_dir_root, 'experiments', 'run_' + str(run_id))


saveName = modelName + '-' + dataset

def test_model(dataset=dataset, save_dir=save_dir, num_classes=num_classes, lr=lr,
                save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval):
    """
        Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
    """

    if modelName == 'C3DVA':
        model = C3DVA_model.C3DVA(num_classes=2, pretrained=True)
        train_params = model.parameters()
    elif modelName == 'CSN':
        model = CSN_model.csn26(num_classes=2, add_landmarks=ADD_LANDMARKS)
        train_params = model.parameters()
    elif modelName == 'R2Plus1D':
        model = R2Plus1D_model.R2Plus1DRegressor(num_classes=2, layer_sizes=LAYER_SIZES)
        train_params = [{'params': R2Plus1D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': R2Plus1D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif modelName == 'Transformer':
        model = transformer_v3.Semi_Transformer(num_classes=40, seq_len=CLIP_LEN)
        train_params = model.parameters()
    elif modelName == 'Resnet3d':
        model = resnet3D.resnet3d18(num_classes=400, pretrained=None)
        train_params = model.parameters()

    checkpoint = torch.load(os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar'),
                       map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
    print("Initializing weights from: {}...".format(
        os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)

    print('Training model on {} dataset...'.format(dataset))
    test_dataloader  = DataLoader(VideoDataset(dataset=dataset, label_type="VA_Set", split=split, \
        clip_len=CLIP_LEN, stride=FRAME_STRIDE, add_landmarks=ADD_LANDMARKS, triplet_label=True), batch_size=1, shuffle=False, num_workers=0,\
         drop_last=True, pin_memory=False)
    test_size = len(test_dataloader.dataset)

    # testing
    model.eval()
    start_time = timeit.default_timer()
    criterion = SetVACriterion(num_classes=20, use_mse=False, is_test=True)
    criterion.to(device)

    val_cccs = []
    aro_cccs = []
    val_mses = []
    aro_mses = []
    with open('test_log.txt', 'a') as test_log:
        test_log.write('=======start=======\n')
        test_log.write('model path: ' + 
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar') + '\n')

    for frames, label_path in (test_dataloader):
        
        print(frames[0][0].split('/')[-2])

        frame_ids = [] # 图片对应的帧号
        for each in frames:
            fid = int(each[0].split('/')[-1].split('.')[0])
            frame_ids.append(fid)

        frame_count = len(frames)
        frame_root_dir = '/' + os.path.join(*frames[0][0].split('/')[1:-1])
        
        # valence_label = np.empty((frame_count-CLIP_LEN*FRAME_STRIDE+1, 1), np.dtype('float32'))
        # arousal_label = np.empty((frame_count-CLIP_LEN*FRAME_STRIDE+1, 1), np.dtype('float32'))
        # preds_all = np.empty((frame_count-CLIP_LEN*FRAME_STRIDE+1, 2), np.dtype('float32'))

        valence_label = np.zeros((frame_ids[-1], 1), np.dtype('float32'))
        arousal_label = np.zeros((frame_ids[-1], 1), np.dtype('float32'))
        preds_all = np.zeros((frame_ids[-1], 2), np.dtype('float32'))

        
        '''读取视频的所有帧'''
        for center in tqdm( frame_ids ):
            if ADD_LANDMARKS:
                buffer = np.empty(( CLIP_LEN, RESIZE_HEIGHT, RESIZE_WIDTH, 4), np.dtype('float32'))
            else:
                buffer = np.empty(( CLIP_LEN, RESIZE_HEIGHT, RESIZE_WIDTH, 3), np.dtype('float32'))
        
            
            '''读取图像'''
            clip = list(range(center-int(CLIP_LEN/2)*FRAME_STRIDE, center+int(CLIP_LEN/2)*FRAME_STRIDE, FRAME_STRIDE))
            
            for i, frame_id in enumerate(clip):
                frame_name = None
                if (frame_id < 1 or frame_id > frame_ids[-1]):
                    frame = np.ones((RESIZE_HEIGHT, RESIZE_WIDTH, 3))*128
                else:
                    frame_name = os.path.join(frame_root_dir,str(frame_id).zfill(5)+'.jpg')
                    #print(center, frames[i])
                    
                    if os.path.exists(frame_name):
                        frame = np.array(cv2.imread(frame_name)).astype(np.float32)
                    else:
                        frame = np.ones((RESIZE_HEIGHT, RESIZE_WIDTH, 3))*128

                    frame = cv2.resize(frame, (RESIZE_HEIGHT, RESIZE_WIDTH))
                # cv2.imshow(frame_name, frame/256)
                # cv2.waitKey(1000)
                # cv2.destroyAllWindows()

                if ADD_LANDMARKS:
                    # 读landmarks
                    if frame_name != None:
                        lm_name = frame_name.replace('image', 'landmarks')
                        lm_img = np.array(cv2.imread(lm_name, cv2.IMREAD_GRAYSCALE)).astype(np.float32)
                        lm_img = cv2.resize(lm_img, (RESIZE_HEIGHT, RESIZE_WIDTH))
                        lm_img = np.expand_dims(lm_img, 2)
                    else:
                        lm_img = np.zeros((RESIZE_HEIGHT, RESIZE_WIDTH, 1))
                    # 把landmark拼接上
                    buffer[i] = np.concatenate((frame, lm_img),2)
                else:
                    buffer[i] = frame
            buffer = (buffer - 128) / 128
            buffer = buffer.transpose((3, 0, 1, 2))
            inputs = torch.from_numpy(buffer).unsqueeze(0)

            # if not split == 'Submission_Set':
            #     avgface_path = frame_root_dir.replace('image', 'avgfaces')
            #     avgface_path = os.path.join(avgface_path, 'avgface.png')
            #     avgface = cv2.imread(avgface_path, cv2.IMREAD_GRAYSCALE)
            #     avgface = np.stack((avgface,)*3, axis=-1)
            #     avgface = (avgface - 128) / 128
            #     avgface = torch.from_numpy(avgface.transpose(2,0,1)).float().unsqueeze(0)

            if not split == 'Submission_Set':
                # 取label
                with open(label_path[0], 'rt') as f:
                    lines = f.read().splitlines()
                    line = lines[center] #去掉第一行
                valence_label[center-1, 0] = float(line.split(',')[0])
                arousal_label[center-1, 0] = float(line.split(',')[1])

            with torch.no_grad():
                # 方式1
                outputs_va, outputs_expr, _, _ = model(inputs.cuda())
                outputs_v = outputs_va[:,:20]
                outputs_a = outputs_va[:,20:40]
                _, valence_pred_reg, arousal_pred_reg = criterion(outputs_v, outputs_a, \
                    valence_label, arousal_label)

                # 方式2
                # logits1, logits2, logits3, corr_t_div_C, corr_s_div_C, feat = model(inputs.cuda())
                # outputs_v2, outputs_a2 = logits2[0][:,:20], logits2[0][:,20:40]
                # _, valence_pred_reg2, arousal_pred_reg2 = criterion(outputs_v2, outputs_a2, \
                #         valence_label, arousal_label)
                # valence_pred_reg = valence_pred_reg2
                # arousal_pred_reg = arousal_pred_reg2

                # visualize_feature(feat, center)

            # visualize_attention(inputs, corr_t_div_C, corr_s_div_C)

            pred_concat = np.concatenate((np.expand_dims(valence_pred_reg,0), np.expand_dims(arousal_pred_reg,0)), 1)
            preds_all[center-1, :] = pred_concat

            # 将无效label去掉
            if valence_label[center-1, 0] == -5:
                preds_all[center-1, :] = -5

        preds_all_valid = []
        valence_label_valid = []
        arousal_label_valid = []
        for each in preds_all:
            if not each[0] == -5:
                preds_all_valid.append(each)

        for each in valence_label:
            if not each[0] == -5:
                valence_label_valid.append(each)

        for each in arousal_label:
            if not each[0] == -5:
                arousal_label_valid.append(each)

        print(len(preds_all_valid), len(valence_label_valid), len(arousal_label_valid))

        preds_all = torch.from_numpy(np.array(preds_all_valid))
        valence_label = torch.from_numpy(np.array(valence_label_valid))
        arousal_label = torch.from_numpy(np.array(arousal_label_valid))

        val_cc2 = concord_cc2(preds_all[:,0] , valence_label[:,0])
        aro_cc2 = concord_cc2(preds_all[:,1], arousal_label[:,0])
        val_mse = torch.nn.MSELoss()(preds_all[:,0], valence_label[:,0])
        aro_mse = torch.nn.MSELoss()(preds_all[:,1], arousal_label[:,0])


        if not split == 'Submission_Set':
            if not os.path.exists(os.path.join(save_dir, 'res')):
                os.mkdir(os.path.join(save_dir, 'res'))
            with open(os.path.join(save_dir, 'res', 'res_' + frames[0][0].split('/')[-2] + '.txt'), 'w') as res:
                for i, each in enumerate(preds_all):
                    res.write(str(preds_all.numpy()[i,0]) + ", " + str(valence_label.numpy()[i,0]) + ', ' \
                    + str(preds_all.numpy()[i,1]) + ", " + str(arousal_label.numpy()[i,0]) + '\n')
        
        if split == 'Submission_Set':
            if not os.path.exists(os.path.join(save_dir, 'submission')):
                os.mkdir(os.path.join(save_dir, 'submission'))
            with open(os.path.join(save_dir, 'submission', frames[0][0].split('/')[-2] + '.txt'), 'w') as res:
                res.write("valence,arousal\n")
                for i, each in enumerate(preds_all):
                    res.write(str(preds_all.numpy()[i,0]) + "," + str(preds_all.numpy()[i,1]) + '\n')
        
            
        print("Val CCC: {:.4f}  Aro CCC: {:.4f}  Val MSE: {:.4f}  Aro MSE: {:.4f}".format(val_cc2, aro_cc2, val_mse, aro_mse))

        with open('test_log.txt', 'a') as test_log:
            log = frames[0][0].split('/')[-2] + ',' + str(float(val_cc2)) + ',' + str(float(aro_cc2)) + ',' + \
                str(float(val_mse)) + ',' + str(float(aro_mse))
            test_log.write(log + '\n')

        val_cccs.append(val_cc2)
        aro_cccs.append(aro_cc2)
        val_mses.append(val_mse)
        aro_mses.append(aro_mse)
    print("[test] Val CCC: {:.4f}  Aro CCC: {:.4f}  Val MSE: {:.4f}  Aro MSE: {:.4f}".format(np.mean(val_cccs), \
       np.mean(aro_cccs), np.mean(val_mses), np.mean(aro_mses)))

    with open('test_log.txt', 'a') as test_log:
            test_log.write(("[test] Val CCC: {:.4f}  Aro CCC: {:.4f}  Val MSE: {:.4f}  Aro MSE: {:.4f}\n".format(np.mean(val_cccs), \
       np.mean(aro_cccs), np.mean(val_mses), np.mean(aro_mses))))
    stop_time = timeit.default_timer()
    print("Execution time: " + str(stop_time - start_time) + "\n")
    
def visualize_feature(feat, idx):
    output_root = "./feature_visualization"
    _, c, t, h, w = feat.shape
    print(feat.shape)
    frame1 = feat[0,:,0,:,:].cpu().detach().numpy()
    frame1 = np.mean(frame1, axis=0) * 255
    frame1 = np.resize(frame1, (28, 28))
    frame1 = (frame1 - np.min(frame1)) / (np.max(frame1) - np.min(frame1) + 1e-8)
    
    cv2.imshow("f1", frame1)
    cv2.waitKey(1000)


    # cv2.imwrite(os.path.join(output_root, "frame1_" + str(idx)))


def visualize_attention(inputs, att_map_temporal, att_map_spatial, clip_len=8):
    imgs = [(x * 128 + 128) / 255 for x in inputs[0].permute(1, 2, 3, 0).detach().cpu().numpy()]
    img = imgs[-1]
    hstack_img = np.hstack(imgs)

    # visualize Temporal Attention
    tmap = att_map_temporal[0].detach().cpu().numpy()
    tmap = tmap**2 / np.max(tmap)**2
    temporal_active_index = np.argsort(tmap[3])[-4:]
    sorted_index = np.argsort(tmap[3])
    print(sorted_index)
    for idx, i in enumerate(temporal_active_index):
        tsrc = i // clip_len
        ttgt = i % clip_len
        cv2.line(hstack_img, (3 * RESIZE_HEIGHT + 46, 10*(idx+1)), (i * RESIZE_HEIGHT + 66, 10*(idx+1)), (120, 120, 120), 2)


    cv2.imshow("test", hstack_img)
    cv2.waitKey(100000)

    # visualize Spacial Attention
    # smap = att_map_spatial[0].detach().cpu().numpy()
    # spatial_active_index = np.argsort(smap.flatten())[-20:]

    # for i in spatial_active_index:
    #     src = i // 49 # source position
    #     tgt = i % 49  # target position
    #     x1 = int((src // 7) * (RESIZE_HEIGHT//7))
    #     y1 = int((src % 7) * (RESIZE_HEIGHT//7))
    #     x2 = int((tgt // 7) * (RESIZE_HEIGHT//7))
    #     y2 = int((tgt % 7) * (RESIZE_HEIGHT//7))
    #     cv2.line(img, (x1, y1), (x2, y2), (120, 120, 120))
    # cv2.imshow("test", img)
    # cv2.waitKey(1000)


## 计算一致性相关系数CCC
def concord_cc2(r1, r2):
	mean_pred = torch.mean((r1 - torch.mean(r1))*(r2 - torch.mean(r2)))
	return (2*mean_pred)/(torch.var(r1) + torch.var(r2) + (torch.mean(r1)- torch.mean(r2))**2)

if __name__ == "__main__":
    test_model()