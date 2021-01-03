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

from dataloaders.ABAW import VideoDataset
from dataloaders.gkd import DataPrefetcher
from network import C3DVA_model, R2Plus1D_model, R2Plus1D_model_ln, R2Plus1D_model_in, R3D_model, CSN_model, \
    Resnet3d_model, r2plus1d, resnet3D, transformer_v3

import ohem_loss
from criterion import SetVACriterion


'''
R2+1D:
1. relu -> tanh
2. bias = False
3. add Linear
4. random init
5. dropout

6. pooling

7. stepsize = 350
8. dropout
9. L=8
10. weight decay
11. AvgPool


c3d:
1. 减少pooling
2. attention
'''

SEED = 2333
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

nEpochs = 4000  # Number of epochs for training
resume_epoch = 0  # Default is 0, change if want to resume
snapshot = 500 # Store a model every snapshot epochs 
lr = 1e-4 # Learning rate
WD = 1e-4
CLIP_LEN = 8
FRAME_STRIDE = 4
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_VAL = 4
VIS_DATA = False
WILL_LR_DECAY = True
STEP_SIZE = 400
ADD_LANDMARKS = False
DEBUG = False
USE_MINI_DATASET = False
CHECK_DIRTY_LIST = False
LAYER_SIZES = (2, 2, 2, 2)

modelName = 'Resnet3d' # Options: C3DVA or R2Plus1D or R2Plus1DLn or Resnet3d or CSN or Transformer

dataset = 'ABAW' # Options: hmdb51 or ucf101

if dataset == 'hmdb51':
    num_classes=51
elif dataset == 'ucf101':
    num_classes = 101
elif dataset == 'ABAW':
    num_classes = 2
else:
    print('We only implemented hmdb and ucf datasets.')
    raise NotImplementedError

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
expr_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = 3 # 自己修改
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))


saveName = modelName + '-' + dataset

def train_model(dataset=dataset, save_dir=save_dir, num_classes=num_classes, lr=lr,
                num_epochs=nEpochs, save_epoch=snapshot, useTest=False, test_interval=0):
    """
        Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
    """

    if modelName == 'C3DVA':
        model = C3DVA_model.C3DVA(num_classes=2, pretrained=True)
        train_params = [{'params': C3DVA_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': C3DVA_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif modelName == 'R2Plus1D':
        model = r2plus1d.r2plus1d18(num_classes=9)
        train_params = model.parameters()
    elif modelName == 'R2Plus1DLn':
        model = R2Plus1D_model_in.R2Plus1DRegressor(num_classes=2, layer_sizes=LAYER_SIZES, pretrained=False)
        train_params = model.parameters()
    elif modelName == 'CSN':
        model = CSN_model.csn26(num_classes=2, mode='ip', add_landmarks=ADD_LANDMARKS)
        train_params = model.parameters()
    elif modelName == 'Resnet3d':
        model = resnet3D.resnet3d18(num_classes=400)
        train_params = model.parameters()
        # fast_lr_params = resnet3D.get_fast_lr_params(model)
        # slow_lr_params = resnet3D.get_slow_lr_params(model)
    elif modelName == 'Transformer':
        model = transformer_v3.Semi_Transformer(num_classes=40, seq_len=CLIP_LEN)
        train_params = model.parameters()
    else:
        print('We only implemented C3D and R2Plus1D models.')
        raise NotImplementedError
    # criterion = nn.MSELoss() 
    criterion = SetVACriterion(num_classes=20, use_mse=True)
    ccc_criterion = ccc_loss()
    ce_criterion = nn.CrossEntropyLoss()
    ohem_criterion = ohem_loss.topk_crossEntrophy()
    optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=WD)
    # optimizer = torch.optim.SGD([
    #         {'params': slow_lr_params, 'lr': lr * 100},
    #         {'params': fast_lr_params, 'lr': lr * 100}], lr=lr, momentum=0.9, weight_decay=WD)

    # print(optimizer.state_dict)
    
    #optimizer = optim.Adam(train_params, lr=lr, weight_decay=WD)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE,gamma=0.33) 
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=128)

    if resume_epoch == 0:
        print("Training {} from scratch...".format(modelName))
    else:
        checkpoint = torch.load(os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar'),
                       map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
        model.load_state_dict(checkpoint['state_dict'])
        #optimizer.load_state_dict(checkpoint['opt_dict'])
        #for state in optimizer.state.values():
        #    for k, v in state.items():
        #        if isinstance(v, torch.Tensor):
        #            state[k] = v.cuda()

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)
    criterion.to(device)
    # ccc_loss.to(device)
    ce_criterion.to(device)
    ohem_criterion.to(device)
    model.cuda()

    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    print('Training model on {} dataset...'.format(dataset))
    train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='Training_Set', clip_len=CLIP_LEN, \
        stride=FRAME_STRIDE, add_landmarks=ADD_LANDMARKS, mini=USE_MINI_DATASET, check_dirty_list=CHECK_DIRTY_LIST, \
            triplet_label=True), batch_size=BATCH_SIZE_TRAIN, \
            shuffle=True, num_workers=4,drop_last=True)
    val_dataloader = DataLoader(VideoDataset(dataset=dataset, split='Validation_Set', clip_len=CLIP_LEN, \
        stride=FRAME_STRIDE, add_landmarks=ADD_LANDMARKS, mini=USE_MINI_DATASET, \
            triplet_label=True), batch_size=BATCH_SIZE_VAL, \
            shuffle=False, num_workers=4,drop_last=True)
    #test_dataloader  = DataLoader(VideoDataset(dataset=dataset, split='test', clip_len=CLIP_LEN, add_landmarks=ADD_LANDMARKS), batch_size=1, num_workers=0,drop_last=True)

    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    #test_size = len(test_dataloader.dataset)
    trainval_loss = {'train': 0.0, 'val': 0.0}
    trainval_loss_va = {'train': 0.0, 'val': 0.0}
    # trainval_loss_expr = {'train': 0.0, 'val': 0.0}

    for epoch in range(resume_epoch, num_epochs):
        # 构建Datarefetcher
        train_fetcher = DataPrefetcher(train_dataloader)
        val_fetcher = DataPrefetcher(val_dataloader)
        trainval_loaders = {'train': train_fetcher, 'val': val_fetcher}

        # each epoch has a training and validation step
        for phase in ['train', 'val']:
            start_time = timeit.default_timer()

            # reset the running loss and corrects
            running_loss = 0.0
            running_loss_va = 0.0
            # running_loss_expr = 0.0

            # set model to train() or eval() mode depending on whether it is trained
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.
            if phase == 'train':
                # scheduler.step() is to be called once every epoch during training
                model.train()
            else:
                model.eval()
            print("====================%s PHASE START====================" % ('TRAIN' if phase == 'train' else 'VALIDATION'))
            # for batch_idx, (inputs, valence_label, arousal_label, _, filename, frame_index_list) in enumerate(trainval_loaders[phase]):
            for batch_idx in range(len(trainval_loaders[phase])):
                inputs, valence_label, arousal_label = trainval_loaders[phase].next()

                # move inputs and labels to the device the training is taking place on
                # inputs = Variable(inputs, requires_grad=False).to(device)
                # valence_label1 = Variable(valence_label[0]).float().to(device)
                # arousal_label1 = Variable(arousal_label[0]).float().to(device)
                # valence_label2 = Variable(valence_label[1]).float().to(device)
                # arousal_label2 = Variable(arousal_label[1]).float().to(device)
                # valence_label3 = Variable(valence_label[2]).float().to(device)
                # arousal_label3 = Variable(arousal_label[2]).float().to(device)
                
                #expr_label = Variable(expr_label_onehot[:,0:7]).to(device)
                #expr_mask = Variable(expr_label_onehot[:,7]).to(device)
                # expr_label = Variable(expr_label).to(device).unsqueeze(1)
                # expr_mask = Variable(expr_label != -1).float().to(device)

                optimizer.zero_grad()
                
                if phase == 'train':
                    outputs_va, _, _ = model(inputs)
                    outputs_v = outputs_va[:,:20]
                    outputs_a = outputs_va[:,20:40]
                    loss_va, valence_pred_reg, arousal_pred_reg = criterion(outputs_v, outputs_a, \
                        valence_label, arousal_label)
                else:
                    with torch.no_grad():
                        outputs_va, _, _ = model(inputs)
                        outputs_v = outputs_va[:,:20]
                        outputs_a = outputs_va[:,20:40]
                        loss_va, valence_pred_reg, arousal_pred_reg = criterion(outputs_v, outputs_a, \
                            valence_label, arousal_label)

                # if phase == 'train':
                #     logits1, logits2, logits3, corr_t_div_C, corr_s_div_C, feat = model(inputs)
                #     # visualize_feature(feat, 1)
                #     outputs_v1, outputs_a1 = logits1[0][:,:20], logits1[0][:,20:40]
                #     loss_va1, valence_pred_reg1, arousal_pred_reg1 = criterion(outputs_v1, outputs_a1, \
                #         valence_label1, arousal_label1)
                #     outputs_v2, outputs_a2 = logits2[0][:,:20], logits2[0][:,20:40]
                #     loss_va2, valence_pred_reg2, arousal_pred_reg2 = criterion(outputs_v2, outputs_a2, \
                #         valence_label2, arousal_label2)
                #     outputs_v3, outputs_a3 = logits3[0][:,:20], logits3[0][:,20:40]
                #     loss_va3, valence_pred_reg3, arousal_pred_reg3 = criterion(outputs_v3, outputs_a3, \
                #         valence_label3, arousal_label3)
                #     outputs_expr = logits2[1]

                # else:
                #     with torch.no_grad():
                #         logits1, logits2, logits3, corr_t_div_C, corr_s_div_C, feat = model(inputs)
                #         outputs_v1, outputs_a1 = logits1[0][:,:20], logits1[0][:,20:40]
                #         loss_va1, valence_pred_reg1, arousal_pred_reg1 = criterion(outputs_v1, outputs_a1, \
                #             valence_label1, arousal_label1)
                #         outputs_v2, outputs_a2 = logits2[0][:,:20], logits2[0][:,20:40]
                #         loss_va2, valence_pred_reg2, arousal_pred_reg2 = criterion(outputs_v2, outputs_a2, \
                #             valence_label2, arousal_label2)
                #         outputs_v3, outputs_a3 = logits3[0][:,:20], logits3[0][:,20:40]
                #         loss_va3, valence_pred_reg3, arousal_pred_reg3 = criterion(outputs_v3, outputs_a3, \
                #             valence_label3, arousal_label3)
                #         outputs_expr = logits2[1]


                '''EXPR CE Loss'''
                # outputs_expr_masked = outputs_expr*expr_mask
                # for i in range(outputs_expr_masked.shape[0]):
                #     if expr_mask[i] == 0:
                #         outputs_expr_masked[i,0] = 10
                # expr_label_masked = expr_label.long().cuda()*expr_mask.long()
                # loss_expr = ce_criterion(outputs_expr_masked, expr_label_masked.squeeze(1))
                # loss_va = (loss_va1 + loss_va2 + loss_va3) / 3
                # loss = 0.6*loss_va + 0.4*loss_expr

                '''Multi-Task Loss Funtion'''
                loss = loss_va

                '''L1 Norm'''
                # L1_reg = 0
                # for param in model.parameters():
                #     L1_reg += torch.sum(torch.abs(param))
                # loss += 1e-7 * L1_reg  # lambda=0.001

                if phase == 'train':
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01) # 梯度裁减，避免梯度爆炸
                    optimizer.step()


                for i in range(inputs.shape[0]):
                    np.set_printoptions(precision=2)

                    # print("v_out: %.3f %.3f %.3f  v_label: %.3f    a_out: %.3f %.3f %.3f   a_label: %.3f" % \
                    #     ( valence_pred_reg1[i], valence_pred_reg2[i], valence_pred_reg3[i], \
                    # float(valence_label2[i]), arousal_pred_reg1[i], arousal_pred_reg2[i], arousal_pred_reg3[i], float(arousal_label2[i])))
                   
                    print("v_out: %.3f    v_label: %.3f    a_out: %.3f   a_label: %.3f" % ( float(valence_pred_reg[i]), \
                    float(valence_label[i]), float(arousal_pred_reg[i]), float(arousal_label[i])))
                    
                    # print("expr_out:            ", outputs_expr[i].detach().cpu().numpy())
                    # print("expr_masked:         ", outputs_expr_masked[i].detach().cpu().numpy())
                    # print("expr_label:          ", expr_label[i].detach().cpu().numpy())
                    # print("expr_label_masked:   ", expr_label_masked[i].detach().cpu().numpy())
                    # print("expr_loss:           ", loss_expr.item())

                print("/////// EPOCH:%d BATCH END (%d/%d) LOSS = %.4f LOSS_VA = %.4f ///////" \
                    % (epoch+1, batch_idx+1, len(trainval_loaders[phase]), trainval_loss[phase], trainval_loss_va[phase]))
                # print("/////// LOSS_VALENCE = %.4f LOSS_AROUSAL = %.4f ///////" \
                #     % (loss_valence.item(), loss_arousal.item()))
                running_loss += loss.item() * inputs.size(0)
                running_loss_va += loss_va.item() * inputs.size(0)
                # running_loss_expr += loss_expr.item() * inputs.size(0)
                # running_corrects += torch.sum(outputs_va == labels.data)

            if WILL_LR_DECAY and phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_loss_va = running_loss_va / trainval_sizes[phase]
            # epoch_loss_expr = running_loss_expr / trainval_sizes[phase]
            trainval_loss[phase] = epoch_loss
            trainval_loss_va[phase] = epoch_loss_va
            # trainval_loss_expr[phase] = epoch_loss_expr

            if phase == 'train':
                writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/train_loss_epoch_VA', epoch_loss_va, epoch)
                # writer.add_scalar('data/train_loss_epoch_EXPR', epoch_loss_expr, epoch)
            else:
                writer.add_scalar('data/val_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/val_loss_epoch_VA', epoch_loss_va, epoch)
                # writer.add_scalar('data/val_loss_epoch_EXPR', epoch_loss_expr, epoch)

            print("[{}] Epoch: {}/{} Loss: {}".format(phase, epoch+1, nEpochs, epoch_loss))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")
            print("====================%s PHASE END====================" % ('TRAIN' if phase == 'train' else 'VALIDATION'))

        if epoch % save_epoch == (save_epoch - 1):
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar'))
            print("Save model at {}\n".format(os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar')))


    writer.close()


def visualize_feature(feat, idx):
    output_root = "./feature_visualization"
    _, c, t, h, w = feat.shape
    print(feat.shape)
    frame1 = feat[0,:,0,:,:].cpu().detach().numpy()
    frame1 = np.mean(frame1, axis=0) * 255
    frame1 = np.resize(frame1, (8, 8))
    frame1 = (frame1 - np.min(frame1)) / (np.max(frame1) - np.min(frame1))
    
    cv2.imshow("frame1", frame1)
    cv2.waitKey(10000)


## 计算一致性相关系数CCC
def concord_cc2(r1, r2): # pred, label
	mean_pred = torch.mean((r1 - torch.mean(r1))*(r2 - torch.mean(r2)))
	return (2*mean_pred)/(torch.var(r1) + torch.var(r2) + (torch.mean(r1)- torch.mean(r2))**2 + 1e-8)

class ccc_loss(nn.Module):
    def __init__(self):
        super(ccc_loss, self).__init__()
        
    def forward(self, r1, r2):
        return 1 - concord_cc2(r1, r2)


if __name__ == "__main__":
    train_model()