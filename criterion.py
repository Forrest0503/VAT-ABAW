import numpy as np
import torch
from torch import nn

class SetVACriterion(nn.Module):
    def __init__(self, num_classes=20, use_mse=False, is_test=False):
        super().__init__()
        self.num_classes = num_classes
        # self.vector_c_valence = np.array([-0.9, -0.65, -0.4, -0.225, -0.1, -0.025, 0.025, 0.075, 0.125, 0.175, 
        #         0.225, 0.275, 0.325, 0.375, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
        # self.vector_c_arousal = np.array([-0.9, -0.65, -0.425, -0.275, -0.15, -0.05, 0.025, 0.075, 0.125, 0.175, 
        #         0.225, 0.275, 0.325, 0.375, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
        self.vector_c_valence = np.array([-0.95, -0.85, -0.75, -0.65, -0.55, -0.45, -0.35, -0.25, -0.15, -0.05, 
                0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
        self.vector_c_arousal = np.array([-0.95, -0.85, -0.75, -0.65, -0.55, -0.45, -0.35, -0.25, -0.15, -0.05, 
                0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
        self.use_mse = use_mse
        self.is_test = is_test

    def forward(self, outputs_v, outputs_a, valence_label, arousal_label):
        """ This performs the loss computation.
        Parameters:
             outputs_va: tensor, [batch, 40]
             valence_label: list of float, [batch]
             arosal_label: list of float, [batch]
        """
        valence_pred_cls = outputs_v
        arousal_pred_cls = outputs_a
        valence_pred_reg = np.zeros((valence_pred_cls.shape[0],))
        arousal_pred_reg = np.zeros((arousal_pred_cls.shape[0],))
        # valence_pred_idx = torch.argmax(valence_pred_cls, dim=1)
        # arousal_pred_idx = torch.argmax(arousal_pred_cls, dim=1)

        for i in range(arousal_pred_cls.shape[0]):
            valence_pred_cls_softmax = nn.Softmax(dim=0)(valence_pred_cls[i])
            arousal_pred_cls_softmax = nn.Softmax(dim=0)(arousal_pred_cls[i])
            valence_pred_reg[i] = np.dot(self.vector_c_valence, valence_pred_cls_softmax.detach().cpu().numpy().T)
            arousal_pred_reg[i] = np.dot(self.vector_c_arousal, arousal_pred_cls_softmax.detach().cpu().numpy().T)

        def valence_to_cls(valence):
            c = 0
            if valence >= -1.0 and valence < -0.8:
                c = 0
            elif valence >= -0.8 and valence < -0.5:
                c = 1
            elif valence >= -0.5 and valence < -0.3:
                c = 2
            elif valence >= -0.3 and valence < -0.15:
                c = 3
            elif valence >= -0.15 and valence < -0.05:
                c = 4
            elif valence >= -0.05 and valence < 0.0:
                c = 5
            elif valence >= 0.0 and valence < 0.05:
                c = 6
            elif valence >= 0.05 and valence < 0.1:
                c = 7
            elif valence >= 0.1 and valence < 0.15:
                c = 8
            elif valence >= 0.15 and valence < 0.2:
                c = 9
            elif valence >= 0.2 and valence < 0.25:
                c = 10
            elif valence >= 0.25 and valence < 0.3:
                c = 11
            elif valence >= 0.3 and valence < 0.35:
                c = 12
            elif valence >= 0.35 and valence < 0.4:
                c = 13
            elif valence >= 0.4 and valence < 0.5:
                c = 14
            elif valence >= 0.5 and valence < 0.6:
                c = 15
            elif valence >= 0.6 and valence < 0.7:
                c = 16
            elif valence >= 0.7 and valence < 0.8:
                c = 17
            elif valence >= 0.8 and valence < 0.9:
                c = 18
            elif valence >= 0.9 and valence <= 1.0:
                c = 19
            return c

        def arousal_to_cls(arousal):
            c = 0
            if arousal >= -1.0 and arousal < -0.8:
                c = 0
            elif arousal >= -0.8 and arousal < -0.5:
                c = 1
            elif arousal >= -0.5 and arousal < -0.35:
                c = 2
            elif arousal >= -0.35 and arousal < -0.2:
                c = 3
            elif arousal >= -0.2 and arousal < -0.1:
                c = 4
            elif arousal >= -0.1 and arousal < 0.0:
                c = 5
            elif arousal >= 0.0 and arousal < 0.05:
                c = 6
            elif arousal >= 0.05 and arousal < 0.1:
                c = 7
            elif arousal >= 0.1 and arousal < 0.15:
                c = 8
            elif arousal >= 0.15 and arousal < 0.2:
                c = 9
            elif arousal >= 0.2 and arousal < 0.25:
                c = 10
            elif arousal >= 0.25 and arousal < 0.3:
                c = 11
            elif arousal >= 0.3 and arousal < 0.35:
                c = 12
            elif arousal >= 0.35 and arousal < 0.4:
                c = 13
            elif arousal >= 0.4 and arousal < 0.5:
                c = 14
            elif arousal >= 0.5 and arousal < 0.6:
                c = 15
            elif arousal >= 0.6 and arousal < 0.7:
                c = 16
            elif arousal >= 0.7 and arousal < 0.8:
                c = 17
            elif arousal >= 0.8 and arousal < 0.9:
                c = 18
            elif arousal >= 0.9 and arousal <= 1.0:
                c = 19
            return c

        if self.is_test: # On test set, dont need to compute loss
            return 0, valence_pred_reg, arousal_pred_reg

        valence_label_cls = []
        arousal_label_cls = []
        for each in valence_label:
            valence_label_cls.append(valence_to_cls(each))
        valence_label_cls = np.array(valence_label_cls)
        
        for each in arousal_label:
            arousal_label_cls.append(arousal_to_cls(each))
        arousal_label_cls = np.array(arousal_label_cls)

        '''CE loss'''
        loss_valence = nn.CrossEntropyLoss()(valence_pred_cls, torch.from_numpy(valence_label_cls).long().cuda())
        loss_arousal = nn.CrossEntropyLoss()(arousal_pred_cls, torch.from_numpy(arousal_label_cls).long().cuda())

        if not self.use_mse:
            alpha = 0.2
            beta = 0.8
            loss = alpha * (loss_valence) + beta * (loss_arousal) / \
                ((alpha + beta) / 2)
        else:
            mse_valence = nn.MSELoss()(torch.from_numpy(valence_pred_reg).cuda(), valence_label)
            mse_arousal = nn.MSELoss()(torch.from_numpy(arousal_pred_reg).cuda(), arousal_label)
            loss = loss_valence + loss_arousal + mse_valence + mse_arousal

        return loss, valence_pred_reg, arousal_pred_reg