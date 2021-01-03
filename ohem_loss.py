import torch
from torch import nn
from torch.autograd import Variable

class topk_crossEntrophy(nn.Module):
    def __init__(self, top_k=0.7):
        super(topk_crossEntrophy, self).__init__()
        self.loss = nn.NLLLoss()
        self.top_k = top_k
        self.softmax = nn.LogSoftmax()
        return
        
    def forward(self, input, target):
        softmax_result = self.softmax(input)
        
        loss = Variable(torch.Tensor(1).zero_()).cuda()
        for idx, row in enumerate(softmax_result):
            gt = target[idx]
            pred = torch.unsqueeze(row, 0)
            gt = torch.unsqueeze(gt, 0)
            cost = self.loss(pred, gt)
            loss = torch.cat((loss, cost.unsqueeze(0)), 0)
            
        loss = loss[1:]
        if self.top_k == 1:
            valid_loss = loss
            
        index = torch.topk(loss, int(self.top_k * loss.size()[0]))
        valid_loss = loss[index[1]]
            
        return torch.mean(valid_loss)