import torch
import torch.nn as nn
from mypath import Path
from network.non_local_embedded_gaussian import NONLocalBlock3D

class C3DVA(nn.Module):
    """
    The C3DVA network.
    """
    GROUP = 1
    CLIP_LEN = 8

    def __init__(self, num_classes, pretrained=False):
        super(C3DVA, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)) #64 4 56 56
        self.bn1 = nn.BatchNorm3d(64)

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)) #128 4 28 28
        self.bn2 = nn.BatchNorm3d(128)


        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)) #256 4 14 14

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)) #512 4 7 7

        
        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(1, 7, 7), stride=(1, 1, 1)) # 512 4 1 1
        self.alpha = nn.Linear(512, 1)
        self.beta = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

        self.non_local5 = NONLocalBlock3D(512)


        self.fc6 = nn.Linear(512, 128)

        self.fc7_va = nn.Linear(128 * 4, 40)
        self.fc7_expr = nn.Linear(128 * 4, 7)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        self.tanh = nn.Tanh()

        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.relu(self.conv1(x))
        x = self.pool1(x)
                
        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)


        x = self.relu(self.conv4a(x))
        #x = self.relu(self.conv4b(x))
        x = self.pool4(x)
        
        
        x = self.relu(self.conv5a(x))
        #x = self.relu(self.conv5b(x))
        #x = self.non_local5(x)
        x = self.pool5(x)
        
        x1 = x[:,:,0,:,:].view(batch_size,-1)
        x2 = x[:,:,1,:,:].view(batch_size,-1)
        x3 = x[:,:,2,:,:].view(batch_size,-1)
        x4 = x[:,:,3,:,:].view(batch_size,-1)

        alpha1 = self.sigmoid(self.alpha(x1))
        alpha2 = self.sigmoid(self.alpha(x2))
        alpha3 = self.sigmoid(self.alpha(x3))
        alpha4 = self.sigmoid(self.alpha(x4))

        x1 = x1 * alpha1
        x2 = x2 * alpha2
        x3 = x3 * alpha3
        x4 = x4 * alpha4

        xs = (x1 + x2 + x3 + x4) / (alpha1 + alpha2 + alpha3 + alpha4)

        cat1 = torch.cat((x1, xs), 1)
        cat2 = torch.cat((x2, xs), 1)
        cat3 = torch.cat((x3, xs), 1)
        cat4 = torch.cat((x4, xs), 1)

        beta1 = self.sigmoid(self.beta(cat1))
        beta2 = self.sigmoid(self.beta(cat2))
        beta3 = self.sigmoid(self.beta(cat3))
        beta4 = self.sigmoid(self.beta(cat4))

        print(float(alpha1), float(alpha2), float(alpha3), float(alpha4))
        print(float(beta1), float(beta2), float(beta3), float(beta4))

        x1 = x1 * beta1
        x2 = x2 * beta2
        x3 = x3 * beta3
        x4 = x4 * beta4

        xs = (x1 + x2 + x3 + x4) / (beta1 + beta2 + beta3 + beta4)
        
        x_va = self.fc7_va(xs)
        x_expr = self.fc7_expr(xs)

        return x_va, x_expr

    def __load_pretrained_weights(self):
        """Initialiaze network."""
        corresp_name = {
                        # Conv1
                        "features.0.weight": "conv1.weight",
                        "features.0.bias": "conv1.bias",
                        # Conv2
                        "features.3.weight": "conv2.weight",
                        "features.3.bias": "conv2.bias",
                        # # Conv3a
                        # "features.6.weight": "conv3a.weight",
                        # "features.6.bias": "conv3a.bias",
                        # # Conv3b
                        # "features.8.weight": "conv3b.weight",
                        # "features.8.bias": "conv3b.bias",
                        # Conv4a
                        # "features.11.weight": "conv4a.weight",
                        # "features.11.bias": "conv4a.bias",
                        # # Conv4b
                        # "features.13.weight": "conv4b.weight",
                        # "features.13.bias": "conv4b.bias",
                        # # Conv5a
                        # "features.16.weight": "conv5a.weight",
                        # "features.16.bias": "conv5a.bias",
                        #  # Conv5b
                        # "features.18.weight": "conv5b.weight",
                        # "features.18.bias": "conv5b.bias",
                        # fc6
                        # "classifier.0.weight": "fc6.weight",
                        # "classifier.0.bias": "fc6.bias",
                        # # fc7
                        # "classifier.3.weight": "fc7.weight",
                        # "classifier.3.bias": "fc7.bias",
                        }

        p_dict = torch.load(Path.model_dir())
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict, strict=False)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def get_1x_lr_params(model):
    """
    This generator returns all the parameters for conv and two fc layers of the net.
    """
    #b = [model.conv1, model.conv2, model.conv3a, model.conv3b, model.conv4a, model.conv4b,
         #model.conv5a, model.conv5b]
    b = [model.conv1, model.conv2, model.conv3a, model.conv3b, model.conv3c, model.conv3d, 
                    model.conv4a, model.conv4b, model.conv4c, model.conv4d, 
                    model.conv5a, model.conv5b, model.conv5c, model.conv5d]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    b = [model.fc6, model.fc7, model.fc8]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k

if __name__ == "__main__":
    inputs = torch.rand(1, 3, 16, 112, 112)
    net = C3DVA(num_classes=2, pretrained=True)

    outputs = net.forward(inputs)
    print(outputs.size())