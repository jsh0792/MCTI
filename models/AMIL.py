import sys
sys.path.append('/data1/jsh/multitask')
from collections import OrderedDict
from os.path import join
import pdb

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.func import *

################################
# Attention MIL Implementation #
################################
class AMIL(nn.Module):
    def __init__(self, omic_input_dim=None, fusion=None, size_arg = "small", dropout=0.25, num_class=2):
        r"""
        Attention MIL Implementation

        Args:
            omic_input_dim (int): Dimension size of genomic features.
            fusion (str): Fusion method (Choices: concat, bilinear, or None)
            size_arg (str): Size of NN architecture (Choices: small or large)
            dropout (float): Dropout rate
            n_classes (int): Output shape of NN
        """
        super(AMIL, self).__init__()
        self.fusion = fusion
        self.size_dict_path = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [256, 256]}

        ### Deep Sets Architecture Construction
        size = self.size_dict_path[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.rho = nn.Sequential(*[nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)])

        ### Constructing Genomic SNN
        if self.fusion is not None:
            hidden = [256, 256]
            fc_omic = [SNN_Block(dim1=omic_input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            self.fc_omic = nn.Sequential(*fc_omic)
        
            if self.fusion == 'concat':
                self.mm = nn.Sequential(*[nn.Linear(256*2, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()])
            else:
                self.mm = None

        self.cls_head = nn.Linear(256, num_class)

    def forward(self, **kwargs):
        x_path = kwargs['wsi']

        A, h_path = self.attention_net(x_path)  # A [6000,1]   h_path [6000,512]
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1) 
        h_path = torch.mm(A, h_path)
        h_path = self.rho(h_path).squeeze()

        if self.fusion is not None:
            x_omic = kwargs['gene']
            h_omic = self.fc_omic(x_omic)
            if self.fusion == 'bilinear':
                h = self.mm(h_path.unsqueeze(dim=0), h_omic.unsqueeze(dim=0)).squeeze()
            elif self.fusion == 'concat':
                h = self.mm(torch.cat([h_path, h_omic], axis=0))
        else:
            h = h_path # [256] vector

        logits = self.cls_head(h).unsqueeze(0)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return logits, hazards, S
    
if __name__ == "__main__":
    data = torch.randn((6000, 1024)).cuda()
    model = AMIL(num_class=2).cuda()
    print(model.eval())
    logits, hazards, S = model(wsi = data)
    print(logits, hazards, S)