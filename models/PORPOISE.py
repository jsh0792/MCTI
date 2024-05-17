import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from os.path import join
from collections import OrderedDict
from Model.func import *
class PorpoiseMMF(nn.Module):
    def __init__(self, 
        omic_input_dim,
        path_input_dim=1024, 
        fusion='bilinear', 
        dropout=0.25,
        n_classes=4,  
        dropinput=0.10,
        size_arg = "small",
        ):
        super(PorpoiseMMF, self).__init__()
        self.fusion = fusion
        self.size_dict_path = {"small": [path_input_dim, 512, 256], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [256, 256]}
        self.n_classes = n_classes

        ### Deep Sets Architecture Construction
        size = self.size_dict_path[size_arg]
        if dropinput:
            fc = [nn.Dropout(dropinput), nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        else:
            fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.rho = nn.Sequential(*[nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)])

        ### Constructing Genomic SNN
        if self.fusion is not None:

            Block = SNN_Block

            hidden = self.size_dict_omic['small']
            fc_omic = [Block(dim1=omic_input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            self.fc_omic = nn.Sequential(*fc_omic)
        
            self.mm = nn.Sequential(*[nn.Linear(256*2, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()])

        self.classifier_mm = nn.Linear(size[2], n_classes)

    def forward(self, **kwargs):
        x_path = kwargs['wsi']
        A, h_path = self.attention_net(x_path)  
        A = torch.transpose(A, 1, 0)
        A_raw = A 
        A = F.softmax(A, dim=1) 
        h_path = torch.mm(A, h_path)
        h_path = self.rho(h_path)

        x_omic = kwargs['gene']
        h_omic = self.fc_omic(x_omic).unsqueeze(0)

        h_mm = self.mm(torch.cat([h_path, h_omic], axis=1))

        logits = self.classifier_mm(h_mm)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return logits, hazards, S

if __name__ == "__main__":
    data = torch.randn((6000, 1024)).cuda()
    gene = torch.randn((13333)).cuda()
    model = PorpoiseMMF(omic_input_dim=13333).cuda()
    print(model.eval())
    _, _ = model(x_path=data, x_omic=gene)
    print('11')