import sys
sys.path.append('/data3/jsh/multitask')
import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.func import *
##########################
#### Genomic FC Model ####
##########################
class SNN(nn.Module):
    def __init__(self, input_dim: int, model_size_omic: str='small', n_classes: int=4, dropout=0.25):
        super(SNN, self).__init__()
        self.n_classes = n_classes
        self.size_dict_omic = {'small': [1024, 64], 'big': [256, 256, 256, 256]}
        
        ### Constructing Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
        for i, _ in enumerate(hidden[1:]):
            fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=dropout))
        self.fc_omic = nn.Sequential(*fc_omic)
        self.classifier = nn.Linear(hidden[-1], n_classes)
        # init_max_weights(self)


    def forward(self,x):
        # x = kwargs['gene']
        features = self.fc_omic(x)
        print(features.shape)
        logits = self.classifier(features).unsqueeze(0)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return logits, hazards, S
    
if __name__ == "__main__":
    data = torch.randn(13333).cuda()
    model = SNN(13333).cuda()
    dict = {'gene':data}
    _, _, _ = model(data)
    print('11')