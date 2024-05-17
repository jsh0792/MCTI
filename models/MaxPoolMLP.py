import torch
import torch.nn as nn
from Model.func import MLP_Block
from einops import rearrange
class MaxPoolMLP(nn.Module):
    def __init__(self, input_dim=1024, num_class=2, surv_num_class=4, dropout=0.25):
        super(MaxPoolMLP, self).__init__()
        hidden = [256, 256, 256]
        fc = [MLP_Block(dim1=input_dim, dim2=hidden[0])]
        for i, _ in enumerate(hidden[1:]):
            fc.append(MLP_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=dropout))
        self.emb = nn.Sequential(*fc)
        self.classifier = nn.Linear(256, num_class)

    def forward(self, **kwargs):
        x = kwargs['wsi']
        x = rearrange(x, 'n d -> d n')
        pooling = nn.MaxPool1d(x.size(1))
        x = pooling(x)
        x = rearrange(x, 'd n -> n d')
        x = self.emb(x)
        logits = self.classifier(x)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return logits, hazards, S


if __name__ == "__main__":
    data = torch.randn((6000, 1024)).cuda()
    model = MaxPoolMLP().cuda()
    print(model.eval())
    _, _ = model(data)
    print('11')