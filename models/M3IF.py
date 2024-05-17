import torch
from torch.nn import Linear, BatchNorm1d, ReLU
import numpy as np
import torch.nn as nn

from torch.autograd import Function
import torch.nn.functional as F
from Model.func import *
import torch
class MMTMBi(nn.Module):
    """
    bi moludal fusion
    """

    def __init__(self, dim_tab, dim_img, ratio=4):
        """
        Parameters
        ----------
        dim_tab: feature dimension of tabular data
        dim_img: feature dimension of MIL image modal
        ratio
        """
        super(MMTMBi, self).__init__()
        dim = dim_tab + dim_img
        dim_out = int(2 * dim / ratio)
        self.fc_squeeze = nn.Linear(dim, dim_out)

        self.fc_tab = nn.Linear(dim_out, dim_tab)
        self.fc_img = nn.Linear(dim_out, dim_img)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, tab_feat, img_feat):
        """
        Parameters
        ----------
        tab_feat: b * c
        skeleton: b * c
        Returns
            表格数据加权结果
            WSI 全局特征加权结果
            WSI 全局特征加权权重
        -------
        """

        squeeze = torch.cat([tab_feat, img_feat], dim=1)

        excitation = self.fc_squeeze(squeeze)
        excitation = self.relu(excitation)

        tab_out = self.fc_tab(excitation)
        img_out = self.fc_img(excitation)

        tab_out = self.sigmoid(tab_out)
        img_out = self.sigmoid(img_out)

        return tab_feat * tab_out, img_feat * img_out, img_out

class InstanceAttentionGate(nn.Module):
    def __init__(self, feat_dim):
        super(InstanceAttentionGate, self).__init__()
        self.trans = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.LeakyReLU(),
            nn.Linear(feat_dim, 1),
        )

    def forward(self, instance_feature, global_feature):
        feat = torch.cat([instance_feature, global_feature], dim=1)
        attention = self.trans(feat)
        return attention
    
class M3IF(nn.Module):
    def __init__(self, img_feat_input_dim=1024, tab_feat_input_dim=32,
                 img_feat_rep_layers=4,
                 num_modal=2,
                 tab_indim=0,
                 local_rank=0,
                 lambda_sparse=1e-3,
                 fusion='mmtm',
                 num_class = 4
                 ):
        super(M3IF, self).__init__()
        self.num_modal = num_modal
        self.local_rank = local_rank
        self.tab_indim = tab_indim
        self.lambda_sparse = lambda_sparse

        self.fusion_method = fusion
        hidden = [256, 256]
        fc_omic = [SNN_Block(dim1=tab_indim, dim2=hidden[0])]
        for i, _ in enumerate(hidden[1:]):
            fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
        self.fc_omic = nn.Sequential(*fc_omic)

        """
        Control tabnet
        """
        feature_fine_tuning_layers = []
        for _ in range(img_feat_rep_layers):
            feature_fine_tuning_layers.extend([
                nn.Linear(img_feat_input_dim, img_feat_input_dim),
                nn.LeakyReLU(),
            ])
        self.feature_fine_tuning = nn.Sequential(*feature_fine_tuning_layers)

        # k agg score
        self.score_fc = nn.ModuleList()

        """modal fusion"""
        # define different fusion methods and related output feature dimension and fusion module

        self.fusion_out_dim = tab_feat_input_dim + img_feat_input_dim
        self.wsi_select_gate = nn.Sequential(
            nn.Linear(img_feat_input_dim, 1),
            nn.Sigmoid()
        )
        self.fusion_out_dim = (img_feat_input_dim * 2) * (num_modal - 1) + tab_feat_input_dim
        self.mmtm = MMTMBi(dim_tab=tab_feat_input_dim, dim_img=img_feat_input_dim)

        self.instance_gate1 = InstanceAttentionGate(img_feat_input_dim)
        self.table_feature_ft = nn.Sequential(
                nn.Linear(tab_feat_input_dim, tab_feat_input_dim)
            )

        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_out_dim, self.fusion_out_dim),
            nn.Dropout(0.5),
            nn.Linear(self.fusion_out_dim, num_class)
        )

    def forward(self, **kwargs):
        gene = kwargs['gene']
        wsi = kwargs['wsi']
        attention_weight_out_list = []

        tab_data = gene

        tab_feat = self.fc_omic(tab_data).unsqueeze(0)

        wsi_feat_scale1 = wsi

        if len(wsi_feat_scale1.size()) == 3:
            # 1 #instance #feat
            wsi_feat_scale1 = wsi_feat_scale1.squeeze(0)
        scale1_bs = wsi_feat_scale1.shape[0]

        wsi_feat_scale1 = self.feature_fine_tuning(wsi_feat_scale1)
        wsi_feat_scale1_gloabl = torch.mean(wsi_feat_scale1, dim=0, keepdim=True)  # instance level mean

        tab_feat_mmtm, wsi_feat1_gloabl, wsi_feat_scale1_gate = self.mmtm(tab_feat, wsi_feat_scale1_gloabl)

        # table feature calculate once more
        tab_feat_ft = self.table_feature_ft(tab_feat_mmtm)

        # weight on feature level
        wsi_feat_scale1 = wsi_feat_scale1 * wsi_feat_scale1_gate

        wsi_feat1_gloabl_repeat = wsi_feat1_gloabl.detach().repeat(scale1_bs, 1)

        # N * 1
        instance_attention_weight = self.instance_gate1(wsi_feat_scale1, wsi_feat1_gloabl_repeat)
        # 1 * N
        instance_attention_weight = torch.transpose(instance_attention_weight, 1, 0)

        instance_attention_weight = torch.softmax(instance_attention_weight, dim=1)

        attention_weight_out_list.append(instance_attention_weight.detach().clone())

        # 1 * N
        wsi_feat_agg_scale1 = torch.mm(instance_attention_weight, wsi_feat_scale1)

        final_feat = torch.cat([tab_feat_ft, wsi_feat_agg_scale1, wsi_feat1_gloabl], dim=1)

        logits = self.classifier(final_feat)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return logits, hazards, S

    def get_params(self, base_lr):
        ret = []

        if self.tabnet is not None:
            tabnet_params = []
            for param in self.tabnet.parameters():
                tabnet_params.append(param)
            ret.append({
                'params': tabnet_params,
                'lr': base_lr
            })

        cls_learning_rate_rate=100
        if self.classifier is not None:
            classifier_params = []
            for param in self.classifier.parameters():
                classifier_params.append(param)
            ret.append({
                'params': classifier_params,
                'lr': base_lr / cls_learning_rate_rate,
            })


        tab_learning_rate_rate = 100
        if self.table_feature_ft is not None:
            misc_params = []
            for param in self.table_feature_ft.parameters():
                misc_params.append(param)
            ret.append({
                'params': misc_params,
                'lr': base_lr / tab_learning_rate_rate,
            })

        mil_learning_rate_rate = 1000
        misc_params = []
        for part in [self.feature_fine_tuning, self.feature_fine_tuning2, self.feature_fine_tuning3,
                     self.instance_gate1, self.instance_gate2, self.instance_gate3,
                     self.wsi_select_gate,
                     self.score_fc]:
            if part is not None:
                for param in part.parameters():
                    misc_params.append(param)
        ret.append({
            'params': misc_params,
            'lr': base_lr / mil_learning_rate_rate,
        })

        misc_learning_rate_rate = 100
        misc_params = []
        for part in [self.mmtm, ]:
            if part is not None:
                for param in part.parameters():
                    misc_params.append(param)
        ret.append({
            'params': misc_params,
            'lr': base_lr / misc_learning_rate_rate,
        })

        return ret
    
if __name__ == "__main__":
    data = torch.randn((6000, 1024)).cuda()
    gene = torch.randn((13333)).cuda()
    model = M3IF(img_feat_input_dim=1024,
                          tab_feat_input_dim=256,
                          img_feat_rep_layers=4,
                          num_modal=2,
                          fusion='mmtm',
                          use_tabnet=True,
                          use_k_agg=False,
                          k_agg=10,
                          tab_indim=13333).cuda()
    print(model.eval())
    results_dict = model(wsi_feat_scale1 = data, gene = gene)
    print(results_dict)