import os 
import torch
cpu_num = 3
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)

import torch.nn as nn
import torch.nn.functional as F
import sys
from Model.func import Attn_Net_Gated, SNN_Block
from einops import rearrange
import numpy as np
import ot
torch.set_printoptions(precision=8)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):

        Q = self.W_q(Q)# ([10, 20, 256])
        K = self.W_k(K)
        V = self.W_v(V)

        Q = self.split_heads(Q) # ([80, 20, 32])
        K = self.split_heads(K)
        V = self.split_heads(V)

        attention_scores = self.scale_dot_product_attention(Q, K, V, mask)

        return attention_scores

    def split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        head_dim = self.d_model // self.num_heads

        x = x.view(batch_size, seq_len, self.num_heads, head_dim)
        x = x.permute(0, 2, 1, 3).contiguous()

        return x.view(batch_size * self.num_heads, seq_len, head_dim)

    def concat_heads(self, x):
        batch_size, seq_len, _ = x.size()
        head_dim = self.d_model // self.num_heads

        x = x.view(batch_size // self.num_heads, self.num_heads, seq_len, head_dim)
        x = x.permute(0, 2, 1, 3).contiguous()

        return x.view(batch_size // self.num_heads, seq_len, self.d_model)

    def scale_dot_product_attention(self, Q, K, V, mask=None):
        head_dim = self.d_model // self.num_heads

        scores = torch.matmul(Q, K.permute(0, 2, 1)) / torch.sqrt(torch.tensor(head_dim, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)

        output = torch.matmul(attention_weights, V)

        return output

class OT_Attn_assem(nn.Module):
    def __init__(self,impl='pot-uot-l2',ot_reg=0.1, ot_tau=0.5) -> None:
        super().__init__()
        self.impl = impl
        self.ot_reg = ot_reg
        self.ot_tau = ot_tau
        print("ot impl: ", impl)
    
    def normalize_feature(self,x):
        x = x - x.min(-1)[0].unsqueeze(-1)
        return x

    def OT(self, weight1, weight2):
        """
        Parmas:
            weight1 : (N, D)
            weight2 : (M, D)
        
        Return:
            flow : (N, M)
            dist : (1, )
        """

        if self.impl == "pot-sinkhorn-l2":
            self.cost_map = torch.cdist(weight1, weight2)**2 # (N, M)
            
            src_weight = weight1.sum(dim=1) / weight1.sum()
            dst_weight = weight2.sum(dim=1) / weight2.sum()
            
            cost_map_detach = self.cost_map.detach()
            flow = ot.sinkhorn(a=src_weight.detach(), b=dst_weight.detach(), 
                                M=cost_map_detach/cost_map_detach.max(), reg=self.ot_reg)
            dist = self.cost_map * flow 
            dist = torch.sum(dist)
            return flow, dist
        
        elif self.impl == "pot-uot-l2":
            a, b = ot.unif(weight1.size()[0]).astype('float64'), ot.unif(weight2.size()[0]).astype('float64')
            self.cost_map = torch.cdist(weight1, weight2)**2 # (N, M)
            
            cost_map_detach = self.cost_map.detach()
            M_cost = cost_map_detach/cost_map_detach.max()
            
            flow = ot.unbalanced.sinkhorn_knopp_unbalanced(a=a, b=b, 
                                M=M_cost.double().cpu().numpy(), reg=self.ot_reg,reg_m=self.ot_tau)
            flow = torch.from_numpy(flow).type(torch.FloatTensor).cuda()
            
            dist = self.cost_map * flow # (N, M)
            dist = torch.sum(dist) # (1,) float
            return flow, dist
        
        else:
            raise NotImplementedError

    def forward(self,x,y):
        '''
        x: (N, 1, D)
        y: (M, 1, D)
        '''
        x = x.squeeze()
        y = y.squeeze()
        
        x = self.normalize_feature(x)
        y = self.normalize_feature(y)
        
        pi, dist = self.OT(x, y)
        return pi.T.unsqueeze(0).unsqueeze(0), dist

class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))
    def forward(self, feats):
        x = self.fc(feats)
        return feats, x

class BClassifier(nn.Module):
    # nonlinear=False
    def __init__(self, input_size, output_class, dropout_v=0.0, dim = 128, nonlinear=True, passing_v=False, max_k = 1, matrixversion=False): # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, dim), nn.ReLU(), nn.Linear(dim, dim), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, dim)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()
        self.matrixversion = matrixversion
        
        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)  
        self.max_k = max_k
        self.output_class = output_class
        self.proj = nn.Linear(1024, 1024, bias=True)
        
    def forward(self, feats, c): # N x K, N x Cls
        device = feats.device
        V = self.v(feats) # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted

        Q = F.normalize(Q, dim=-1, p=2)
        V = F.normalize(V, dim=-1, p=2)

        # handle multiple classes without for loop
        m_values, _ = torch.max(c, dim=1)
        _, m_indices = torch.sort(m_values, dim=0, descending=True)
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0:self.max_k])

        q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        # A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
        A = F.softmax( F.normalize(A, dim=-1, p=2), 1)
        B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V
                
        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        B = self.proj(B)
        if self.matrixversion:
            B = F.gelu(B)
            return B
        C = self.fcc(B) # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B 
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B x num_heads x N x N

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x

class AttentionBlock(nn.Module):
    def __init__(self, dim = 1024, depth = 2, num_heads = 1, dim_head = 1024, mlp_dim = 1024, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, num_heads = num_heads)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class DSMIL_enc_matrixversion(nn.Module):
    def __init__(self, i_classifier, b_classifier, dropout = 0, output_dim = 1024):
        super(DSMIL_enc_matrixversion, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier

        cls_fc = [  nn.Linear(1024, output_dim), nn.ReLU(),
                nn.Linear(output_dim, output_dim), nn.ReLU(),
                nn.Linear(output_dim, output_dim)]
        self.cls_proj = nn.Sequential(*cls_fc)
        
    def forward(self, **kwargs):
        x = kwargs['data'] # [13665, 1024]
        feats, classes = self.i_classifier(x)
        emb = self.b_classifier(feats, classes) # [1, 256, 1024]
        cls_emb = self.cls_proj(emb).squeeze() # [256, 1024]
        return cls_emb

class CustomAttn(nn.Module):
    def __init__(
            self,
            dim = 1024,
            num_class = 256
            ):
        super().__init__()
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj_v = nn.Linear(256, num_class, bias=False)
        self.rescale = nn.Parameter(torch.ones(1, 1))
        self.proj = nn.Linear(dim, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.GELU(),
            nn.Linear(dim, dim, bias=False),
        )

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
       
        n, c = x_in.shape
        x = x_in.reshape(n, c)
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        v = self.proj_v(v).transpose(0, 1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        v = F.normalize(v, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        # x = attn @ v   # b,heads,d,hw
        x = v @ attn # b, heads, dim, n
        out_c = self.proj(x)
        out_p = self.pos_emb(v)
        out = out_c + out_p

        return out

class TG_MSA(nn.Module):    # task guided multi-head self-attention: TG-MSA
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            device
    ):
        super().__init__()
        self.device = device
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Linear(dim, dim),
            GELU(),
            nn.Linear(dim, dim),
        )
        self.dim = dim

    def forward(self, x_in, task_x):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, n, c = x_in.shape
        x = x_in.reshape(b, n, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)

        q, k, v, task_x = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp, task_x))
        
        tmp_task_x = task_x.clone().detach()
        tmp_v_inp = v_inp.clone().detach()
        M = ot.dist(tmp_task_x.squeeze().cpu().numpy(), tmp_v_inp.squeeze().cpu().numpy()) 

        mean_value = np.mean(M)
        std_deviation = np.std(M)
        M = (M - mean_value) / std_deviation

        uni_a, uni_b = np.ones((n,)) / n, np.ones((n,)) / n
        # Ges = torch.tensor(ot.sinkhorn(a, b, M, lambd)).to(self.device).unsqueeze(0).unsqueeze(0)
        Ges = torch.tensor(ot.emd(uni_a, uni_b, M), dtype=torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)  

        guided_emb = Ges @ task_x * v
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        guided_emb = guided_emb.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        # x = attn @ v   # b,heads,d,hw
        x = attn @ guided_emb # b, heads, dim, n
        x = x.permute(0, 3, 1, 2)    # Transpose  b, n, heads, dim
        # x = x.reshape(b, n, self.num_heads * self.dim_head)
        x = rearrange(x, 'b n h d -> b n (h d)')
        out_c = self.proj(x).view(b, n, c)
        out_p = self.pos_emb(v_inp)
        out = out_c + out_p

        return out

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class TGAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            num_blocks,
            device,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PreNorm(dim, TG_MSA(dim=dim, dim_head=dim_head, heads=heads, device = device)),
                PreNorm(dim, FeedForward(dim=dim, hidden_dim=dim))
            ]))

    def forward(self, x, guided_emb):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        # x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x = x, task_x = guided_emb) + x
            x = ff(x) + x
        
        return x
    
class EncBlock(nn.Module):

    def __init__(self, dim = 1024, dim_head = 1024, heads = 1, num_blocks = 2, device = None):
        super().__init__()
        self.device = device
        self.cls_AB = AttentionBlock(dim = dim, num_heads = heads)
        self.surv_AB = AttentionBlock(dim = dim, num_heads = heads)
        self.cls_TGAB = TGAB(dim = dim, dim_head = dim_head, heads = heads, num_blocks = num_blocks, device = device)
        self.surv_TGAB = TGAB(dim = dim, dim_head = dim_head, heads = heads, num_blocks = num_blocks, device = device)
        self.fusion = nn.Conv2d(2, 1, 1)

    def forward(self, x, cls_emb, surv_emb):
        cls_emb = self.cls_AB(cls_emb)
        surv_emb = self.surv_AB(surv_emb)
        x_cls = self.cls_TGAB(x, cls_emb)
        x_surv = self.surv_TGAB(x, surv_emb)
        x = self.fusion(torch.cat([x_cls.unsqueeze(0), x_surv.unsqueeze(0)], dim = 1)).squeeze(0)
        return x, x_cls, x_surv

class BottleNeckLayer(nn.Module):

    def __init__(self, dim = 1024, dim_head = 1024, heads = 1, num_blocks = 2, device = None):
        super().__init__()
        self.device = device
        self.cls_TGAB = TGAB(dim = dim, dim_head = dim_head, heads = heads, num_blocks = num_blocks, device = device)
        self.surv_TGAB = TGAB(dim = dim, dim_head = dim_head, heads = heads, num_blocks = num_blocks, device = device)
        self.fusion = nn.Conv2d(2, 1, 1)

    def forward(self, x, cls_emb, surv_emb):
        x_cls = self.cls_TGAB(x, cls_emb)
        x_surv = self.surv_TGAB(x, surv_emb)
        x = self.fusion(torch.cat([x_cls, x_surv]))
        return x

class DecBlock(nn.Module):

    def __init__(self, dim = 1024, dim_head = 1024, heads = 1, num_blocks = 2, device = None):
        super().__init__()
        self.device = device
        self.x_AB = AttentionBlock(dim = dim, num_heads = heads)
        self.cls_TGAB = TGAB(dim = dim, dim_head = dim_head, heads = heads, num_blocks = num_blocks, device = device)
        self.surv_TGAB = TGAB(dim = dim, dim_head = dim_head, heads = heads, num_blocks = num_blocks, device = device)
        self.fusion_cls = nn.Conv2d(2, 1, 1)
        self.fusion_surv = nn.Conv2d(2, 1, 1)
        self.fusion_net = nn.Conv2d(2, 1, 1)
        self.fusion_x_cls = nn.Conv2d(2, 1, 1)
        self.fusion_x_surv = nn.Conv2d(2, 1, 1)

    def forward(self, x, cls_emb, surv_emb):
        x = self.x_AB(x)
        x_cls = self.cls_TGAB(cls_emb, self.fusion_x_cls(x))
        x_surv = self.surv_TGAB(surv_emb, self.fusion_x_surv(x))
        x = self.fusion_net(x)
        return x, x_cls, x_surv

class MCTI(nn.Module):
    def __init__(self, gene_dim = 20000, surv_num_class = 4, cls_num_class = 2, dropout = 0, 
                 stage = 2, gene_init = 'zero', token_num = 8, embed_num = 256, embed_dim = 1024, num_heads = 1,
                 num_blocks=[2,4,4], device = None):
        super(MCTI, self).__init__()
        self.device = device

        self.surv_num_class = surv_num_class
        self.cls_num_class = cls_num_class
        self.token_num = token_num
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.stage = stage

        self.task_token_cls = nn.Parameter(torch.randn(1, token_num, self.embed_dim))  
        self.task_token_surv = nn.Parameter(torch.randn(1, token_num, self.embed_dim))
        self.task_token_mix = nn.Parameter(torch.randn(1, token_num, self.embed_dim))

        if gene_init == 'zero':
            self.gene_bias = nn.Parameter(torch.zeros(embed_num, self.embed_dim))
        elif gene_init == 'rand':
            self.gene_bias = nn.Parameter(torch.randn(embed_num, self.embed_dim))

        self.size_dict_omic = {'small': [self.embed_dim, self.embed_dim], 'big': [256, 256, 256, 256]}
        self.gene_size_dict_omic = {'small': [8192, 8192], 'big': [256, 256, 256, 256]} 
        ### Constructing Genomic SNN
        hidden = self.gene_size_dict_omic['small']
        fc_omic = [SNN_Block(dim1=gene_dim, dim2=hidden[0])]
        for i, _ in enumerate(hidden[1:]):
            fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=dropout))
        self.Gene_encode = nn.Sequential(*fc_omic)

        self.i_classifier = FCLayer(in_size=1024, out_size=self.cls_num_class)  # DSMIL 
        self.b_classifier = BClassifier(input_size=1024, output_class=self.embed_dim, dropout_v=0, nonlinear=1, max_k=self.embed_num, matrixversion=True)
        self.WSI_encode = DSMIL_enc_matrixversion(self.i_classifier, self.b_classifier, output_dim=self.embed_dim)

        fc = [nn.Linear(self.embed_dim, self.embed_dim), nn.GELU()]
        # attention_net = Attn_Net_Gated(L=self.embed_dim, D=self.embed_dim, dropout=dropout, n_classes=embed_num)
        attention_net = CustomAttn() 
        fc.append(attention_net)
        self.attention_net_wsi = nn.Sequential(*fc)
    
        self.multi_head_attention = nn.Identity()   # in gene, multi head attention

        self.encoder_layers = nn.ModuleList([])

        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                EncBlock(
                    dim=embed_dim, num_blocks=num_blocks[i], dim_head=embed_dim, heads=1, device = device)
            ]))
        
        self.bottleneck = BottleNeckLayer(device = device)
        
        self.decoder_leyers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_leyers.append(nn.ModuleList([
                DecBlock(
                    dim=embed_dim, num_blocks=num_blocks[i], dim_head=embed_dim, heads=1, device = device)
            ]))

        self.cls_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, self.cls_num_class)
        )

        self.surv_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, self.surv_num_class)
        )

        self.multi_head = MultiHeadAttention(8192, 8)
        self.coattn = OT_Attn_assem()

    def forward(self, **kwargs):
        WSI_data = kwargs['wsi_data']   # [patch_num, 1024] (torch.Size([18426, 1024]))
        Gene_data = kwargs['gene_data'] #  [gene_dim] (torch.Size([19437])

        WSI_embed = self.WSI_encode(data=WSI_data)  # torch.Size([256, 1024])
        WSI_embed = self.attention_net_wsi(WSI_embed)   
        
        Gene_embed = self.Gene_encode(Gene_data).unsqueeze(0)   # torch.Size([1, 8192])
        Gene_embed = Gene_embed.unsqueeze(0) 
        Gene_embed = self.multi_head(Gene_embed, Gene_embed, Gene_embed)  # [8, 1024]

        A_coattn, _ = self.coattn(WSI_embed, Gene_embed)
        WSI_x = torch.mm(A_coattn.squeeze(), WSI_embed.squeeze())   

        x = torch.cat((WSI_embed.squeeze().unsqueeze(dim=0), WSI_x.squeeze().unsqueeze(dim=0), Gene_embed.squeeze().unsqueeze(dim=0)), dim = 1) # torch.Size([1, 512, 1024])

        cls_emb = torch.cat((x, self.task_token_cls), dim = 1)  # torch.Size([1, 512, 1024]) 
        surv_emb = torch.cat((x, self.task_token_surv), dim = 1)# torch.Size([1, 512, 1024])
        x = torch.cat((x, self.task_token_mix), dim = 1)

        fea_encoder = []
        for (blk) in self.encoder_layers:
            x, cls_emb, surv_emb = blk[0](x, cls_emb, surv_emb)
            fea_encoder.append(x)

        x = self.bottleneck(x, cls_emb, surv_emb)

        for i, (blk) in enumerate(self.decoder_leyers):
            x, cls_emb, surv_emb = blk[0](torch.cat([x, fea_encoder[self.stage-1-i]], dim=0), cls_emb, surv_emb)
            # x, cls_emb, surv_emb = blk[0](x, cls_emb, surv_emb)

        cls_emb = torch.mean(cls_emb, dim=1).squeeze()
        surv_emb = torch.mean(surv_emb, dim=1).squeeze()

        cls_logits = self.cls_head(cls_emb).unsqueeze(0)
        surv_logits = self.surv_head(surv_emb).unsqueeze(0)

        hazards = torch.sigmoid(surv_logits)
        S = torch.cumprod(1 - hazards, dim=1)

        return cls_logits, surv_logits, S, hazards

if __name__ == "__main__":
    wsi_data = torch.randn((20000, 1024)).cuda()
    gene_data = torch.randn(17500).cuda()
    model = MCTI(gene_dim=17500, token_num=240, embed_dim=1024, device='cuda').cuda()
    with torch.no_grad():
        cls_logits, surv_logits, S, hazards = model(wsi_data=wsi_data, gene_data=gene_data, mode=4)
    risk = -torch.sum(S, dim=1).cpu().numpy()
