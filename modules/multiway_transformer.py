from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
from timm.models.layers import DropPath, trunc_normal_
import torch
import torch.nn as nn
from pytorch3d.ops import sample_farthest_points, knn_points

from .modules import Token_Embed #, Group


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    

class Adapter(nn.Module):
    def __init__(self, d_model, bottleneck,):
        super().__init__()

        self.mlp = nn.Sequential(OrderedDict([
            ("down_proj", nn.Linear(d_model, bottleneck)),
            ("relu", nn.ReLU()),
            ("up_proj", nn.Linear(bottleneck, d_model))
        ]))

    def forward(self, x):
        return self.mlp(x)

class AdapterZero(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.zeros_like(x)

class ResidualAttentionBlock(nn.Module):
    def __init__(
            self, 
            d_model: int, 
            expert_dim: int,
            img_adapter_att_dim: int,
            img_adapter_mlp_dim: int,
            pcd_adapter_att_dim: int,
            pcd_adapter_mlp_dim: int,
            n_head: int, 
            attn_mask: torch.Tensor = None, 
            drop_path = 0.0,):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)

        self.img_mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.pcd_mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, expert_dim)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(expert_dim, d_model))
        ])) if expert_dim > 0 else None

        self.img_adapter_att = Adapter(d_model, img_adapter_att_dim) if img_adapter_att_dim > 0 else AdapterZero()
        #self.img_adapter_mlp = Adapter(d_model, img_adapter_mlp_dim) if img_adapter_mlp_dim > 0 else AdapterZero()
        self.pcd_adapter_att = Adapter(d_model, pcd_adapter_att_dim) if pcd_adapter_att_dim > 0 else AdapterZero()
        self.pcd_adapter_mlp = Adapter(d_model, pcd_adapter_mlp_dim) if pcd_adapter_mlp_dim > 0 else AdapterZero()

        self.img_ln_2 = LayerNorm(d_model)
        self.pcd_ln_2 = LayerNorm(d_model)# if expert_dim > 0 else None

        self.attn_mask = attn_mask
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, modality_type):
        if modality_type == "img":
            x = x + self.attention(self.ln_1(x)) + self.drop_path(self.img_adapter_att(self.ln_1(x)))
            x = x + self.img_mlp(self.img_ln_2(x))
        elif modality_type == "pcd":
            x = x + self.attention(self.ln_1(x)) + self.drop_path(self.pcd_adapter_att(self.ln_1(x)))
            if self.pcd_mlp is not None:
                x = x + self.drop_path(self.pcd_mlp(self.pcd_ln_2(x)))
            else:
                x = x + self.img_mlp(self.pcd_ln_2(x)) + self.drop_path(self.pcd_adapter_mlp(self.pcd_ln_2(x)))
        else:
            raise Exception("Wrong modality type")
        return x
    


class IMGPatchEmbed(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.type_embedding = nn.Parameter(scale * torch.randn(width))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = x + self.type_embedding
        return x


class PCDPatchEmbed(nn.Module):
    def __init__(self, num_group, group_size, width):
        super().__init__()
        self.token_embed = Token_Embed(in_c=6, out_c=width)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        #self.group = Group(num_group, group_size)
        self.num_group = num_group
        self.group_size = group_size
        self.encoder_pos_embed = nn.Sequential(
                            nn.Linear(3, width),
                            nn.GELU(),
                            nn.Linear(width, width),
                        )
        self.type_embedding = nn.Parameter(scale * torch.randn(width))
        
    def forward(self, pc):
        #neighbors, centers = self.group(pc) #(B,N,3) --> (B,G,M,3)(B,G,3)
        centers, _ = sample_farthest_points(pc, K=self.num_group, ) #(B,G,3)
        _, _, neighbors = knn_points(p1=centers, p2=pc, K=self.group_size, return_nn=True) #(B,G,M,3)
        x = torch.cat([neighbors, neighbors - centers.unsqueeze(2)], dim=3) #(B,G,M,6)
        x = self.token_embed(x)  #(B,G,C)
        x = torch.cat([self.class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1], device=x.device), x], dim=1) #B,G+1,C
        centers = torch.cat([torch.zeros(centers.shape[0], 1, 3).to(x.device), centers], dim=1) #B,G+1,3
        pos = self.encoder_pos_embed(centers) #B,G+1,C
        x = x + pos #B,G+1,C
        x = x + self.type_embedding
        return x



class MultiWayTransformer(nn.Module):
    def __init__(self, 
                 input_resolution: int, 
                 patch_size: int, 
                 num_group, 
                 group_size, 
                 expert_dim,
                 img_adapter_att_dim,
                 img_adapter_mlp_dim,
                 pcd_adapter_att_dim,
                 pcd_adapter_mlp_dim,
                 width=768, 
                 layers=12, 
                 heads=12, 
                 output_dim=256, 
                 attn_mask: torch.Tensor = None,
                 drop_path_rate=0.,):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.scale = width ** -0.5
        self.layers = layers
        self.img_patch_emb = IMGPatchEmbed(input_resolution, patch_size, width)
        self.pcd_patch_emb = PCDPatchEmbed(num_group, group_size, width)
        self.img_ln_pre = LayerNorm(width)
        self.pcd_ln_pre = LayerNorm(width)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]  
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(
            d_model=width, 
            expert_dim=expert_dim, 
            img_adapter_att_dim=img_adapter_att_dim,
            img_adapter_mlp_dim=img_adapter_mlp_dim,
            pcd_adapter_att_dim=pcd_adapter_att_dim,
            pcd_adapter_mlp_dim=pcd_adapter_mlp_dim,
            n_head=heads, 
            attn_mask=attn_mask, 
            drop_path=dpr[i],
            ) 
            for i in range(layers)])
        
        self.img_ln_post = LayerNorm(width)
        self.pcd_ln_post = LayerNorm(width)
        self.img_proj = nn.Parameter(self.scale * torch.randn(width, output_dim))
        self.pcd_proj = nn.Parameter(self.scale * torch.randn(width, 1024))
        self.pcd_inv_head = nn.Sequential(
                            nn.BatchNorm1d(1024),
                            nn.ReLU(inplace=True),
                            nn.Linear(1024, output_dim)) #256
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def encode_img(self, x, modality_type):
        x = self.img_patch_emb(x)
        x = self.img_ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for blk in self.resblocks:
            x = blk(x, modality_type=modality_type)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.img_ln_post(x[:, 0])
        x = x @ self.img_proj
        return x
    
    def encode_pcd(self, x, modality_type):
        x = self.pcd_patch_emb(x)
        x = self.pcd_ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for blk in self.resblocks:
            x = blk(x, modality_type=modality_type)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.pcd_ln_post(x[:, 0])
        x = x @ self.pcd_proj #1024
        inv_feat = self.pcd_inv_head(x) #256
        return inv_feat, x


    def forward(self, x: torch.Tensor, modality_type=None):
        if modality_type == 'img':
            x = self.encode_img(x, modality_type)
        elif modality_type == 'pcd':
            x = self.encode_pcd(x, modality_type)
        else:
            raise Exception("Wrong modality type")
        return x


