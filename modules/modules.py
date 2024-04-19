import torch
import torch.nn as nn




class Token_Embed(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.first_conv = nn.Sequential(
            nn.Conv1d(in_c, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, out_c, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G M 6
            -----------------
            feature_global : B G C
        '''
        bs, g, n, c = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, c)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 768 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 768
        return feature_global.reshape(bs, g, self.out_c)