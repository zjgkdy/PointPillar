"""
PointPillars fork from SECOND.
Code written by Alex Lang and Oscar Beijbom, 2018.
Licensed under MIT License [see LICENSE].
"""

import torch
from torch import nn
from torch.nn import functional as F

from second.pytorch.utils import get_paddings_indicator
from torchplus.nn import Empty
from torchplus.tools import change_default_args


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        """
        Pillar Feature Net Layer.
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer. This layer performs a similar role as second.pytorch.voxelnet.VFELayer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """

        super().__init__()
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        if use_norm:
            BatchNorm1d = change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)

        self.linear = Linear(in_channels, self.units)
        self.norm = BatchNorm1d(self.units)

    def forward(self, inputs):
        """_summary_

        Args:
            inputs (_type_): pillar点组 [P N D]

        Returns:
            _type_: _description_ 特征 [P 1 D']
        """
        x = self.linear(inputs)                                                         # MLP: [P N D']
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()    # BatchNorm
        x = F.relu(x)                                                                   # ReLU
        x_max = torch.max(x, dim=1, keepdim=True)[0]                                    # MaxPooling: [P 1 D']

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)  # [P N D']
            x_concatenated = torch.cat([x, x_repeat], dim=2)  # [P N 2D']
            return x_concatenated


class PillarFeatureNet(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=(64,),
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1)):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """

        super().__init__()
        self.name = 'PillarFeatureNet'
        assert len(num_filters) > 0

        # 输入特征数
        num_input_features += 5
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        # Pillar 网格尺寸, X/Y 偏置
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]

        # PillarFeatureNet
        num_filters = [num_input_features] + list(num_filters)  # [9， 64]
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(PFNLayer(in_filters, out_filters, use_norm, last_layer=last_layer))
        self.pfn_layers = nn.ModuleList(pfn_layers)

    def forward(self, features, num_voxels, coors):
        """_summary_

        Args:
            features (_type_): Pillar点组信息  [P N D]
            num_voxels (_type_): Pillar有效点数  [P]
            coors (_type_): Pillar点坐标  [P N 3]

        Returns:
            _type_: Pillar特征向量  [P D']
        """
        # 扩展几何特征：到 pillar 点组质心的三轴偏移
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)  # [P 1 3]
        f_cluster = features[:, :, :3] - points_mean  # [P N 3]

        # 扩展几何特征：到 pillar 中心的二轴偏移
        f_center = torch.zeros_like(features[:, :, :2])  # [P N 2]
        f_center[:, :, 0] = features[:, :, 0] - (coors[:, 3].float().unsqueeze(1) * self.vx + self.x_offset)  # [P N 1]
        f_center[:, :, 1] = features[:, :, 1] - (coors[:, 2].float().unsqueeze(1) * self.vy + self.y_offset)  # [P N 1]

        # 特征拼接
        features_ls = [features, f_cluster, f_center]
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)  # 计算L2范数：[P N 1]
            features_ls.append(points_dist)
        features = torch.cat(features_ls, dim=-1)  # [P N D+5]

        # Pillar点数不足时零填充
        voxel_count = features.shape[1]  # N
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)  # [P N]
        mask = torch.unsqueeze(mask, -1).type_as(features)  # [P N 1]
        features *= mask  # [P N 1]

        # PFNLayers：前向传播
        for pfn in self.pfn_layers:
            features = pfn(features)  # [P 1 D']

        return features.squeeze()  # 去除张量中所有大小为 1 的维度：[P D']


class PointPillarsScatter(nn.Module):
    def __init__(self,
                 output_shape,              # 输出稠密伪图像尺寸
                 num_input_features=64):    # 输入特征通道数
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """

        super().__init__()
        self.name = 'PointPillarsScatter'
        self.output_shape = output_shape
        self.ny = output_shape[2]
        self.nx = output_shape[3]
        self.nchannels = num_input_features

    def forward(self, voxel_features, coords, batch_size):

        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(self.nchannels, self.nx * self.ny, dtype=voxel_features.dtype,
                                 device=voxel_features.device)

            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_itt
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, self.nchannels, self.ny, self.nx)

        return batch_canvas
