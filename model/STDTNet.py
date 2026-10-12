
import math
from importlib import import_module

import torch
from torch import nn

from graph.tools import get_anatomical_graph


class DPAGC(nn.Module):

    def __init__(self, in_channels, out_channels, adjacency, stride=1):
        super().__init__()
        relation_channels = out_channels // 8
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, (3, 1), stride=(stride, 1),
            padding=(1, 0), groups=in_channels, bias=False,
        )
        self.point_rel = nn.Conv2d(in_channels, relation_channels, 1, bias=False)
        self.point_out = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.relation_proj = nn.Conv2d(relation_channels, out_channels, 1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("adjacency", adjacency.clone())
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        temporal = self.depthwise(x)
        relation = self.point_rel(temporal).mean(dim=2)
        features = self.point_out(temporal)
        difference = torch.tanh(relation.unsqueeze(-1) - relation.unsqueeze(-2))
        channel_graph = self.alpha * self.relation_proj(difference) + self.adjacency
        frame_mean = features.mean(dim=1)
        frame_graph = torch.tanh(frame_mean.unsqueeze(-1) - frame_mean.unsqueeze(-2))
        sequence_path = torch.einsum("bcuv,bctv->bctu", channel_graph, features)
        frame_path = torch.einsum("btuv,bctv->bctu", frame_graph, features)
        return self.bn(self.beta * sequence_path + self.gamma * frame_path)


class MSTC(nn.Module):

    def __init__(self, channels):
        super().__init__()
        width = channels // 4
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, width, 1, bias=False),
                nn.Conv2d(width, width, (3, 1), padding=(1, 0), bias=False),
            ),
            nn.Sequential(
                nn.Conv2d(channels, width, 1, bias=False),
                nn.Conv2d(width, width, (3, 1), padding=(2, 0),
                          dilation=(2, 1), bias=False),
            ),
            nn.Sequential(
                nn.Conv2d(channels, width, 1, bias=False),
                nn.MaxPool2d((3, 1), stride=1, padding=(1, 0)),
            ),
            nn.Conv2d(channels, width, 1, bias=False),
        ])
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return self.bn(torch.cat([branch(x) for branch in self.branches], dim=1))


class STBlock(nn.Module):

    def __init__(self, in_channels, out_channels, adjacency, stride=1):
        super().__init__()
        self.dpagc = nn.Sequential(
            DPAGC(in_channels, out_channels, adjacency, stride),
            DPAGC(out_channels, out_channels, adjacency),
            DPAGC(out_channels, out_channels, adjacency),
        )
        self.mstc = MSTC(out_channels)
        self.residual = (
            nn.Identity() if in_channels == out_channels and stride == 1 else
            nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1), bias=False)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.mstc(self.dpagc(x)) + self.residual(x))


class GSTT(nn.Module):

    def __init__(self, num_frames, num_joints, adjacency, dropout):
        super().__init__()
        self.num_heads = 4
        self.head_dim = 32
        self.token_proj = nn.Linear(128, 128)
        position = torch.arange(num_frames, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(torch.arange(0, 128, 2) * (-math.log(10000.0) / 128))
        temporal_pe = torch.zeros(num_frames, 128)
        temporal_pe[:, 0::2] = torch.sin(position * frequency)
        temporal_pe[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer("temporal_pe", temporal_pe[:, None, :])
        self.spatial_pe = nn.Parameter(torch.empty(num_joints, 128))
        nn.init.normal_(self.spatial_pe, mean=0.0, std=0.02)
        self.rho_t = nn.Parameter(torch.tensor(1.0))
        self.rho_s = nn.Parameter(torch.tensor(1.0))
        self.mu = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("block_bias", torch.kron(torch.eye(num_frames), adjacency))
        self.norm1 = nn.LayerNorm(128)
        self.qkv = nn.Linear(128, 3 * 128)
        self.output_proj = nn.Linear(128, 128)
        self.quality_gate = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1))
        self.norm2 = nn.LayerNorm(128)
        self.ffn = nn.Sequential(nn.Linear(128, 512), nn.ReLU(), nn.Linear(512, 128))
        self.dropout = nn.Dropout(dropout)

    def forward(self, local_features, quality):
        batch, channels, frames, joints = local_features.shape
        tokens = local_features.permute(0, 2, 3, 1)
        tokens = self.token_proj(tokens)
        tokens = tokens + self.rho_t * self.temporal_pe + self.rho_s * self.spatial_pe
        tokens = tokens.reshape(batch, frames * joints, channels)
        qkv = self.qkv(self.norm1(tokens)).reshape(
            batch, frames * joints, 3, self.num_heads, self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        logits = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        standard = torch.softmax(logits, dim=-1)
        graph = torch.softmax(logits + self.mu * self.block_bias, dim=-1)
        zeta = torch.sigmoid(self.quality_gate(quality)).reshape(batch, 1, 1, 1)
        attention = zeta * standard + (1 - zeta) * graph
        context = (attention.to(value.dtype) @ value).transpose(1, 2).reshape(
            batch, frames * joints, channels,
        )
        tokens = tokens + self.dropout(self.output_proj(context))
        return tokens + self.dropout(self.ffn(self.norm2(tokens)))


class Model(nn.Module):

    def __init__(self, num_class, num_point, num_person=1,
                 graph="graph.ntu_rgb_d.Graph", graph_args=None,
                 in_channels=3, num_frame=64, drop_out=0.1):
        super().__init__()
        if num_person != 1 or in_channels != 3:
            raise ValueError("STDT-Net expects three coordinates and one selected performer.")
        if num_frame < 4:
            raise ValueError("num_frame must be at least four.")
        module_name, class_name = graph.rsplit(".", 1)
        self.graph = getattr(import_module(module_name), class_name)(**(graph_args or {}))
        if self.graph.num_node != num_point:
            raise ValueError("num_point does not match the selected skeleton graph.")
        self.num_class = num_class
        self.num_point = num_point
        self.num_frame = num_frame
        adjacency = torch.from_numpy(get_anatomical_graph(num_point, self.graph.neighbor))
        self.register_buffer("adjacency", adjacency)
        self.register_buffer("quality_mean", torch.zeros(3))
        self.register_buffer("quality_std", torch.ones(3))
        self.blocks = nn.ModuleList()
        previous = in_channels
        for index, channels in enumerate([64] * 4 + [128] * 6):
            stride = 2 if index in (4, 7) else 1
            self.blocks.append(STBlock(previous, channels, adjacency, stride))
            previous = channels
        self.gstt = GSTT((num_frame + 3) // 4, num_point, adjacency, drop_out)
        self.global_proj = nn.Linear(128, 128)
        self.fusion_gate = nn.Sequential(nn.Linear(259, 16), nn.ReLU(), nn.Linear(16, 1))
        self.dropout = nn.Dropout(drop_out)
        self.fc = nn.Linear(128, num_class)
        self.apply(initialize_weights)
        for weight in self.gstt.qkv.weight.chunk(3, dim=0):
            nn.init.xavier_uniform_(weight)

    @torch.no_grad()
    def set_quality_stats(self, mean, std):
        mean = torch.as_tensor(mean, dtype=self.quality_mean.dtype, device=self.quality_mean.device)
        std = torch.as_tensor(std, dtype=self.quality_std.dtype, device=self.quality_std.device)
        if mean.shape != (3,) or std.shape != (3,):
            raise ValueError("Quality mean and std must each contain three values.")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Quality statistics must be finite, with strictly positive std.")
        self.quality_mean.copy_(mean)
        self.quality_std.copy_(std.clamp_min(1e-6))

    def forward(self, x, q_raw):
        if x.ndim != 5 or tuple(x.shape[1:]) != (3, self.num_frame, self.num_point, 1):
            raise ValueError("Expected X[B,3,num_frame,num_point,1].")
        if q_raw.ndim != 2 or q_raw.shape != (x.shape[0], 3):
            raise ValueError("Expected raw quality statistics q_raw[B,3].")
        quality = (q_raw - self.quality_mean) / self.quality_std
        local = x[..., 0]
        for block in self.blocks:
            local = block(local)
        global_features = self.gstt(local, quality)
        local_pooled = local.mean(dim=(2, 3))
        global_pooled = self.global_proj(global_features.mean(dim=1))
        weight = torch.sigmoid(self.fusion_gate(torch.cat(
            [local_pooled, global_pooled, quality], dim=1,
        )))
        fused = weight * local_pooled + (1 - weight) * global_pooled
        return self.fc(self.dropout(fused))


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
