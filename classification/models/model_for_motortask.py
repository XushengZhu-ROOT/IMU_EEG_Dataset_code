"""CBraMod backbone with a three-layer MLP classification head over all patch features.

Adapted from the model_for_*.py files of CBraMod (https://github.com/wjq-learning/CBraMod, MIT licence).
"""
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from .cbramod import CBraMod


class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()
        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30,
            n_layer=12, nhead=8
        )
        if param.use_pretrained_weights:
            map_location = torch.device(f'cuda:{param.cuda}')
            self.backbone.load_state_dict(torch.load(param.foundation_dir, map_location=map_location))
        self.backbone.proj_out = nn.Identity()

        self.classifier = nn.Sequential(
            Rearrange('b c s d -> b (c s d)'),
            nn.Linear(param.channel_size * param.window_size * 200, param.window_size * 200),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(param.window_size * 200, 200),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(200, param.num_of_classes),
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.classifier(feats)
