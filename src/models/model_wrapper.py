import torch
from torch import nn

import src.models.ecg_resnet

FEATURE_DIM = 3
RR_INTERVAL_WIDTH = 150

class ModelWrapper(nn.Module):
    def __init__(self, arch, pretrained_backbone_path, use_feature=False):
        super().__init__()
        self.arch = arch
        self.pretrained_backbone_path = pretrained_backbone_path
        self.use_feature = use_feature
        self.num_classes = 2
        
        self.feature_layer = nn.Sequential(
            nn.Linear(FEATURE_DIM, FEATURE_DIM),
            nn.ReLU(),
            nn.Linear(FEATURE_DIM, FEATURE_DIM)
        )
        
        self.rr_layer = nn.Sequential(
            nn.Linear(RR_INTERVAL_WIDTH, RR_INTERVAL_WIDTH),
            nn.ReLU(),
            nn.Linear(RR_INTERVAL_WIDTH, RR_INTERVAL_WIDTH)
        )

        if pretrained_backbone_path is None:
            self.backbone = getattr(src.models.ecg_resnet, arch)()
            if use_feature:
                rep_dim = self.backbone.rep_dim + (FEATURE_DIM + RR_INTERVAL_WIDTH)
            else:
                rep_dim = self.backbone.rep_dim
            self.classifier = nn.Linear(rep_dim, self.num_classes)
        else:
            backbone_model = torch.load(pretrained_backbone_path)
            self.backbone = backbone_model.backbone
            self.classifier = backbone_model.classifier
    
    def forward(self, x, feature=None, rr=None):
        rep = self.backbone(x)
        if self.use_feature:
            feature_rep = self.feature_layer(feature)
            rr_rep = self.rr_layer(rr)
            rep = torch.cat((rep, feature_rep, rr_rep), dim=1)
        cls = self.classifier(rep)
        return cls
