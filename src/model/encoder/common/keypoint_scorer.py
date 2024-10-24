import torch
import torch.nn as nn
import torch.nn.functional as F

class KeypointScorer(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(KeypointScorer, self).__init__()

        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)


    def forward(self, features, h, w):

        device = features.device

        features = features[0]
        N, input_dim, _, _ = features.shape

        self.upsampler = nn.Upsample(size=(h, w), mode='bilinear', align_corners=False)
        self.betas = nn.Parameter(torch.ones(N, device=device))

        upsampled_features = self.upsampler(features)
        alphas = F.softmax(self.betas, dim=0)
        weighted_features = alphas.view(N, 1, 1, 1) * upsampled_features

        score_maps = self.mlp(weighted_features.permute(0, 2, 3, 1).reshape(-1, input_dim))
        score_maps = F.softmax(score_maps.view(N, -1), dim=-1)

        return score_maps.view(N, h, w), alphas
