import torch

class FcSiteStructure(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.fc(x)
