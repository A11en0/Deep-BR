import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, feature_dim, label_num, device, model_args=None):
        super(Model, self).__init__()
        self.fcs = nn.ModuleList()
        self.device = device

        self.layer = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, label_num)
        )

    def forward(self, x):
        x = self.view_feature(x)
        outputs = self.layer(x)
        outputs = torch.sigmoid(outputs)
        return outputs

    def view_feature(self, x):
        outputs = torch.tensor([]).to(self.device)
        for idx, key in enumerate(x):
            outputs = torch.cat((outputs, x[key]), dim=1)
        return outputs

if __name__ == '__main__':
    f1 = torch.randn(1000, 20)
    f2 = torch.randn(1000, 20)

