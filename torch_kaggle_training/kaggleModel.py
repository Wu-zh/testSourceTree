# -*- coding: utf-8 -*-
# author: wuzhuohao
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.getcwd()
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F



class KaggleModel(nn.Module):
    def __init__(self, in_channel=3, num_layer=5) -> None:
        super().__init__()
        in_dim = in_channel
        out_dim = 32
        layers: list[nn.Module] = []
        for i in range(num_layer):
            layers.extend([
                nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=3),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2)
            ])
            in_dim = out_dim
            out_dim = out_dim * 2
        out_dim = in_dim
        layers.pop()
        layers.append(nn.MaxPool2d(kernel_size=4))
        self.features = nn.Sequential(*layers)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        self.gender_classifier = nn.Sequential(
            nn.Linear(out_dim, out_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(out_dim, 1, bias=True),
            nn.Sigmoid()
        )

        self.age_classifier = nn.Sequential(
            nn.Linear(out_dim, out_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(out_dim, 1, bias=True),
            nn.ReLU(inplace=True)
        )
            
    
    def init_weight(self):
        for m in self.modules:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, 1)
        gender = self.gender_classifier(out)   
        gender = gender.view(-1)
        age = self.age_classifier(out)
        age = age.view(-1)
        
        
        return age, gender



if __name__ == "__main__":
    # print network structure
    from torchsummary import summary
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = KaggleModel().to(device=device)
    summary(net, (3, 128, 128), batch_size=1, device=device)
    


