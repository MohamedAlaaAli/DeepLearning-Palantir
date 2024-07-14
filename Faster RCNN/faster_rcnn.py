import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16


################################################### Backbone Function #################################################
def get_backbone(pretrained = False):
    
    model = vgg16(pretrained=pretrained)
    fe = list(model.features)
    req_features = []
    k = torch.zeros((1, 3, 800, 800)).float()
    for i in fe:
        k = i(k)
        if k.size()[2] < 800//16:
            break
        req_features.append(i)

    faster_rcnn_fe_extractor = nn.Sequential(*req_features)
    return faster_rcnn_fe_extractor

################################################### Backbone Function #################################################

