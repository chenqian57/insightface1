import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter

import sys
sys.path.append('/home/qiujing/cqwork/insightface/recognition/_evaluation_/ijb/1')

from iresnet import (iresnet18, iresnet34, iresnet50, iresnet100,
                          iresnet200)


def get_model(name, **kwargs):
    # resnet
    if name == "ir18":
        return iresnet18(False, **kwargs)
    elif name == "ir34":
        return iresnet34(False, **kwargs)
    elif name == "ir50":
        return iresnet50(False, **kwargs)
    elif name == "ir100":
        return iresnet100(False, **kwargs)
    elif name == "ir200":
        return iresnet200(False, **kwargs)
    else:
        raise ValueError()






