import math
import torch
from torch.optim import Adam
from torch_optimizer import Yogi
from torchaudio.functional import amplitude_to_DB


def pick_optimizer(optim_name):
    return {
        'adam': Adam,
        'yogi': Yogi
    }.get(optim_name, Adam)

def spec_to_db(x_spec, top_db=80, amin=1e-5):
    x_spec = x_spec.transpose(-3, -1).contiguous()
    x_spec = torch.view_as_complex(x_spec)
    x_mag = torch.abs(x_spec)
    x_db = amplitude_to_DB(x_mag, multiplier=20., amin=amin, db_multiplier=math.log10(max(amin, 1.)), top_db=top_db)
    return x_db