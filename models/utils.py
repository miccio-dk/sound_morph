import math
import torch
from torchaudio.functional import amplitude_to_DB


def spec_to_db(x_spec):
    x_spec = x_spec.transpose(-3, -1).contiguous()
    x_spec = torch.view_as_complex(x_spec)
    x_mag = torch.abs(x_spec)
    x_db = amplitude_to_DB(x_mag, multiplier=20., amin=1e-5, db_multiplier=math.log10(max(1e-5, 1.)), top_db=80)
    return x_db