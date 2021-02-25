import math
import torch
from torchaudio.functional import amplitude_to_DB


class Stft(object):
    def __init__(self, n_fft, window=None, return_complex=False, **kwargs):
        self.n_fft = n_fft
        self.window = torch.Tensor(window) if window is not None else None
        self.return_complex = return_complex
        self.kwargs = kwargs

    def __call__(self, x):
        x_spec = torch.stft(x, self.n_fft, 
                            window=self.window.to(x.device), 
                            return_complex=self.return_complex, 
                            **self.kwargs)
        x_spec = x_spec.transpose(-1, -3)
        return x_spec
    

class Istft(object):
    def __init__(self, n_fft, window=None, return_complex=False, **kwargs):
        self.n_fft = n_fft
        self.window = torch.Tensor(window) if window is not None else None
        self.return_complex = return_complex
        self.kwargs = kwargs

    def __call__(self, x_spec):
        x_spec = x_spec.transpose(-3, -1)
        x = torch.istft(x_spec, self.n_fft, 
                        window=self.window.to(x_spec.device), 
                        return_complex=self.return_complex, 
                        **self.kwargs)
        return x

    
class ConvToMag(object):
    def __init__(self):
        pass

    def __call__(self, x_spec):
        x_spec = x_spec.transpose(-3, -1).contiguous()
        x_spec = torch.view_as_complex(x_spec)
        x_mag = torch.abs(x_spec)
        return x_mag

    
class ConvToDb(object):
    def __init__(self):
        pass

    def __call__(self, x_spec):
        x_mag = ConvToMag()(x_spec)
        x_db = amplitude_to_DB(x_mag, multiplier=20., amin=1e-5, db_multiplier=math.log10(max(1e-5, 1.)), top_db=120.)
        return x_db
