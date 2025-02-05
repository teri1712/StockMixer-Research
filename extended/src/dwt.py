import pywt
import torch
import torch.nn as nn
import math


class DWT:
    def __init__(self):
        self.wavelet = "db4"

    def compute(self, x):
        # batch_size, time_steps, channels = x.shape
        coeffs = pywt.wavedec(
            x.detach().cpu().numpy(),
            self.wavelet,
            mode="periodization",
            level=1,
            axis=1,
        )
        cA, cD = coeffs
        cD = torch.tensor(cD).to(x.device)
        cA = torch.tensor(cA).to(x.device)
        return cA, cD
