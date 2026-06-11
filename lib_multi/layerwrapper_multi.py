import torch
import torch.nn as nn
from bitsandbytes.nn import Linear8bitLt
from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import QuantLinear

# Define WrappedGPT class
class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none", is_int8=False):
        self.layer = layer
        if is_int8:
            self.dev = self.layer.qweight.device
            self.rows = layer.qweight.data.shape[-1]
            self.columns = layer.qweight.data.shape[0] * 4
        else:
            self.dev = self.layer.weight.device
            self.rows = layer.weight.data.shape[0]
            self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        # print(self.layer_name)
        # print(inp.shape)
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        # print(inp.shape)
        tmp = inp.shape[0]
        # print(f'tmp: {tmp}')
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, Linear8bitLt) or isinstance(self.layer, QuantLinear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        # print(inp.shape)
        # print(self.scaler_row.shape)
        # print(f'row: {self.rows}')
        # print(f'column: {self.columns}')
        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        # inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples

