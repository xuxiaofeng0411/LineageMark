import torch
import torch.nn as nn
from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import QuantLinear
from bitsandbytes.nn import Linear8bitLt


LINEAR_LAYER_TYPES = (nn.Linear, Linear8bitLt, QuantLinear)


class WrappedGPT:
    """
    Track per-input-channel activation statistics for a wrapped linear module.
    """

    def __init__(self, layer, layer_id=0, layer_name="none", is_int8=False):
        self.layer = layer
        self.layer_id = layer_id
        self.layer_name = layer_name

        self.dev, self.rows, self.columns = self._read_layer_layout(is_int8)
        self.scaler_row = torch.zeros(self.columns, device=self.dev)
        self.nsamples = 0

    def _read_layer_layout(self, is_int8):
        if is_int8:
            qweight = self.layer.qweight
            return qweight.device, qweight.data.shape[-1], qweight.data.shape[0] * 4

        weight = self.layer.weight
        return weight.device, weight.data.shape[0], weight.data.shape[1]

    def _reshape_linear_input(self, inp):
        if not isinstance(self.layer, LINEAR_LAYER_TYPES):
            return inp
        if inp.dim() == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        return inp.t()

    @staticmethod
    def _ensure_batch_axis(inp):
        if inp.dim() == 2:
            return inp.unsqueeze(0)
        return inp

    def _merge_batch_statistics(self, inp, batch_size):
        self.scaler_row *= self.nsamples / (self.nsamples + batch_size)
        self.nsamples += batch_size
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples

    def add_batch(self, inp, out):
        inp = self._ensure_batch_axis(inp)
        batch_size = inp.shape[0]
        inp = self._reshape_linear_input(inp)
        self._merge_batch_statistics(inp, batch_size)
