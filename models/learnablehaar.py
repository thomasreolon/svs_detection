import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

"""
never tried to implement
"""
# original code from micronet

class BinaryActivation(Function):
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        output = torch.sign(input)
        output[output == 0] = 1
        return output

    @staticmethod
    def backward(self, grad_output):
        (input,) = self.saved_tensors
        grad_input = grad_output.clone()
        return grad_input

class WeightsToBinary(Function):
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        output = torch.sign(input)
        output[output == 0] = 1
        return output

    @staticmethod
    def backward(self, grad_output):
        (input,) = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1.0)] = 0
        grad_input[input.le(-1.0)] = 0
        return grad_input

class BiasToInt(Function):
    @staticmethod
    def forward(self, input):
        output = torch.round(input)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class QuantConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        super(QuantConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
    
    def float_weights_to_binary(weight):
        weight = WeightsToBinary.apply(weight)

    def float_bias_to_integer(bias):
        bias = BiasToInt.apply(bias)

    def forward(self, input):
        # weights only 0 or 1
        bin_weight = self.float_weights_to_binary(self.weight)

        # bias only integer
        int_bias   = self.float_bias_to_integer(self.bias)

        # torch convolution
        output = F.conv2d(
            input,
            bin_weight,
            int_bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

        # binarize output
        output = BinaryActivation.apply(output)

        return output


class HaarBlock(nn.Module):
    def __init__(self, c_in, c_out, k=5, first=False):
        super().__init__()
        k = k//2 *2 +1
        h = k*2
        self.first = first

        self.conv1 = QuantConv2d(c_in, h, k, 1, k//2, groups=c_in)
        self.conv2 = QuantConv2d(h, c_out, 1, 1, k//2)
        self.conv3 = QuantConv2d(c_out, c_out, 1, 3, 1)

    def forward(self, input):
        if self.first:
            input = BinaryActivation.apply(input-input.mean())
        y = self.conv1(input)
        y = self.conv2(y)
        return self.conv3(y)


