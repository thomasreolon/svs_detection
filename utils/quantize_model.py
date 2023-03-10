from sys import platform
from micronet.compression.quantization.wbwtab.quantize import prepare as preparebin
from micronet.compression.quantization.wqaq.dorefa.quantize import prepare as prepare8bit
import torch
from torch.quantization import quantize_dynamic

from bnn import BConfig, prepare_binary_model
from bnn.ops import BasicInputBinarizer, BasicScaleBinarizer, XNORWeightBinarizer



def quantize(model, method):
    if method=='no':
        return model
    if method=='8bit':
        # quant_inference true: does not quantize weights (only activations)
        return prepare8bit(model, quant_inference=False) # DoReFa-Net paper
    if method=='binary':
        # A & W are the type of quantization, default 2  # quant inference if you do not train it
        return preparebin(model, A=2, W=2, quant_inference=False)  # XNorNet paper


    if method=='4bit':
        return prepare8bit(model, a_bits=4, w_bits=4, quant_inference=False)
    if method=='2bit':
        return prepare8bit(model, a_bits=2, w_bits=2, quant_inference=False)
    if method=='1bit':
        return prepare8bit(model, a_bits=1, w_bits=1, quant_inference=False)


    if method=='binary_bnn': #### BACKWARD FAILS
        bconfig = BConfig(
            activation_pre_process = BasicInputBinarizer,
            activation_post_process = BasicScaleBinarizer,
            weight_pre_process = XNORWeightBinarizer.with_args(center_weights=True)
        )
        return prepare_binary_model(model, bconfig)
    if method=='8bit_torch':  #### NO SUPP HARDWARE
        # qnnpack --> ARM
        # fbgemm --> x86 linux/mac
        # None --> Win
        engine =  'fbgemm' if (platform == "linux" or platform == "linux2") else 'qnnpack'
        torch.backends.quantized.engine = engine 
        return quantize_dynamic(model, dtype=torch.qint8, inplace=False)
    else:
        raise NotImplementedError(f'Quantization method {method} not supported')
