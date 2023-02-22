from sys import platform
from micronet.compression.quantization.wbwtab.quantize import prepare
from micronet.compression.quantization.wqaq.dorefa.quantize import prepare as prepare8bit
import torch
from torch.quantization import quantize_dynamic




def quantize(model, method):
    if method=='no':
        return model
    if method=='binary':
        # it uses soooo much more GPU RAM...
        return prepare(model, quant_inference=True)  # A & W are the type of quantization, default 2  # quant inference if you do not train it
    if method=='8bit':
        return prepare8bit(model, quant_inference=True)
    if method=='8bit_torch':
        # qnnpack --> ARM
        # fbgemm --> x86 linux/mac
        # None --> Win
        engine =  'fbgemm' if (platform == "linux" or platform == "linux2") else 'qnnpack'
        torch.backends.quantized.engine = engine 
        return quantize_dynamic(model, dtype=torch.qint8, inplace=False)
    else:
        raise NotImplementedError(f'Quantization method {method} not supported')
