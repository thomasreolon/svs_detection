import torch.nn as nn
import torch

import torch.nn as nn
import torch


def correct_pad(input_shape, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling
    Args:
        input_shape ([tuple/int]): [Input size]
        kernel_size ([tuple/int]): [Kernel size]
    Returns:
        [tuple]: [Padding coeffs]
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_shape[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_shape[0] % 2, 1 - input_shape[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
 
    return (int(correct[1] - adjust[1]), int(correct[1]), int(correct[0] - adjust[0]), int(correct[0]))


def preprocess_input(x, **kwargs):
    """Normalise channels between [-1, 1]
    Args:
        x ([Tensor]): [Contains the image, number of channels is arbitrary]
    Returns:
        [Tensor]: [Channel-wise normalised tensor]
    """

    return (x / 128.) - 1


def get_xpansion_factor(t_zero, beta, block_id, num_blocks):
    """Compute expansion factor based on the formula from the paper
    Args:
        t_zero ([int]): [initial expansion factor]
        beta ([int]): [shape factor]
        block_id ([int]): [id of the block]
        num_blocks ([int]): [number of blocks in the network]
    Returns:
        [float]: [computed expansion factor]
    """
    return (t_zero * beta) * block_id / num_blocks + t_zero * (num_blocks - block_id) / num_blocks

class ReLUMax(torch.nn.Module):
    def __init__(self, max):
        super(ReLUMax, self).__init__()
        self.max = max
        self.relu = torch.nn.ReLU(inplace=True)
    
    def forward(self, x):
        return torch.clamp(self.relu(x), max = self.max)

class HSwish(torch.nn.Module):
    def __init__(self):
        super(HSwish, self).__init__()
    
    def forward(self, x):
        return x * nn.ReLU6(inplace=True)(x + 3) / 6

class SEBlock(torch.nn.Module):
    """Implements squeeze-and-excitation block"""

    def __init__(self, in_channels, out_channels, h_swish=True):
        """Constructor of SEBlock
        Args:
            in_channels ([int]): [Input number of channels]
            out_channels ([int]): [Output number of channels]
            h_swish (bool, optional): [Whether to use the h_swish or not]. Defaults to True.
        """
        super(SEBlock, self).__init__()

        self.glob_pooling = lambda x: nn.functional.avg_pool2d(x, x.size()[2:])

        self.se_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            padding="same",
            bias=False,
        )

        self.se_conv2 = nn.Conv2d(
            out_channels,
            in_channels,
            kernel_size=1,
            bias=False,
            padding="same"
        )

        if h_swish:
            self.activation = HSwish()
        else:
            self.activation = ReLUMax(6)

    def forward(self, x):
        """Executes SE Block
        Args:
            x ([Tensor]): [input tensor]
        Returns:
            [Tensor]: [output of squeeze-and-excitation block]
        """
        inp = x
        x = self.glob_pooling(x)
        x = self.se_conv(x)
        x = self.activation(x)
        x = self.se_conv2(x)
        x = torch.sigmoid(x)
        x = x.expand_as(inp) * inp

        return x


class DepthwiseConv2d(torch.nn.Conv2d):
    """Depthwise 2D conv
    Args:
        torch ([Tensor]): [Input tensor for convolution]
    """

    def __init__(self,
                 in_channels,
                 depth_multiplier=1,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False,
                 padding_mode='zeros'
                 ):
        out_channels = in_channels * depth_multiplier
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode
        )


class SeparableConv2d(torch.nn.Module):
    """Implements SeparableConv2d"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 activation=torch.nn.functional.relu,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 padding_mode='zeros',
                 depth_multiplier=1,
                 ):
        """Constructor of SeparableConv2d
        Args:
            in_channels ([int]): [Input number of channels]
            out_channels ([int]): [Output number of channels]
            kernel_size (int, optional): [Kernel size]. Defaults to 3.
            stride (int, optional): [Stride for conv]. Defaults to 1.
            padding (int, optional): [Padding for conv]. Defaults to 0.
            dilation (int, optional): []. Defaults to 1.
            bias (bool, optional): []. Defaults to True.
            padding_mode (str, optional): []. Defaults to 'zeros'.
            depth_multiplier (int, optional): [Depth multiplier]. Defaults to 1.
        """
        super().__init__()

        self._layers = torch.nn.ModuleList()

        depthwise = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            dilation=1,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

        spatialConv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding="same",
            dilation=dilation,
            # groups=in_channels,
            bias=bias,
            padding_mode=padding_mode
        )

        bn = torch.nn.BatchNorm2d(
            out_channels,
            eps=1e-3,
            momentum=0.999
        )
        
        self._layers.append(depthwise)
        self._layers.append(spatialConv)
        self._layers.append(bn)
        self._layers.append(activation)

    def forward(self, x):
        """Executes SeparableConv2d block
        Args:
            x ([Tensor]): [Input tensor]
        Returns:
            [Tensor]: [Output of convolution]
        """
        for l in self._layers:
            x = l(x)

        return x



class PhiNetConvBlock(nn.Module):
    """Implements PhiNet's convolutional block"""

    def __init__(self, in_shape, expansion, stride, filters, has_se, block_id=None, res=True, h_swish=True, k_size=3, dp_rate=0.05):
        """Defines the structure of the PhiNet conv block
        Args:
            in_shape ([Tuple]): [Input shape, as returned by Tensor.shape]
            expansion ([Int]): [Expansion coefficient]
            stride ([Int]): [Stride for conv block]
            filters ([Int]): [description]
            block_id ([Int]): [description]
            has_se (bool): [description]
            res (bool, optional): [description]. Defaults to True.
            h_swish (bool, optional): [description]. Defaults to True.
            k_size (int, optional): [description]. Defaults to 3.
        """
        super(PhiNetConvBlock, self).__init__()
        self.skip_conn = False

        self._layers = torch.nn.ModuleList()
        in_channels = in_shape[0]
        # Define activation function
        if h_swish:
            activation = HSwish()
        else:
            activation = ReLUMax(6)

        if block_id:
            # Expand
            conv1 = nn.Conv2d(
                in_channels, int(expansion * in_channels),
                kernel_size=1,
                padding="same",
                bias=False,
            )

            bn1 = nn.BatchNorm2d(
                int(expansion * in_channels),
                eps=1e-3,
                momentum=0.999,
            )

            self._layers.append(conv1)
            self._layers.append(bn1)
            self._layers.append(activation)
        
        self._layers.append(nn.Dropout2d(dp_rate))

        d_mul = 1
        in_channels_dw = int(expansion * in_channels) if block_id else in_channels
        out_channels_dw = in_channels_dw * d_mul
        dw1 = DepthwiseConv2d(
            in_channels=in_channels_dw,
            depth_multiplier=d_mul,
            kernel_size=k_size,
            stride=stride,
            bias=False,
            padding="same" if stride == 1 else k_size//2,
            # name=prefix + 'depthwise'
        )

        bn_dw1 = nn.BatchNorm2d(
            out_channels_dw,
            eps=1e-3,
            momentum=0.999,
        )

        self._layers.append(dw1)
        self._layers.append(bn_dw1)
        self._layers.append(activation)

        if has_se:
            num_reduced_filters = max(1, int(in_channels * 0.25))
            se_block = SEBlock(int(expansion * in_channels), num_reduced_filters, h_swish=h_swish)
            self._layers.append(se_block)

        conv2 = nn.Conv2d(
            in_channels=int(expansion * in_channels),
            out_channels=filters,
            kernel_size=1,
            padding="same",
            bias=False,
        )

        bn2 = nn.BatchNorm2d(
            filters,
            eps=1e-3,
            momentum=0.999,
        )

        self._layers.append(conv2)
        self._layers.append(bn2)

        if res and in_channels == filters and stride == 1:
            self.skip_conn = True

    def forward(self, x):
        """Executes PhiNet's convolutional block
        Args:
            x ([Tensor]): [Conv block input]
        Returns:
            [Tensor]: [Output of convolutional block]
        """
        if self.skip_conn:
            inp = x

        for l in self._layers:
            x = l(x)

        if self.skip_conn:
            return x + inp

        return x



class PhiNet(nn.Module):
    def __init__(self, res=96, in_channels=3, out_layers=[-1], alpha=0.2,  B0=7, beta=1.0, squeeze_excite=False, conv2d_input=False, h_swish=False, t_zero=6, first_conv_filters=48, b1_filters=24, b2_filters=48, 
                 downsampling_layers=[5, 7], conv5_percent=0, first_conv_stride=2, residuals=True, pool=False):
        """Generates PhiNets architecture
        Args:
            res (int, optional): [base network input resolution]. Defaults to 96.
            in_channels (int, optional): [channel number for the input tensor]. Defaults to 3.
            out_layers (list, optional): [which layers' outputs should be returned]. Defaults to last.
            alpha (float, optional): [base network width multiplier]. Defaults to 0.35.
            B0 (int, optional): [base network number of blocks]. Defaults to 7.
            beta (float, optional): [shape factor]. Defaults to 1.0.
            t_zero (int, optional): [initial expansion factor]. Defaults to 6.
            first_conv_filters (int, optional): [description]. Defaults to 48.
            b1_filters (int, optional): [description]. Defaults to 24.
            b2_filters (int, optional): [description]. Defaults to 48.
            squeeze_excite (bool, optional): [SE blocks - Enable for performance, disable for compatibility]. Defaults to False.
            downsampling_layers (list, optional): [Indices of downsampling blocks (between 5 and B0)]. Defaults to [5,7].
            conv5_percent (int, optional): [if last conv layers should use a kernel of 5]. Defaults to 0.
            first_conv_stride (int, optional): [Downsampling at the network input - first conv stride]. Defaults to 2.
            h_swish (bool, optional): [Approximate Hswish activation - Enable for performance, disable for compatibility (gets replaced by relu6)]. Defaults to False.
            residuals (bool, optional): [disable residual connections to lower ram usage - residuals]. Defaults to True.
            conv2d_input (bool, optional): [use a normal convolution before phinet]. Defaults to False.
            pool (bool, optional): [use pooling to downsaple]. Defaults to False.
        """
        super(PhiNet, self).__init__()
        self.out_layers = out_layers
        counter, self._out_dims = 1, []

        num_blocks = round(B0)

        self._layers = torch.nn.ModuleList()

        # Define self.activation function
        if h_swish:
            activation = HSwish()
        else:
            activation = ReLUMax(6)
            
        mp = nn.MaxPool2d((2, 2))

        # use BN at beginning
        bn_base = nn.BatchNorm2d(int(in_channels))
        self._layers.append(bn_base)

        # first block
        if not conv2d_input:            
            sep1 = SeparableConv2d(
                in_channels,
                int(first_conv_filters * alpha),
                kernel_size=3,
                stride=(first_conv_stride, first_conv_stride),
                padding=1,
                bias=False,
                activation=activation
            )

            self._layers.append(sep1)
            # self._layers.append(activation)

            block1 = PhiNetConvBlock(
                in_shape=(int(first_conv_filters * alpha), res / first_conv_stride, res / first_conv_stride),
                filters=int(b1_filters * alpha),
                stride=1,
                expansion=1,
                has_se=False,
                res=residuals,
                h_swish=h_swish
            )
            
            self._layers.append(block1)
            if counter in out_layers:
                self._out_dims.append(int(b1_filters * alpha))
        else:
            for i in range(len(out_layers)): # 
                out_layers[i] = out_layers[i]-1 if out_layers[i]>0 else out_layers[i]
            c1 = nn.Conv2d(
                in_channels,
                int(b1_filters*alpha),
                kernel_size=(3,3),
                bias=False
            )
            
            self._layers.append(c1)
            self._layers.append(activation)

        # second block
        counter += 1
        block2 = PhiNetConvBlock(
            (int(b1_filters * alpha), res / first_conv_stride, res / first_conv_stride),
            filters=int(b1_filters * alpha),
            stride=2 if (not pool) else 1,
            expansion=get_xpansion_factor(t_zero, beta, 1, num_blocks),
            block_id=1,
            has_se=squeeze_excite,
            res=residuals,
            h_swish=h_swish
        )
        if counter in out_layers:
            self._out_dims.append(int(b1_filters * alpha))
        
        counter += 1
        block3 = PhiNetConvBlock(
            (int(b1_filters * alpha), res / first_conv_stride / 2, res / first_conv_stride / 2),
            filters=int(b1_filters * alpha),
            stride=1,
            expansion=get_xpansion_factor(t_zero, beta, 2, num_blocks),
            block_id=2,
            has_se=squeeze_excite,
            res=residuals,
            h_swish=h_swish
        )
        if counter in out_layers:
            self._out_dims.append(int(b1_filters * alpha))

        counter += 1
        block4 = PhiNetConvBlock(
            (int(b1_filters * alpha), res / first_conv_stride / 2, res / first_conv_stride / 2),
            filters=int(b2_filters * alpha),
            stride=2 if (not pool) else 1,
            expansion=get_xpansion_factor(t_zero, beta, 3, num_blocks),
            block_id=3,
            has_se=squeeze_excite,
            res=residuals,
            h_swish=h_swish
        )
        if counter in out_layers:
            self._out_dims.append(int(b2_filters * alpha))
        counter += 1

        self._layers.append(block2)
        if pool:
            self._layers.append(mp)
        self._layers.append(block3)
        self._layers.append(block4)
        if pool:
            self._layers.append(mp)

        
        block_id = 4
        block_filters = b2_filters
        spatial_res = res / first_conv_stride / 4
        in_channels_next = int(b2_filters * alpha)
        while num_blocks >= block_id:
            if block_id in downsampling_layers:
                block_filters *= 2
                if pool:
                    self._layers.append(mp)
            
            pn_block = PhiNetConvBlock(
                    (in_channels_next, spatial_res, spatial_res),
                    filters=int(block_filters * alpha),
                    stride=(2 if (block_id in downsampling_layers) and (not pool) else 1),
                    expansion=get_xpansion_factor(t_zero, beta, block_id, num_blocks),
                    block_id=block_id,
                    has_se=squeeze_excite,
                    res=residuals,
                    h_swish=h_swish,
                    k_size=(5 if (block_id / num_blocks) > (1 - conv5_percent) else 3)
            )
            
            if counter in out_layers:
                self._out_dims.append(int(block_filters * alpha))
            counter += 1

            self._layers.append(pn_block)
            in_channels_next = int(block_filters * alpha)
            spatial_res = spatial_res / 2 if block_id in downsampling_layers else spatial_res
            block_id += 1

        if counter in out_layers or -1 in out_layers:
            self._out_dims.append(int(block_filters * alpha))


    
    def forward(self, x):
        """Executes PhiNet network
        Args:
            x ([Tensor]): [input batch]
        """
        outputs = []

        i = 0
        for l in self._layers:
            x = l(x)
            if not isinstance(l, PhiNetConvBlock): continue
            if i+1 in self.out_layers:
                outputs.append(x)
            i += 1

        if -1 in self.out_layers:
            outputs.append(x)


        return outputs


class Resnet(nn.Module):
    def __init__(self, c1=1, c2=32):
        super().__init__()
        self._out_dims = [64, 128, c2]
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=False)
        model.conv1 = nn.Conv2d(c1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.layer3[5] = nn.Conv2d(256, c2, 3, padding='same')

        self.model = nn.ModuleList([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3])

    def forward(self, x):
        y = []
        for i, m in enumerate(self.model):
            x = m(x)
            if i in {4,5,6}:
                y.append(x)
        return y


if __name__=='__main__':

    net = PhiNet(160, out_layers=[5,7,-1], B0=7, alpha=0.35, in_channels=1, pool=False)
    s1 = [v.shape for v in net(torch.rand(1,1,120,160))]
    s2 = [v.shape for v in net(torch.rand(1,1,240,320))]

    print(s1, '\n', s2)

    assert all([v1[2]*2==v2[2] and v1[3]*2==v2[3] for v1,v2 in zip(s1,s2) ]), "NOOOOOOOOOOOOOOO"


