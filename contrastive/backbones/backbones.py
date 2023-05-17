import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from torch import Tensor
from collections import OrderedDict


class _DropoutNd(nn.Module):
    __constants__ = ['p', 'inplace']
    p: float
    inplace: bool

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super(_DropoutNd, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def extra_repr(self) -> str:
        return 'p={}, inplace={}'.format(self.p, self.inplace)


class Dropout3d_always(_DropoutNd):
    r"""Randomly zero out entire channels (a channel is a 3D feature map,
    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 3D tensor :math:`\text{input}[i, j]`).
    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    Alwyas applies dropout also during evaluation

    As described in the paper
    `Efficient Object Localization Using Convolutional Networks`_ ,
    if adjacent pixels within feature maps are strongly correlated
    (as is normally the case in early convolution layers) then i.i.d. dropout
    will not regularize the activations and will otherwise just result
    in an effective learning rate decrease.

    In this case, :func:`nn.Dropout3d` will help promote independence between
    feature maps and should be used instead.

    Args:
        p (float, optional): probability of an element to be zeroed.
        inplace (bool, optional): If set to ``True``, will do this operation
            in-place

    Shape:
        - Input: :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)`.
        - Output: :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)`
                  (same shape as input).

    Examples::

        >>> m = nn.Dropout3d(p=0.2)
        >>> input = torch.randn(20, 16, 4, 32, 32)
        >>> output = m(input)

    .. _Efficient Object Localization Using Convolutional Networks:
       https://arxiv.org/abs/1411.4280
    """

    def forward(self, input: Tensor) -> Tensor:
        return F.dropout3d(input, self.p, True, self.inplace)


class ConvNet(pl.LightningModule):
    r"""3D-ConvNet model class, based on

    Attributes:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first
            convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate
        num_classes (int) - number of classification classes
            (if 'classifier' mode)
        in_channels (int) - number of input channels (1 for sMRI)
        mode (str) - specify in which mode DenseNet is trained on,
            must be "encoder" or "classifier"
        memory_efficient (bool) - If True, uses checkpointing. Much more memory
            efficient, but slower. Default: *False*.
            See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, in_channels=1, encoder_depth=3,
                 num_representation_features=256,
                 drop_rate=0.1, memory_efficient=False,
                 in_shape=None):

        super(ConvNet, self).__init__()

        self.num_representation_features = num_representation_features
        self.drop_rate = drop_rate

        # Decoder part
        self.in_shape = in_shape
        c, h, w, d = in_shape
        self.encoder_depth = encoder_depth

        # receptive field downsampled 2 times
        self.z_dim_h = h//2**self.encoder_depth
        self.z_dim_w = w//2**self.encoder_depth
        self.z_dim_d = d//2**self.encoder_depth

        modules_encoder = []
        for step in range(encoder_depth):
            in_channels = 1 if step == 0 else out_channels
            out_channels = 16 if step == 0 else 16 * (2**step)
            modules_encoder.append(
                ('conv%s' % step,
                 nn.Conv3d(in_channels, out_channels,
                           kernel_size=3, stride=1, padding=1)
                 ))
            modules_encoder.append(
                ('norm%s' % step, nn.BatchNorm3d(out_channels)))
            modules_encoder.append(('LeakyReLU%s' % step, nn.LeakyReLU()))
            modules_encoder.append(
                ('DropOut%s' % step, nn.Dropout3d(p=drop_rate)))
            modules_encoder.append(
                ('conv%sa' % step,
                 nn.Conv3d(out_channels, out_channels,
                           kernel_size=4, stride=2, padding=1)
                 ))
            modules_encoder.append(
                ('norm%sa' % step, nn.BatchNorm3d(out_channels)))
            modules_encoder.append(('LeakyReLU%sa' % step, nn.LeakyReLU()))
            modules_encoder.append(
                ('DropOut%sa' % step, nn.Dropout3d(p=drop_rate)))
            self.num_features = out_channels
        # flatten and reduce to the desired dimension
        modules_encoder.append(('Flatten', nn.Flatten()))
        modules_encoder.append(
            ('Linear',
             nn.Linear(
                 self.num_features*self.z_dim_h*self.z_dim_w*self.z_dim_d,
                 self.num_representation_features)
             ))
        self.encoder = nn.Sequential(OrderedDict(modules_encoder))

    def forward(self, x):
        out = self.encoder(x)
        return out.squeeze(dim=1)



def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output
    return bn_function


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size,
                 drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(
            num_input_features, track_running_stats=True)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1,
                                           stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(
            bn_size * growth_rate, track_running_stats=True)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.memory_efficient and any(prev_feature.requires_grad
                                         for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))

        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)

        return new_features


class _DenseBlock(pl.LightningModule):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(
            num_input_features, track_running_stats=True))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features,
                                          num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet(pl.LightningModule):
    r"""3D-DenseNet model class, based on
    `"Densely Connected Convolutional Networks"
    <https://arxiv.org/pdf/1608.06993.pdf>`_

    Attributes:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first
            convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
            (if 'classifier' mode)
        in_channels (int) - number of input channels (1 for sMRI)
        num_representation_features (int) - size of latent space
        num_outputs (int) -  size of output space
        projection_head_type (str) - Type of projection head
            (either "linear\" or "non-linear")
        mode (str) - specify in which mode DenseNet is trained on,
            must be "encoder" or "classifier" or "decoder"
        memory_efficient (bool) - If True, uses checkpointing. Much more memory
            efficient, but slower. Default: *False*.
            See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate=32, block_config=(3, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0,
                 in_channels=1, num_representation_features=256,
                 memory_efficient=False, in_shape=None):

        super(DenseNet, self).__init__()

        self.num_representation_features = num_representation_features
        self.in_shape = in_shape
        self.drop_rate = drop_rate

        # First convolution
        self.encoder = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(in_channels, num_init_features, kernel_size=7,
                                stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.encoder.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                num_output_features = max(num_features // 2, 2)
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_output_features)
                self.encoder.add_module('transition%d' % (i + 1), trans)
                num_features = num_output_features
            print("NUM FEATURES", num_features)
        
        print(640*30)
        #print(math.prod(num_features))
        self.encoder.add_module('Flatten', nn.Flatten())
        self.encoder.add_module('Linear', nn.Linear(num_features, self.num_representation_features))


        # Init. with kaiming
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.5)
                nn.init.constant_(m.bias, 0)
        for m in self.encoder:
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.5)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Eventually keep the input images for visualization
        # self.input_imgs = x.detach().cpu().numpy()
        out = self.encoder(x)
        return out.squeeze(dim=1)
