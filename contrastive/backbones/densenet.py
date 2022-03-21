from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


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
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1,
                                           stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
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


class _DenseBlock(nn.Module):
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
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
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
        mode (str) - specify in which mode DenseNet is trained on,
            must be "encoder" or "classifier"
        memory_efficient (bool) - If True, uses checkpointing. Much more memory
            efficient, but slower. Default: *False*.
            See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate=32, block_config=(3, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0,
                 num_classes=1000, in_channels=1,
                 num_representation_features=256,
                 num_outputs=64,
                 mode="encoder",
                 memory_efficient=False,
                 in_shape=None,
                 depth=3):

        super(DenseNet, self).__init__()

        assert mode in {'encoder', 'decoder', 'classifier'},\
            "Unknown mode selected: %s" % mode


        self.mode = mode
        self.num_representation_features = num_representation_features
        self.num_outputs = num_outputs

        # Decoder part
        self.in_shape = in_shape
        c, h, w, d = in_shape
        self.depth = depth
        self.z_dim_h = h//2**self.depth # receptive field downsampled 2 times
        self.z_dim_w = w//2**self.depth
        self.z_dim_d = d//2**self.depth

        # First convolution
        self.features = nn.Sequential(OrderedDict([
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
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                num_output_features = max(num_features // 2, 2)
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_output_features)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_output_features

        self.num_features = num_features
        print(f"num_features = {num_features}")

        if self.mode == "classifier":
            # Final batch norm
            self.features.add_module('norm5', nn.BatchNorm3d(num_features))
            # Linear layer
            self.classifier = nn.Linear(num_features, num_classes)
        elif self.mode == "encoder":
            self.hidden_representation = nn.Linear(
                num_features, self.num_representation_features)
            self.head_projection = nn.Linear(self.num_representation_features,
                                             self.num_outputs)
        elif self.mode == "decoder":
            self.hidden_representation = nn.Linear(
                num_features, self.num_representation_features)
            self.develop = nn.Linear(self.num_representation_features,
                                     64 *self.z_dim_h * self.z_dim_w* self.z_dim_d)
            modules_decoder = []
            out_channels = 64
            for step in range(self.depth-1):
                in_channels = out_channels
                out_channels = in_channels // 2
                ini = 1 if step==0 else 0
                modules_decoder.append(('convTrans3d%s' %step, nn.ConvTranspose3d(in_channels,
                            out_channels, kernel_size=2, stride=2, padding=0, output_padding=(ini,0,0))))
                modules_decoder.append(('normup%s' %step, nn.BatchNorm3d(out_channels)))
                modules_decoder.append(('ReLU%s' %step, nn.ReLU()))
                modules_decoder.append(('convTrans3d%sa' %step, nn.ConvTranspose3d(out_channels,
                            out_channels, kernel_size=3, stride=1, padding=1)))
                modules_decoder.append(('normup%sa' %step, nn.BatchNorm3d(out_channels)))
                modules_decoder.append(('ReLU%sa' %step, nn.ReLU()))
            modules_decoder.append(('convtrans3dn', nn.ConvTranspose3d(16, 1, kernel_size=2,
                            stride=2, padding=0)))
            modules_decoder.append(('conv_final', nn.Conv3d(1, 2, kernel_size=1, stride=1)))
            self.decoder = nn.Sequential(OrderedDict(modules_decoder))


        # Init. with kaiming
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        
        if self.mode == "decoder":

            # This loads pretrained weight
            path = "/host/volatile/jc225751/Runs/33_MIDL_2022_reviews/Output/t-0.1/n-004_o-4/logs/default/version_0/checkpoints/epoch=299-step=8399.ckpt"
            pretrained = torch.load(path)
            model_dict = self.state_dict()
            for n, p in pretrained['state_dict'].items():
                if n in model_dict:
                    model_dict[n] = p
            self.load_state_dict(model_dict)

            # This freezes all layers except projection head layers
            layer_counter = 0
            for (name, module) in self.named_children():
                print(f"Module name = {name}")

            for (name, module) in self.named_children():
                if name == 'features':
                    for layer in module.children():
                        for param in layer.parameters():
                            param.requires_grad = False
                        
                        print('Layer "{}" in module "{}" was frozen!'.format(layer_counter, name))
                        layer_counter+=1
            for param in self.hidden_representation.parameters():
                param.requires_grad = False
            print('Layer "{}" in module "{}" was frozen!'.format(layer_counter, "representation"))
            for (name, param) in self.named_parameters():
                print(f"{name}: learning = {param.requires_grad}")

    def forward(self, x):
        # Eventually keep the input images for visualization
        # self.input_imgs = x.detach().cpu().numpy()
        features = self.features(x)
        if self.mode == "classifier":
            out = F.relu(features, inplace=True)
            out = F.adaptive_avg_pool3d(out, 1)
            out = torch.flatten(out, 1)
            out = self.classifier(out)
        elif self.mode == "encoder":
            out = F.relu(features, inplace=True)
            out = F.adaptive_avg_pool3d(out, 1)
            out = torch.flatten(out, 1)

            out = self.hidden_representation(out)
            out = F.relu(out, inplace=True)
            out = self.head_projection(out)
        elif self.mode == "decoder":
            out = F.relu(features, inplace=True)
            out = F.adaptive_avg_pool3d(out, 1)
            out = torch.flatten(out, 1)

            out = self.hidden_representation(out)    
            out = F.relu(out, inplace=True)
            out = self.develop(out)
            out = out.view(out.size(0), 16 * 2**(self.depth-1), self.z_dim_h, self.z_dim_w, self.z_dim_d)
            out = self.decoder(out)

        return out.squeeze(dim=1)

    def get_current_visuals(self):
        return self.input_imgs


def _densenet(arch, growth_rate, block_config, num_init_features, **kwargs):
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    return model


def densenet121(**kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks"
    <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a download progress bar to stderr
        memory_efficient (bool) - If True, uses checkpointing:
            much more memory efficient, but slower. Default: *False*.
            See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet('densenet121', 32, (6, 12, 24, 16), 64, **kwargs)
