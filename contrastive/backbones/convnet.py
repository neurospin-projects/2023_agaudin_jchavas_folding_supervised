from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp



class ConvNet(pl.LightningModule):
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

    def __init__(self, in_channels=1, encoder_depth=3,
                 num_representation_features=256,
                 num_outputs=64, projection_head_dims=None,
                 drop_rate=0.1, mode="encoder",
                 memory_efficient=False,
                 in_shape=None):

        super(ConvNet, self).__init__()

        assert mode in {'encoder', 'evaluation', 'decoder'},\
            "Unknown mode selected: %s" % mode


        self.mode = mode
        self.num_representation_features = num_representation_features
        self.num_outputs = num_outputs
        if projection_head_dims:
            self.projection_head_dims = projection_head_dims
        else:
            self.projection_head_dims = [num_outputs]
        self.drop_rate = drop_rate

        # Decoder part
        self.in_shape = in_shape
        c, h, w, d = in_shape
        self.encoder_depth = encoder_depth
        self.z_dim_h = h//2**self.encoder_depth # receptive field downsampled 2 times
        self.z_dim_w = w//2**self.encoder_depth
        self.z_dim_d = d//2**self.encoder_depth


        modules_encoder = []
        for step in range(encoder_depth):
            in_channels = 1 if step == 0 else out_channels
            out_channels = 16 if step == 0  else 16 * (2**step)
            modules_encoder.append(('conv%s' %step, nn.Conv3d(in_channels, out_channels,
                    kernel_size=3, stride=1, padding=1)))
            modules_encoder.append(('norm%s' %step, nn.BatchNorm3d(out_channels)))
            modules_encoder.append(('LeakyReLU%s' %step, nn.LeakyReLU()))
            modules_encoder.append(('conv%sa' %step, nn.Conv3d(out_channels, out_channels,
                    kernel_size=4, stride=2, padding=1)))
            modules_encoder.append(('norm%sa' %step, nn.BatchNorm3d(out_channels)))
            modules_encoder.append(('LeakyReLU%sa' %step, nn.LeakyReLU()))
            modules_encoder.append(('Dropout', nn.Dropout3d(p=drop_rate)))
            self.num_features = out_channels
        # flatten and reduce to the desired dimension
        modules_encoder.append(('Flatten', nn.Flatten()))
        modules_encoder.append(('Linear', 
                    nn.Linear(self.num_features*self.z_dim_h*self.z_dim_w*self.z_dim_d,
                              self.num_representation_features)))
        self.encoder = nn.Sequential(OrderedDict(modules_encoder))


        if (self.mode == "encoder") or (self.mode == 'evaluation'):
            # build a projection head
            projection_head = []
            input_size = self.num_representation_features
            for i, dim_i in enumerate(self.projection_head_dims):
                output_size = dim_i
                projection_head.append(('Linear%s' %i, nn.Linear(input_size, output_size)))
                projection_head.append(('ReLU%s' %i, nn.ReLU()))
                input_size = output_size
            projection_head.append(('Output layer' ,nn.Linear(input_size,
                                                             self.num_outputs)))
            self.projection_head = nn.Sequential(OrderedDict(projection_head))

        elif self.mode == "decoder":
            self.hidden_representation = nn.Linear(
                self.num_features, self.num_representation_features)
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
        for m in self.encoder:
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.5)
                nn.init.constant_(m.bias, 0)
        for m in self.projection_head:
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
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
        out = self.encoder(x)

        if (self.mode == "encoder") or (self.mode == 'evaluation'):
            if self.drop_rate > 0:
                out = F.dropout(out, p=self.drop_rate,
                                training=self.training)
            out = self.projection_head(out)
            

        elif self.mode == "decoder":
            out = F.relu(out, inplace=True)
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
    model = ConvNet(growth_rate, block_config, num_init_features, **kwargs)
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
