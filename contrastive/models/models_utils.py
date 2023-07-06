import numpy as np
import torch.nn as nn
try:
    from soma import aims
except ImportError:
    print("INFO: you are not in a brainvisa environment. Probably OK.")



def build_converter(config, concat_latent_spaces_size):
    """Builds the linear transition between the end of the encoders and 
    the common projection heads. It returns an empty Sequential if the 
    fusion is not desired (specified in config).

    Also return the size of the latent space.
    
    Arguments:
        - config: a config file
        - concat_latent_spaces_size: cumulated size of 
        all the backbones outputs."""
    
    num_representation_features = concat_latent_spaces_size
    converter = nn.Sequential()
    
    # presence of an actuial converter set in config
    if config.fusioned_latent_space_size > 0:
        num_representation_features = config.fusioned_latent_space_size
        converter.append(nn.Linear(concat_latent_spaces_size,
                                   num_representation_features))
        
        # set converter activation
        converter_activation = config.converter_activation
        if converter_activation == 'sigmoid':
            converter.append(nn.Sigmoid())
        elif converter_activation == 'relu':
            converter.append(nn.LeakyReLU())
        elif converter_activation == 'linear':
            pass
        else:
            raise ValueError(f"Such activation ({converter_activation}) is not handled for converter.")
    
    return converter, num_representation_features


def bv_checks(model, filenames):
    """Untested"""
    vol_file = f"{model.config.data[0].crop_dir}/{filenames[0]}" +\
               f"{model.config.data[0].crop_file_suffix}"
    vol = aims.read(vol_file)
    model.sample_ref_0 = np.asarray(vol)
    if not np.array_equal(model.sample_ref_0[..., 0],
                            model.sample_k[0, 0, ...]):
        raise ValueError(
            "Images files don't match!!!\n"
            f"Subject name = {filenames[0]}\n"
            f"Shape of reference file = "
            f"{model.sample_ref_0[...,0].shape}\n"
            f"Shape of file read from array = "
            f"{model.sample_k[0,0,...].shape}\n"
            f"Sum of reference file = "
            f"{model.sample_ref_0.sum()}\n"
        f"Sum of file read from array = "
            f"{model.sample_k[0,...].sum()}")


def get_projection_head_shape(config, num_representation_features):
    """Define the shapes of the projection head layers.
    Prioritize the shapes explicitely specified in config"""

    # specified in config
    if config.proj_layers_shapes is not None:
        layers_shapes = config.proj_layers_shapes
    else:
        # else, construct it in a standardized way
        if config.mode == 'encoder':
            output_shape = num_representation_features
        elif config.mode == 'classifier':
            output_shape = 2
        elif config.mode == 'regresser':
            output_shape = 1
        else:
            raise ValueError(f"Mode {config.mode} doesn't exist.")
        layers_shapes = [num_representation_features] * (config.length_projection_head - 1) + [output_shape]
    
    return layers_shapes