import torch

from captum.attr import LayerGradCam

from contrastive.data.utils import change_list_device


# grad cam maps
def compute_grad_cam(loader, model, with_labels=True):
    """Return the gradients map of model for all subjects contained in loader.
    The gradient map is done only on the last conv layer of the model.
    
    Returns a dictionary with two keys, one for each label."""
    filenames_list = []
    attribution_list = []
    grad_cam_dict = {}

    # map for positive and negative classes
    for target in [0,1]:
        for batch in loader:

            if with_labels:
                inputs, filenames, labels, _ = \
                    model.get_full_inputs_from_batch_with_labels(batch)
            else:
                inputs, filenames = model.get_full_inputs_from_batch(batch)

            inputs = change_list_device(inputs, 'cuda')
            input = inputs[0][:, 0, ...]

            GC = LayerGradCam(model, model.backbones[0].encoder.conv2a)
            attributions = GC.attribute(input.unsqueeze(0), target=target, relu_attributions=True)
            attribution_list.append(attributions)

            filenames_duplicate = [item for item in filenames]
            filenames_list = filenames_list + filenames_duplicate

        if len(attribution_list)!=0:
            attributions = torch.cat(attribution_list, dim=0)
            attributions = attributions.detach().cpu().numpy()
            grad_cam_dict[str(target)] = {file: attribution for file, attribution in zip(filenames_list, attributions)}
    
    return grad_cam_dict


def compute_all_grad_cams(loaders_dict, model, with_labels=True):
    """Compute the gradient maps of model for all the loaders in loaders_dict.
    
    Arguments:
        - loaders_dict: dictionary with the subset name as key ('train', 'val', etc) 
        and the corresponding loader as value.
        - model: the neural network to evaluate gradcam on. Should have at least 1 conv layer.
        
    Returns a dictionary structured the following way subset > classification_class > subject"""

    attributions_dict = {}
    for subset_name, loader in loaders_dict.items():
        grad_cam_dict = compute_grad_cam(loader, model, with_labels=with_labels)
        attributions_dict[subset_name] = grad_cam_dict
    return attributions_dict


