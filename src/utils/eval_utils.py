import torch
from timm.data.transforms import str_to_interp_mode
from torchvision import transforms


def binary_accuracy(output, target):
    with torch.no_grad():
        output = output > 0
        return torch.sum(output==target).item()/len(target)*100

def get_basic_transform(data_config):
    input_size = data_config['input_size']
    if isinstance(input_size, (tuple, list)):
        input_size = input_size[-2:]
    else:
        input_size = (input_size, input_size)

    mean = (data_config['mean'][0],)
    std = (data_config['std'][0],)
    interpolation = data_config['interpolation']

    tf = [
        transforms.ToTensor(),
        transforms.Resize(
            input_size,
            interpolation=str_to_interp_mode(interpolation),
        ),
        transforms.Normalize(mean=mean, std=std),
    ]

    return transforms.Compose(tf)