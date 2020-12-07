import torch
import torchvision


def rotate_45(ts):
    rotate = torchvision.transforms.RandomRotation(degrees=(45, 45))
    crop = torchvision.transforms.CenterCrop(size=280)
    resize = torchvision.transforms.Resize(size=400, interpolation=3)
    transform = torchvision.transforms.Compose([rotate, crop, resize])
    return transform(ts)


def rotate_90(ts):
    rotate = torchvision.transforms.RandomRotation(degrees=(90, 90))
    return rotate(ts)


def rotate_60(ts):
    rotate = torchvision.transforms.RandomRotation(degrees=(45, 45))
    crop = torchvision.transforms.CenterCrop(size=300)
    resize = torchvision.transforms.Resize(size=400, interpolation=3)
    transform = torchvision.transforms.Compose([rotate, crop, resize])
    return transform(ts)


def rotate_30(ts):
    rotate = torchvision.transforms.RandomRotation(degrees=(30, 30))
    crop = torchvision.transforms.CenterCrop(size=300)
    resize = torchvision.transforms.Resize(size=400, interpolation=3)
    transform = torchvision.transforms.Compose([rotate, crop, resize])
    return transform(ts)


def rotate_75(ts):
    rotate = torchvision.transforms.RandomRotation(degrees=(75, 75))
    crop = torchvision.transforms.CenterCrop(size=300)
    resize = torchvision.transforms.Resize(size=400, interpolation=3)
    transform = torchvision.transforms.Compose([rotate, crop, resize])
    return transform(ts)


def random_rotate(ts, p):
    """
    Random apply rotations with 5 pre-defined degrees
    """
    rand = torch.rand(1).item()
    p = p/5
    if rand <= p:
        ts = rotate_30(ts)
    elif p < rand <= 2 * p:
        ts = rotate_45(ts)
    elif 2 * p < rand <= 3 * p:
        ts = rotate_60(ts)
    elif 3 * p < rand <= 4 * p:
        ts = rotate_75(ts)
    elif 4 * p < rand <= 5 * p:
        ts = rotate_90(ts)
    else:
        ts = ts
    return ts
