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


def rotate_135(ts):
    rotate = torchvision.transforms.RandomRotation(degrees=(135, 135))
    crop = torchvision.transforms.CenterCrop(size=280)
    resize = torchvision.transforms.Resize(size=400, interpolation=3)
    transform = torchvision.transforms.Compose([rotate, crop, resize])
    return transform(ts)


def rotate_180(ts):
    rotate = torchvision.transforms.RandomRotation(degrees=(180, 180))
    return rotate(ts)


def rotate_210(ts):
    rotate = torchvision.transforms.RandomRotation(degrees=(210, 210))
    crop = torchvision.transforms.CenterCrop(size=300)
    resize = torchvision.transforms.Resize(size=400, interpolation=3)
    transform = torchvision.transforms.Compose([rotate, crop, resize])
    return transform(ts)


def rotate_225(ts):
    rotate = torchvision.transforms.RandomRotation(degrees=(225, 225))
    crop = torchvision.transforms.CenterCrop(size=280)
    resize = torchvision.transforms.Resize(size=400, interpolation=3)
    transform = torchvision.transforms.Compose([rotate, crop, resize])
    return transform(ts)


def rotate_240(ts):
    rotate = torchvision.transforms.RandomRotation(degrees=(240, 240))
    crop = torchvision.transforms.CenterCrop(size=300)
    resize = torchvision.transforms.Resize(size=400, interpolation=3)
    transform = torchvision.transforms.Compose([rotate, crop, resize])
    return transform(ts)


def rotate_270(ts):
    rotate = torchvision.transforms.RandomRotation(degrees=(270, 270))
    return rotate(ts)


def random_rotate(ts, p):
    """
    Random apply rotations with 5 pre-defined degrees
    """
    rand = torch.rand(1).item()
    p = p / 8
    if rand <= p:
        ts = rotate_45(ts)
    elif p < rand < 2 * p:
        ts = rotate_90(ts)
    elif 2 * p < rand <= 3 * p:
        ts = rotate_135(ts)
    elif 3 * p < rand <= 4 * p:
        ts = rotate_180(ts)
    elif 4 * p < rand <= 5 * p:
        ts = rotate_210(ts)
    elif 5 * p < rand <= 6 * p:
        ts = rotate_225(ts)
    elif 6 * p < rand <= 7 * p:
        ts = rotate_240(ts)
    elif 7 * p < rand <= 8 * p:
        ts = rotate_270(ts)
    else:
        ts = ts
    return ts
