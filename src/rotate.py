import torch
import torchvision


def rotate_n45(ts, n=1, theta=15):
    """
    Apply random rotation on tensor within the range of (n*45-theta, n*45+theta)
    """
    assert n in [1, 3, 5, 7], "n should be 1,3,5 or 7"
    rotate = torchvision.transforms.RandomRotation(degrees=(n * 45 - theta, n * 45 + theta))
    crop = torchvision.transforms.CenterCrop(size=280)
    resize = torchvision.transforms.Resize(size=400, interpolation=3)
    transform = torchvision.transforms.Compose([rotate, crop, resize])
    return transform(ts)


def rotate_n90(ts, n=1):
    """
    Rotate the tensor by 90,180 or 270
    """
    assert n in [1, 2, 3], "n should be 1,2 or 3"
    rotate = torchvision.transforms.RandomRotation(degrees=(n * 90, n * 90))
    return rotate(ts)


def random_rotate(ts, prob=0.875):
    """
    Random apply rotations with pre-defined range of degrees and probability
    """
    rand = torch.rand(1).item()
    p = prob / 7
    if rand <= p:  # 30~60
        ts = rotate_n45(ts, n=1)
    elif p < rand < 2 * p:  # 90
        ts = rotate_n90(ts, n=1)
    elif 2 * p < rand <= 3 * p:  # 120~150
        ts = rotate_n45(ts, n=3)
    elif 3 * p < rand <= 4 * p:  # 180
        ts = rotate_n90(ts, n=2)
    elif 4 * p < rand <= 5 * p:  # 210~240
        ts = rotate_n45(ts, n=5)
    elif 5 * p < rand <= 6 * p:  # 270
        ts = rotate_n90(ts, n=3)
    elif 6 * p < rand <= 7 * p:  # 300~330
        ts = rotate_n45(ts, n=7)
    else:
        ts = ts
    return ts
