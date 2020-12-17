import os
import torch
import skimage.io as io
from UNet import *
from loader import *
from training import *
from mask_to_submission import *


def predict_mask(image, model):
    if len(image.size()) < 4:
        image = torch.unsqueeze(image, dim=0)
    mask_pred = model(image)
    prediction = torch.sigmoid(mask_pred)
    prediction = (prediction > 0.5).float()
    prediction = torch.squeeze(prediction).numpy()
    return prediction


root_dir = ""  # here is your root dir
# load our pretrained model
checkpoint_path = os.path.join(root_dir, '/pretrained/UNet_64_pretrained.pt')
net = UNet(n_channels=3, n_classes=1, n_filters=64)
epoch = load_from_checkpoint(checkpoint_path, net, optimizer=None, scheduler=None, verbose=True)
# load test images
test_dir = os.path.join(root_dir, '/data/test_set_images')
test_set = TestDataset(test_dir, num_imgs=50, to_numpy=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
# make prediction and save binary images to /pred folder
SAVE_PATH = os.path.join(root_dir, '/pred')
for i, batch in enumerate(test_loader):
    img = batch["image"]
    idx = batch["ID"]
    print(idx)
    pred = predict_mask(img, net)
    io.imsave(os.path.join(SAVE_PATH, '%.3d.png' % (i + 1)), pred)
# generate submission csv to /submission folder
predict_path = os.path.join(root_dir, '/pred')
submission_path = os.path.join(root_dir, '/submission')
make_submission(predict_path, test_size=50, submission_filename=os.path.join(submission_path, "submission.csv"))
