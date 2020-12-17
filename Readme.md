# Road Segmentation with U-Net
**`AIcrowd username`:** dongxu **`AIcrowd submission ID`:** 108563 (F1=0.875) 

**`Team Members`:** Dongxu Guo, Jiaan Zhu, Lei Wang

This GitHub hosts the code for Project 2 of Machine Learnin(CS-433). In this project we implement and train neural network for road segmentation,
i.e. assigning labels `road=1`, `background=0` to each pixel in satilite images. The model is based on U-Net from [Ronneberger et al. (2015)](https://arxiv.org/pdf/1505.04597.pdf).

![](u-net.png)


The train and test [data](https://github.com/aschneuw/ml-chiefs/tree/master/data) contains 100 and 50 images of size 400x400 and 604x604 respectively. Please kindly download the images from the [official site](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/dataset_files) on AiCrowd. 

<a name="dependencies"></a>
## Environment and Dependencies
The setup requires a default Unix Environment. The interface is written and tested using python 3.8. The interface also requires
the following external libraries:<br/>
* PyTorch(v1.7)
* scikit-learn
* scikit-image

<a name="model training"></a>
## Training and Prediction
To generate our final AICrowd submission please download our pretrained model [link](). Unzip and place the state dictionary (weights) in  ```/pretrained``` folder. Then execute ```run.py```, you will get the ```submission.csv``` under the  ```submission\``` folder and prediction in ```pred\``` folder.


You can always retrain our model by running:
```
python3 train.py
```
with the appropriate optional arguments, if needed.

The optional arguments can be used to:
  - specify the train/validation split ratio
  - change the hyper-parameters
  - modify the model architecture
  - specify the model save path and saving conditions

The defualt setting is what gives us the best performed model. Howerver, since we use random data augmentation and do not set the seed, the exact reproducibility of our result is not ensured.

If [TensorBoard](https://www.tensorflow.org/tensorboard/) is installed, metrics (training losses and validation score) can be tracked and visualized. To launch Tensorboard, run: 

`tensorboard --logdir=path/to/logdir`

## Modules
All modules are provided in src.

### ```run.py```

Script to generate the same submission file as we submitted in AICrowd with pretrained models.

### ```train.py```

Main script to retrain models with your customized settings.

### ```UNet.py```

Implementation of the modified U-Net model(with optional dropout and batch-normalization).

### ```loader.py```
Dataloader for loading training and testing data.

### ```training.py```

Fucnctions for training, validaing, saving and loading models.

### ```dice.py```

Define Dice coefficient, the metric we use for validation.

### ```rotate.py```

Functions for random rotations in data augmentation


### ```mask_to_submission.py```

Helper functions for generating AICrowd submission file.

## License

The project is licensed under the MIT License.
