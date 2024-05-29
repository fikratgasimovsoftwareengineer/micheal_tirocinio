"""
06. PyTorch Transfer Learning
What is transfer learning?
Transfer learning involves taking the parameters of what one model has
learned on another dataset and applying to our own problem
* Pre-trained model = foundation models

After getting the right versions let's import the code we've written
in previous sections so that we don't have to write it all again
"""
import torch # version -> 2.2.2+cu121 | we wanted 1.12+
import torchvision # version -> '0.17.2+cu121' | we wanted 0.13+
import torchinfo
from torch import nn
from torchvision import transforms
from torchinfo import summary
import matplotlib.pyplot as plt
# Import python files with useful functions
import data_setup
import engine
import os
import requests
import zipfile
from pathlib import Path

# Setup device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
1. Get data
We need our pizza, steak, sushi data to build a transfer learning model on
"""
# Setup path to a data folder
data_path = Path('/home/michel/Desktop/TopNetwork/08: PyTorch/PYTORCH_NOTEBOOKS/Data')
image_path = data_path / 'pizza_steak_sushi'

# If the image folder doesn't exist, download it and prepare it...
if image_path.is_dir():
    print(f'The given directory already exists, skipping download')
else:
    print(f"The given directory doesn't exist, downloading now...")
    image_path.mkdir(parents=True, exist_ok=True)
    # Download pizza, steak and sushi data
    with open(data_path / 'pizza_steak_sushi.zip', 'wb') as f:
        request = requests.get('https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip')
        print('Downloading pizza, steak, sushi data...')
        f.write(request.content)
    # Unzip pizza, steak, sushi data
    with zipfile.ZipFile(data_path / 'pizza_steak_sushi.zip', 'r') as zip_ref:
        print('Unzipping pizza, steak and sushi data...')
        zip_ref.extractall(image_path)
    # Remove .zip file
    os.remove(data_path / 'pizza_steak_sushi.zip')

# Setup directory path
train_dir = image_path / 'train'
test_dir = image_path / 'test'

"""
2. Create Datasets and Dataloaders
Now we got some data, want to turn it into Pytorch DataLoaders.
To do so, we can use data_setup.py and the create_dataloaders() function
There's one thing we have to think about when loading: how to transform it?
And with torchvision there's two ways to do this:
1. Manually create transforms - we define what transforms you want
you data to go through
2. Automatically created transforms - the transforms your data are
defined by the model you'd like to use

Important point: when using a pretrained model, it's important that the
data (including your custom data) that you pass through it is transformed
in the same way that the data the model was trained on



2.1 Creating a transform for torchvision.models (manual creation)
torchvision.model contains pretrained models (ready for transfer learning)
We need to normalize the images with a standard, 
in this case we use the standard of ImageNet
"""
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
manual_transoforms = transforms.Compose([
    transforms.Resize((224, 224)), # resize the image
    transforms.ToTensor(), # get images into range 0 to 1
    normalize]) # make sure images have the same distribution
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=str(train_dir),
                                                                               test_dir=str(test_dir),
                                                                               transform=manual_transoforms,
                                                                               batch_size=32)
# print(train_dataloader, test_dataloader, class_names)

"""
2.2 Creating a transform from torchvision.models (auto creation)
There is the possibility for automatic data transform creation based
on the pretrained model weights you're using
"""
# Get a set of pretrained model weights
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT # default = best available weights

# Get the transforms used to create our pretrained weights
auto_transforms = weights.transforms()
print(auto_transforms)

# Create Dataloaders usign automatic transform
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=str(train_dir),
                                                                               test_dir=str(test_dir),
                                                                               transform=auto_transforms,
                                                                               batch_size=32)
# print(train_dataloader, test_dataloader, class_names)

"""
3. Getting a pretrained model
There are various places to get a pretrained model, such as:
-  PyTorch domains libraries (torchvision, torchtext, torchaudio, torchrec)
-  Torch image models (timm library)
-  HuggingFace Hub
-  Paperswithcode SOTA



3.1 Which pretrained model shoul you use?
Experiment, experiment... experiment!
The whole idea of transfer learning: take an already well-performing
model from a problem space similar to your own and then customize to
your own problem

Thing to consider are:
1. Speed - how fast does it need to runs?
2. Size - how big is the model?
3. Performance - how well does it go on your chosen problem?

Where does the model live? On the device? (self driving car)
Or on a server? Which model should we chose?
For our case (deploying FoodVision Mini on a mobile device),
EffNetB0 is one of our best options in terms of performance vs size
However, in light of the Bitter Lesson, if we had infinite compute,
we'd likely pick the biggest model + most parameters + most general
we could - http://www.incompleteideas.net/IncIdeas/BitterLesson.html



3.2 Setting up a pretrained model
Want to create an instance of pretrained EffNetB0 -
https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html
When setting up a pretrained model we only modify the input and output layer,
this is called feature extraction, usefull for small amount of data
If you have large amount of data you will do also fine-tuning, you might
change the input layer but will change some of the last layers
"""
# How to setup a pretrained model
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
model = torchvision.models.efficientnet_b0(weights=weights)

"""
3.3 Getting a summary of our model with torchinfo.summary() for feature extraction
"""
summary(model=model,
        input_size=(1, 3, 224, 224), # BS, C, H, W
        col_names=['input_size', 'output_size', 'num_params', 'trainable'],
        col_width=20,
        row_settings=['var_names'])

"""
3.4 Freezing the base model and changing the output layer to suit our needs
With a feature extractor model, typically you will 'freeze' the base layers of a
pretrained/foundation model and update the output layers to suit your own problem
"""
# Freeze all the base layers in EffNetB0
for param in model.features.parameters():
    # print(param)
    param.requires_grad = False

# Update the classifier head of our model to suit our problem
print(model.classifier)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True), # the same of the original
    # In our case we need to change only the out_features to the 3 foods
    nn.Linear(in_features=1280, # feature vector coming in
              out_features=len(class_names))) # how many classes do we have (3)
print(model.classifier)
