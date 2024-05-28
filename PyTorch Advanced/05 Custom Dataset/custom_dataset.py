"""
We've used some datasets with pytorch before
But how do you get youe own data into pytorch?
One of the ways to do so is via: custom datasets

Domain libraries
Depending on what you're working one you'll want to look into each of the pytorch
domain libraries for existing data loading functions and customizable data loading functions

Resources:
* Book version: https://www.learnpytorch.io/04_pytorch_custom_datasets/
* Ground truth notebook: https://github.com/mrdbourke/pytorch-deep-learning/blob/main/04_pytorch_custom_datasets.ipynb



0. Importing Pytorch and setting up device-agnostic code
"""
import torch
from torch import nn
# 1.
import requests
import zipfile
from pathlib import Path
# 2.
import os
# 2.1
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# 3.
from torchvision import transforms
# 4.
from torchvision import datasets
# 4.1
from torch.utils.data import DataLoader
# 5.
from typing import Tuple, Dict, List, Any
# 5.2
from torch.utils.data import Dataset
# 7.4
import torchinfo
from torchinfo import summary
# 7.6
from tqdm.auto import tqdm
# 7.7
from timeit import default_timer as timer

# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else 'cpu'

"""
1. Get data
Our dataset is a subset of the Food 101 dataset
Food101 starts 101 different classes of food
Our dataset starts with 3 classes of food and only 10% of the images 
(75 training - 25 testing)
Why?
When starting our ML projects it's important to try things on a small scale 
and then increase the scale when necessary.
The whole point is to speed up how fast you can experiment
"""
# Setup path to a data folder
data_path = Path('/home/michel/Desktop/TopNetwork/08: PyTorch/PYTORCH_NOTEBOOKS/Data')
image_path = data_path / 'pizza_steak_sushi'

# If the image folder doesn't exist, download it and prepare it...
if image_path.is_dir():
    print(f'{image_path} directory already exists... skipping download')
else:
    print(f"{image_path} doesn't exist, downloading now...")
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
print(image_path)

"""
2. Becoming one with the data (data preparation and data exploration)
"""
def walk_through_dir(dir_path):
    """Walks through dir_path returning its contents"""
    for dirpath, dirnames, filesnames in os.walk(dir_path):
        print(f'There are {len(dirnames)} directories and {len(filesnames)} images in "{dirpath}"')

walk_through_dir(image_path)

# Setup train and testing paths
train_dir = image_path / 'train'
test_dir = image_path / 'test'
print(f'{train_dir}\n{test_dir}')

"""
2.1 Visualizing an image
Let's write some code to:
1. Get all of the image paths
2. Pick a random image path using Python's random.choice()
3. Get the image class name using pathlib.Path.parent.stem
4. Since we're working with images, let's open the image with Python's PIL
5. We'll then show the image and print metadata
"""
# Set seed
# random.seed(42)

# 1. Get all image paths
image_path_list = list(image_path.glob('*/*/*.jpg'))

# 2. Get a random image path
random_image_path = random.choice(image_path_list)
print(random_image_path)

# 3. Get image classe from path (the image class
# is the name of the directory where the image is stored)
image_class = random_image_path.parent.stem
print(image_class)

# 4. Open image
img = Image.open(random_image_path)

# 5. Print metadata
print(f'Random image path: {random_image_path}')
print(f'Image class: {image_class}')
print(f'Image height: {img.height}')
print(f'Image width: {img.width}')
# Use .show for see the image
# img.show()

# Turn the image in an array
img_as_array = np.asarray(img)

# Plot the image with matplotlib
# plt.figure(figsize=(10, 7))
# plt.imshow(img_as_array) # NOT NECESSARY as an array
# plt.title(f'Image class: {image_class} | '
#           f'Image shape: {img_as_array.shape} -> [height, width, color channels]')
# plt.axis(False)
# plt.show()

"""
3. Transforming data
Before we can use our image data with Pytorch:
1. Turn your target data into tensors (in our case, numerical representation of our
image)
2. Turn it into a torch.utils.data.Dataset and subsequently a torch.utils.data.Dataloader,
we'll call these Dataset and DataLoader



3.1 Transforming data with torchvision.transforms
Tranforms help you get your images ready to be used with a model/perform data augmentation
- check the documenation
"""
# Write a transform for image, you can use compose or sequential
data_transform = transforms.Compose([
    # Resize our images to 64x64, only because our preious model were set for this size
    transforms.Resize(size=(64, 64)),
    # Flip the images randomly on the horizontal (data augmentation)
    transforms.RandomHorizontalFlip(p=0.5), # we set a chance 50%
    # Turn the image in a torch.Tensor
    transforms.ToTensor()
])
tensor_img = data_transform(img) # as an array gives error
print(tensor_img, tensor_img.shape, tensor_img.dtype)

def plot_transformed_images(image_paths, transform, n=3, seed=None):
    """Selects random images from a path of images and load/tranforms
    them then plots the original vs the transformed version"""
    if seed:
        random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(nrows=1, ncols=2)
            ax[0].imshow(f)
            ax[0].set_title(f'Original\nSize: {f.size}')
            ax[0].axis(False)

            # Transform and plot target image
            tranformed_image = transform(f).permute(1, 2, 0)
            # Transform puts the colors first but plt wants them last
            ax[1].imshow(tranformed_image)
            ax[1].set_title(f'Transformed\nShape: {tranformed_image.shape}')
            ax[1].axis('off')
            fig.suptitle(f'Class: {image_path.parent.stem}', fontsize=16)
            plt.show()

# plot_transformed_images(image_paths=image_path_list, transform=data_transform, n=3, seed=42)

"""
4. Option 1: Loading image data usign ImageFolder
We can load image classification data using torchvision.datasets.ImageFolder
- check documentation
"""
# Use ImageFolder to create dataset(s)
train_data = datasets.ImageFolder(root=str(train_dir),
                                  transform=data_transform, # a transform for the data
                                  target_transform=None) # a transform for the label/target
test_data = datasets.ImageFolder(root=str(test_dir),
                                  transform=data_transform)
print(train_data, test_data)

# Get class names as list
class_names = train_data.classes
print(class_names)

# Get class names as dict
class_dict = train_data.class_to_idx
print(class_dict)

# Check the lenght of our dataset
print(len(train_data), len(test_data))

# Index on the train_data Dataset to get a single image and label
img, label = train_data[0][0], train_data[0][1]
print(f'Image tensor:\n {img}')
print(f'Image shape: {img.shape}')
print(f'Image datatype: {img.dtype}')
print(f'Image label: {label}')
print(f'Label datatype: {type(label)}')

# Rearrange the order of dimension
img_permute = img.permute(1, 2, 0)

# Print our differente shapes
print(f'Original shape:  {img.shape} -> [C, H, W]')
print(f'Image permute: {img_permute.shape} -> [H, W, C]')

# Plot image
# plt.figure(figsize=(10, 7))
# plt.imshow(img_permute)
# plt.axis(False)
# plt.title(class_names[label], fontsize=14)
# plt.show()

"""
4.1 Turn loaded images into Dataloaders
A Dataloader is going to help us turn our Dataset's into iterables and we can
customize the batch_size so our model can see batch_size images at a time
"""
# Turn train and test datasets into dataloaders
BATCH_SIZE = 1
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              num_workers=1, # you can select how many cores use
                              shuffle=True)
test_dataloader = DataLoader(dataset=test_data,
                              batch_size=BATCH_SIZE,
                              num_workers=1,
                              shuffle=False)
print(train_dataloader, test_dataloader, len(train_dataloader), len(test_dataloader))
img, label = next(iter(train_dataloader))

# Batch size (BS) will now be 1, you can change the batch if you like
print(f'Image shape: {img.shape} -> [BS, C, H, W]')
print(f'Label shape: {label.shape}')

"""
5. Option 2: Loading Image Data with a custom Dataset
1. Want to be able to load images from file
2. Want to be able to get class names from the Dataset
3. Want to be able to get classes as dictionary from the Dataset

Pros:
* Can create a Dataset out of almost anything...
* Not limited to Pytorch pre-built Dataset functions

Cons:
* ...but it doesn't mean it will work
* Using a custom Dataset often results in us writing more code, which
could be prone to errors or performance issues

All custom datasets in pytorch, often subclass torch.utils.data.Dataset
- check documentation
"""
# Instance of torchvision.datasets.ImageFolder()
print(train_data.classes, train_data.class_to_idx)

"""
5.1 Creating a helper function to get class names
We want a function to:
1. Get the class names using os.scandir() to traverse a target directory
(ideally the directory is in standard image classification format)
2. Raise an error if the class names aren't found (if this happens,
there might be something wrong with the directory structure)
3. Turn the class names into a dict and a list and return them
"""
# Setup path for target directory
target_directory = train_dir
print(f'Target dir: {target_directory}')

# Get the class names from the target directory
class_names_found = sorted([entry.name for entry in list(os.scandir(target_directory))])
def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folder names in a target directory"""
    # 1. Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    # 2. Raise and error if class names could not be found
    if not classes:
        raise FileNotFoundError(f"Could't find any classes in {directory}, check file structure")

    # 3. Create a dictionary of index labels (computers prefer numbers rather than strings as labels)
    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
    return classes, class_to_idx

print(find_classes(str(target_directory)))

"""
5.2 Create a custom Dataset to replicate ImageFolder
To create our own custom dataset:
1. Subclass torch.utils.data.Dataset
2. Init our subclass with a target directory (the directory we'd like to get data
from) as well as a transform if we'd like to transform our data
3. Create several attibutes
    * paths - paths of our images
    * transform - the transform we'd like to use
    * classes - a list of the target classes
    * class_to_idx - a dict of the target classes mapped to integer labels
4. Create a function to load_images(), this function will open an image
5. Overwrite the __len()__ method to return the length of our dataset
6. Overwrite the __getitem()__ method to return a given sample when passed an index
"""
# 1. Subclass torch.util.data.Dataset
class ImageFolderCustom(Dataset):
    # 2. Initialize our custom dataset
    def __init__(self, targ_dir: str, transform=None):
        # 3. Create class attributes
        # Get all the image paths
        self.paths = list(Path(targ_dir).glob('*/*.jpg'))
        # Setup transforms
        self.trasform = transform
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(targ_dir)

    # 4. Create a function to load images
    def load_image(self, index: int) -> Image.Image:
        """Opens and image via a path and returns it"""
        image_path = self.paths[index]
        return Image.open(image_path)

    # 5. Overwrite __len__()
    # All subclasses should overwrite __len__
    def __len__(self) -> int:
        """Returns the total number of samples"""
        return len(self.paths)

    # 6. Overwrite __get_item() method to return a particular sample
    # All subclasses should overwrite __get_item__
    def __getitem__(self, index: int) -> tuple[Any, int] | tuple[Image, int]:
        """Returns one sample of data, data and label (X, y)"""
        img = self.load_image(index)
        class_name = self.paths[index].parent.name # expects path in format: data_folder/class_name/image.jpg
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        if self.trasform:
            return self.trasform(img), class_idx # return data, label (X, y)
        else:
            return img, class_idx # return untransformed image and label

# Create a transform
train_transforms = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])
# Usually you leave the test data intact if possible
test_transforms = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()])

# Test out ImageFolderCustom
train_data_custom = ImageFolderCustom(targ_dir=str(train_dir),
                                      transform=train_transforms)
test_data_custom = ImageFolderCustom(targ_dir=str(test_dir),
                                      transform=test_transforms)
print(len(train_data), len(train_data_custom))
print(len(test_data), len(test_data_custom))
print(train_data_custom.classes)
print(train_data_custom.class_to_idx)

# Check for equality between original ImageFolder Dqataset and ImageFolderCustom
print(train_data_custom.classes==train_data.classes)
print(test_data_custom.classes==test_data.classes)

"""
5.3 Create a function to display random images
1. Take in a Dataset and a number of other parameters such as class names and how
many images to visualize
2. To prevent the display getting out of hand, let's cap the number of images to
see at 10
3. Set the random see for reprofucibility
4. Get a list of random sample indexes from the target dataset
5. Setup a matplotlib plot
6. Loop through the random sample images indexes and plot them with matplotlib
7. Make sure the dimensions of our images line up with matplotlib (HWC)
"""
# 1. Create a function to take in a dataset
def display_random_images(dataset: torch.utils.data.Dataset,
                          classes: List[str] = None,
                          n: int = 10,
                          display_shape: bool = True,
                          seed: int = None):
    # 2. Adjust display if n is too high
    if n > 10:
        display_shape = False
        n = 10
        print("For display purposes n shouldn't be larger than 10, setting to 10 and removing shape display")

    # 3. Set the seed
    if seed:
        random.seed(seed)

    # 4. Get random indexes
    random_samples_idx = random.sample(range(len(dataset)), k=n)

    # 5. Setup plot
    plt.figure(figsize=(16, 8))

    # 6. Loop through random indexes and plot them with matplotlib
    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]

        # 7. Adjust tensor dimensions for plotting
        targ_image_adjust = targ_image.permute(1, 2, 0)

        # Plot adjusted samples
        plt.subplot(1, n, i+1)
        plt.imshow(targ_image_adjust)
        plt.axis(False)
        title = 'NaN'
        if classes:
            title = f'Class: {classes[targ_label]}'
            if display_shape:
                title = title + f"\nshape: {targ_image_adjust.shape}"
        plt.title(title)
    plt.show()

# Display random images from the ImageFolder created Dataset
# display_random_images(train_data, n=5, classes=class_names, seed=None)

# Display random images from the ImageFolderCustom Dataset
# display_random_images(train_data_custom, n=20, classes=class_names, seed=None)

"""
5.4 Turn custom loaded images into DataLoader
"""
BATCH_SIZE = 64
NUM_WORKERS = os.cpu_count()
train_dataloader_custom = DataLoader(dataset=train_data_custom,
                                     batch_size=BATCH_SIZE,
                                     num_workers=NUM_WORKERS,
                                     shuffle=True)
test_dataloader_custom = DataLoader(dataset=test_data_custom,
                                     batch_size=BATCH_SIZE,
                                     num_workers=NUM_WORKERS,
                                     shuffle=False)
print(train_dataloader_custom, test_dataloader_custom)

# Get image and label from custom dataloader
img_custom, label_custom = next(iter(train_dataloader_custom))

# Print out the shapes
print(img_custom.shape, label_custom.shape)

"""
6. Other forms of transforms (data augmentation)
Data augmentation is the preocsess of artificially adding diversity to our training data
In the case of image data, this may mean applying various image transformations to the training images
This practice hopefully result
Let's take a look at one particular type of data augmentation used to
train Pytorch vision models to state of the art levels...
"""
# Let's look at trivial augment - check documentation
train_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()])
test_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor()])

# Get all image paths
image_path_list = list(image_path.glob('*/*/*.jpg'))

# Plot random transformed images
# plot_transformed_images(image_paths=image_path_list, transform=train_transform, n=3, seed=None)

"""
7. Model 0: TinyVGG without data augmentation
Let's replicate the TinyVGG architecture from the CNN Expliner website:
https://poloclub.github.io/cnn-explainer/



7.1 Creating transforms and loading data for model 0
"""
# Create simple transform
simple_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()])

# 1. Load and transform data
train_data_simple = datasets.ImageFolder(root=str(train_dir),
                                         transform=simple_transform)
test_data_simple = datasets.ImageFolder(root=str(test_dir),
                                        transform=simple_transform)

# 2. Turn the datasets into DatLoaders
# Setup batch size and number of works
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()

# Create DataLoader
train_dataloader_simple = DataLoader(dataset=train_data_simple,
                                     batch_size=BATCH_SIZE,
                                     shuffle=True,
                                     num_workers=NUM_WORKERS)
test_dataloader_simple = DataLoader(dataset=test_data_simple,
                                     batch_size=BATCH_SIZE,
                                     shuffle=False,
                                     num_workers=NUM_WORKERS)

"""
7.2 Create TinyVGG model class
"""
class TinyVGG(nn.Module):
    """
    Model architecture that replicates the TinyVGG
    model from CNN explainer website
    """
    def __init__(self, input_shape: int,
                 hidden_units: int,
                 output_shape: int) -> None:
        super().__init__()
        #  Create a convolutional block
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1, # stride 1 removes 2 pixels from the image
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)) # default is same like kernel_size
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, # max pool of 2 halves the pixel in the image
                         stride=2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # check output shape of conv_block_2, or use torchinfo
            nn.Linear(in_features=hidden_units*13*13,
                      out_features=output_shape))

    def forward(self, x):
        # Benefits from operator fusion, aka speeds up gpu performance
        # https://horace.io/brrr_intro.html
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))

torch.manual_seed(42)
model_0 = TinyVGG(3, 10, len(class_names)).to(device)
print(model_0)

"""
7.3 Try a forward pass on a single image (to test the model)
"""
image_batch, label_batch = next(iter(train_dataloader_simple))
print(image_batch.shape, label_batch.shape)

# Try forward pass
model_0(image_batch)

"""
7.4 Use torchinfo to get an idea of the shapes going through our model
"""
summary(model_0, input_size=[1, 3, 64, 64])

"""
7.5 Create train and test loops functions
* train_step - takes a model and dataloader and trains on it
* test_step - takes a model and dataloader and evaluates on it
"""
def train_step_no_acc_fn(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               # accuracy_fn,
               device: torch.device = device):
    """Performs a training with model trying to learn on data_loader.
    This version doesn't take an accuracy function as parameter.
    It also returns train loss and accuracy"""
    train_loss, train_acc = 0, 0
    # Put the model into training mode
    model.train()
    # Add a loop to through the training batches
    for batch, (X, y) in enumerate(data_loader):
        # Put data on target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass (outputs the raw logits from the model)
        y_pred = model(X)

        # 2. Calculate loss and accuracy (per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() # accumulate train loss
        # train_acc += accuracy_fn(y_true=y,
        #                         y_pred=y_pred.argmax(dim=1)) # from logits to prediction labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5.Optimizer step
        optimizer.step()

        # Calculate accuracy metric (without pre-built function)
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class==y).sum().item()/len(y_pred)

    # Divide total train loss and acc by lenght of train dataloader
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    return train_loss, train_acc

def test_step_no_acc_fn(model: torch.nn.Module,
             data_loader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module,
             # accuracy_fn,
             device: torch.device = device):
    """Performs a testing loop step on model going over data_loader
    This version doesn't take an accuracy function as parameter.
    It also returns test loss and accuracy"""
    test_loss, test_acc = 0, 0
    # Put the model in eval mode
    model.eval()
    # Turn on inference mode context manager
    with torch.inference_mode():
        for X, y in data_loader:
            # Send the data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred = model(X)

            # 2. Calculate the loss/acc
            test_loss += loss_fn(test_pred, y).item()
            # test_acc += accuracy_fn(y_true=y,
            #                         y_pred=test_pred.argmax(dim=1)) # from logits to prediction labels

            # Calculate the accuracy
            test_pred_labels = test_pred.argmax(dim=1)
            test_acc += (test_pred_labels==y).sum().item()/len(test_pred_labels)

        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        return test_loss, test_acc

"""
7.6 Creating a train function to combine train_step and test_step
"""
# 1. Create a train function that takes in various model parameters + optimizer + dataloader
def train(model: torch.nn.Module,
          train_dataloader,
          test_dataloader,
          optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5,
          device=device):
    """Performs training and evaluation of a model if given
    a train and test function"""

    # 2. Create empty results dictionary
    results = {'train_loss': [],
               'train_acc':  [],
               'test_loss': [],
               'test_acc': []}

    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step_no_acc_fn(model=model,
                                                     data_loader=train_dataloader,
                                                     loss_fn=loss_fn,
                                                     optimizer=optimizer,
                                                     device=device)
        test_loss, test_acc = test_step_no_acc_fn(model=model,
                                                     data_loader=test_dataloader,
                                                     loss_fn=loss_fn,
                                                     device=device)

        # 4. Print out what's happening
        print(f'Epoch: {epoch}\n'
              f'Train loss: {train_loss:.4f} Train acc: {train_acc*100:.2f}%\n'
              f'Test loss: {test_loss:.4f} Train acc: {test_acc*100:.2f}%\n')

        # 5. Update results dictionary
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)

    # 6. Return the filled results at the end of the epochs
    return results

"""
7.7 Train and evaluate
"""
# Set random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Set number of epochs
NUM_EPOCHS = 10

# Recreate and instance of TinyVGG
model_0 = TinyVGG(input_shape=3, hidden_units=10, output_shape=len(train_data.classes)).to(device)

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)

# Start the timer
start_time = timer()

# Train model_0
model_0_results = train(model=model_0,
                        train_dataloader=train_dataloader_simple,
                        test_dataloader=test_dataloader_simple,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=NUM_EPOCHS)

# End timer and print out how long it took
end_time = timer()
total_time = end_time - start_time
print(f'Total training time: {total_time:.2f} seconds')
