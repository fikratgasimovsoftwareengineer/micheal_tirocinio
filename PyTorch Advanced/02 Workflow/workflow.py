"""
02. Pytorch Workflow

Let's explore an example PyTorch end-to-end workflow.
Resources:
    Ground truth notebook: https://github.com/mrdbourke/pytorch-deep-learning/blob/main/01_pytorch_workflow.ipynb
    Book version of notebook: https://www.learnpytorch.io/01_pytorch_workflow/
    Ask a question: https://github.com/mrdbourke/pytorch-deep-learning/discussions
"""
what_were_covering = {1: 'data (prepare and load)',
                      2: 'build model',
                      3: 'fitting the model to data (training)',
                      4: 'making predictions and evaluating a model (inference)',
                      5: 'saving and loading a model',
                      6: 'putting it all together'}
import torch
from torch import nn # contains all pytorch building blocks for neural networks
import matplotlib.pyplot as plt

# Check pytorch version
print(torch.__version__)

"""
1. Data (preparing and loading)
Data can be almost anything... in machine learning
* Excel spredsheet
* Images of any kind
* Videos (Youtube has lots of data...)
* Audio like songs or podcasts
* DNA
* Text
Machine learning is a game of two parts:
1. Get data into a numerical representation
2. Build a model to learn patterns in that numerical representation
To showcase this, let's create some "known" data using linear representation formula
We'll use a linear regression formula to make a straight line with 'know' "parameters"
"""
# Create known parameters
weight = 0.7
bias = 0.3

# Create
start = 0
end = 1
step = 0.02
# A capital rapresents a matrix or a tensor, lowercase a vector
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

print(X[:10], y[:10])
print(len(X), len(y)) # 50 50

"""
Splitting data into training and test sets (one of the most important concepts in machine learning in general)
Let's create a training and test test with our data.
"""
# Create a train/test split of 80-20
train_split = int(0.8 * len(X))
print(train_split) # 40
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]
print(len(X_train), len(y_train), len(X_test), len(y_test)) # 40 40 10 10

"""
How might we better visualize our data?
This is where the data explorer's motto comes in!
"Visualize, visualize, visualize!"
"""
def plot_predictions(train_data=X_train, train_labels=y_train,
                     test_data=X_test, test_labels=y_test,
                     predictions=None):
    """
    Plots training and test data and compares predictions
    """
    plt.figure(figsize=(10, 7))
    # Plot training data in blue
    plt.scatter(train_data, train_labels, c='b', s=4, label='Training data')
    # Plot test data in green
    plt.scatter(test_data, test_labels, c='g', s=4, label='Testing data')
    # Are there predictions?
    if predictions is not None:
        # Plot the predictions if they exist
        plt.scatter(test_data, predictions, c='r', s=4, label='Predictions')
    # Show the legend
    plt.legend(prop={'size': 14})
    plt.show()

# plot_predictions()

"""
2. Build model
Our first Pytorch model!

What our model does:
* Start with random values (weight & bias)
* Look at training data and adjust the random values to better
represent (or get closer to) the ideal values (the weight & bias values
we used to create the data)

How does it do so?
Through two main algorithms:
1. Gradient descent https://youtu.be/IHZwWFHWa-w
2. Backpropagation https://youtu.be/tIeHLnjs5U8
"""
# Create a linear regression model class
class LinearRegressionModel(nn.Module): # <- almost everything in pytorch inherits from nn.Module
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.rand(1, # start with a random weight and try to adjust it to the ideal weight
                                               requires_grad=True, # can this parameter be updated via gradient descent?
                                               dtype=torch.float)) # PyTorch loves the datatype torch.float32
        self.bias = nn.Parameter(torch.rand(1, # start with a random bias and try to adjust it to the ideal bias
                                            requires_grad=True, # can this parameter be updated via gradient descent?
                                            dtype=torch.float)) # PyTorch loves the datatype torch.float32

    # Forward method to define the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor: # x is the input data
        return self.weights * x + self.bias # this is the linear regression formula

"""
Pytorch model building essentials

* torch.nn - contain all of the building block for computational graphs
(neural networks can be a computational graph)
* torch.nn.Parameter - what parameters should our model try and learn,
often a Pytorch layer from torch.nn will set these for us
* torch.nn.Module - the base class for all neural network module, 
if you subclass it, you should overwrite forward()
* torch.optim - this where the optimizers in pytorch live, they will
help with gradient descent
* def forward() - all nn.module subclasses require you to overwrite
forward(), this method decides what happens in the forward computation



Checking the contents of our Pytorch model

How we've created a model, let's see what's inside...
We can check out model parameters or what's inside with .parameters()
"""
# Set a random seed
torch.manual_seed(42)
# Create an instance of the model (this is a subcalss of the nn.module)
model_0 = LinearRegressionModel()
print(model_0) # LinearRegressionModel()
# Check out the parameters
print(model_0.parameters()) # <generator object Module.parameters at 0x7bf0a5babdf0>
print(list(model_0.parameters()))
# List named parameters
print(model_0.state_dict()) # OrderedDict([('weights', tensor([0.8823])), ('bias', tensor([0.9150]))])
# We want our model weights and bias be like the ones we set at the start

"""
Making prediction using 'torch.inference_mode()'
To check our model's predictive power, let's see how well it predicts y_test based on x_test
When we pass data trought our model, it's going to run it through the forward() method
"""
# Make predictions with model
with torch.inference_mode(): # a better .no_grad()
    y_preds = model_0(X_test)
print(y_preds)
plot_predictions(predictions=y_preds)

"""
3. Train model

The whole idea of training is for a model from some unknown parameters (may be random) to some known parameters
Or in other words from poor representation of the data to a better representation of the data
One way to measure how poor or how wrong your model predictions are is to use a loss function
* Note: Loss function may also be called cost function or criterion in different areas.
For our case, we're going to refer to it as a loss function.

Things we need to train:
* Loss function: a function to measure how wrong your model's predictions are
to the ideal outputs, lower is better.
* Optimizer: takes the loss of the model and adjusts the model's parameters
(weights & bias) to improve the loss function

And specifically for pytorch we need:
* A training loop
* A testing loop
"""
# Checkout our model's parameters
print(model_0.state_dict())
# Setup a loss function
loss_fn = nn.L1Loss()
# Setup a optimizer (stocasthic gradient descent)
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.01) # learning rate = possibly the most important hyperparameter you can set
