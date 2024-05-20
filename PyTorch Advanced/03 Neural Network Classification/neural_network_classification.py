"""
03. Neural network classification with pytorch

Classification is a problem of predictind whatever something is one thing or another
(there can be multiple things as the options)

Book of this lecture: https://www.learnpytorch.io/02_pytorch_classification/
All other resources: https://github.com/mrdbourke/pytorch-deep-learning
Ask a question: https://github.com/mrdbourke/pytorch-deep-learning/discussions
"""
import torch
from torch import nn
import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

"""
1. Make classification data and get it ready
"""
# Make 1000 examples
n_samples = 1000
# Create circles
X, y = make_circles(n_samples, noise=0.03, random_state=42)
print(len(X), len(y)) # 1000 1000
print(f'First 5 examples of X: \n{X[:5]}')
print(f'First 5 examples of y: \n{y[:5]}')
# Make DataFrame of circle data
circles = pd.DataFrame({'X1': X[:, 0],
                        'X2': X[:, 1],
                        'label': y})
print(circles)
# Visualize x 3
plt.scatter(x=X[:, 0],
            y=X[:, 1],
            c=y,
            cmap=plt.cm.RdYlBu)
# plt.show()
"""
Note: the data we're working with is often referred to as a toy dataset, a dataset
that is small enough to experiment but still sizeable enough to
practice the fundamentals



1.1 Check input and output shapes
"""
print(X.shape, y.shape) # (1000, 2) (1000,)
# View the first example of features and labels
X_sample = X[0]
y_sample = y[0]
print(f'Values for one sample of X: {X_sample} and the same for y: {y_sample}') # [0.75424625 0.23148074], 1
print(f'Shapes for one sample of X: {X_sample.shape} and the same for y: {y_sample.shape}') # (2, ), ()

"""
1.2 Turn data into tensors and create train and test splits
"""
# Turn data into tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
print(X[:5], y[:5]) # tensor
print(type(X), X.dtype, y.dtype) # <class 'torch.Tensor'> torch.float32 torch.float32
# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, # 0.2 = 20% of data will be test & 80% will be train
                                                    random_state=42)
print(len(X_train), len(X_test), len(y_train), len(y_test)) # 800 200 800 200
print(n_samples) # 1000

"""
2. Building a model
Let's build a model to classify our blue and red dots
To do so we want:
1. Setup device agnostic code so our code will run on an accellerato (GPU) if there is one
2. Construct a model (by subclassing nn.Module)
3. Define loss function and optimizer
4. Create a training and test loop
"""
# Make device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

"""
Now we've setup device agnostic code let's creat a model
1. Subbclasses nn.module (almost all models in Pytorch subclass nn.module
2. Create 2 nn.linear() laters that are capable of handling the shapes of our data
3. Define a forward() method that outlines the forward pass (or forward computation)
4. Instatiate an instance of our model class and send it to target device)
"""
# 1. Construct a model that subclass nn.Module
print(X_train.shape) # torch.Size([800, 2])
print(y_train[:5]) # tensor([1., 0., 0., 0., 1.])
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        # 2. Create 2 nn.Linear layers capable of handling the shape of our data
        self.layer_1 = nn.Linear(in_features=2, out_features=5) # takes 2 features and upscales to 5 features
        # The next in features must match the previous out features
        self.layer_2 = nn.Linear(in_features=5, out_features=1) # takes in 5 features from previous layer and outputs a single feature (same shape as y)

        # Or you can use sequential, if is a non complex network
        # self.two_linear_layers = nn.Sequential(
        #                nn.Linear(in_features=2, out_features=5),
        #                nn.Linear(in_features=5, out_features=1))
    # 3. Define a forward() method that outlines the forward pass
    def forward(self, x):
        return self.layer_2(self.layer_1(x)) # x (input) -> layer 1 -> layer 2 -> output
        # return self.two_linear_layers(x)

# 4. Instantiate an instance of our model class and send it to the target device
model_0 = CircleModelV0().to(device)
print(model_0)

# Let's replicate the model above using nn.Sequential
model_0 = nn.Sequential(nn.Linear(in_features=2, out_features=5),
                         nn.Linear(in_features=5, out_features=1)).to(device)
print(model_0)
print(model_0.state_dict())
# Make predictions
with torch.inference_mode():
    untrained_preds = model_0(X_test.to(device))
print(f'Lenght of predictions: {len(untrained_preds)}, ' # 200
      f'Shape: {untrained_preds.shape}') # torch.Size([200, 1]
print(f'Lenght of test samples: {len(X_test)}, ' # 200
      f'Shape: {X_test.shape}') # torch.Size([200, 2]
print(f'\nFirst 10 predictions:\n{torch.round(untrained_preds[:10])}')
print(f'\nFirst 10 labels:\n{y_test[:10]}')
print(X_test[:10], y_test[:10])

"""
2.1 Setup loss function and optimizer
Which loss function or optimizer should you use?
Again... this is problem specific
For example for regression you might want MAE or MSE
For classification you might want binary cross entropy or categorical cross entropy
As a reminder, the loss function measures how wrong your models predictions are
And for optimizers two of the most common and useful are SGD and adam,
however there are many built-in options with pytorch
* For the loss function we're going to use torch.nn.BECWithLogitsLoss(),
for more check binary cross entropy (BCE) and logits in deep learning
* For different optimizers check torch.optim on pytorch site
"""
# Setup the loss function
# loss_fn = nn.BCELoss # BCELoss = requires inputs to have gone through the sigmoid activation function prior to input to BCELoss
loss_fn = nn.BCEWithLogitsLoss() # = sigmoid activation function built-in, basically sigmoid + BCELoss but more stable
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)
# Calculate accuracy - what percentage does our model get right
"""
def accuracy_fn(y_ true, y_pred):
    correct = torch.ep(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc
"""
"""
3. Train model
To train our model, we're going to need to build a training loop:
1. Forward pass
2. Calculate the loss
3. Optimizer zero grad
4. Loss backward (backpropagation)
5. Optimizer step (gradient descent)



3.1 Going from: raw logits -> prediction probabilities -> prediction labels

Our model outputs are going to be raw logits
We can convert this logits into prediction probabilities by passing them to
some kind of activation function (sigmoid for binary classification and 
softmax for multiclass classification)
Then we convert our model's prediction probabilities to prediction labels
by either rounding them or taking the argmax()
"""
# 1. View the first 5 outputs of the forward pass on the test data
# If you are doing prediction use eval and inference
model_0.eval()
with torch.inference_mode():
    y_logits = model_0(X_test.to(device))[:5]
print(y_logits)
print(y_test[:5])
# Use the sigmoid activation function on our model logits to turn them into prediction probabilities
y_pred_probs = torch.sigmoid(y_logits)
print(y_pred_probs)
"""
For our prediction probability values, we need to perform a range-style rounding on them:
* y_pred_probs >= 0.5 y=1 (class 1)
* y_pred_probs < 0.5 y=0 (class 0)
"""
print(torch.round(y_pred_probs))
# Find the predicted labels
y_preds = torch.round(y_pred_probs)
# In full (logits -> pred probs -> pred labels)
y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))
# Check for equality
print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))
# Get rid of extra dimension
y_preds.squeeze()
print(y_preds)

"""
3.2 Building a training and test loop
"""
