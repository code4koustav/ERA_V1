# Implementation of simple model trined on MNIST dataset built in Pytorch farmework using google colab
## Data Overview
MNIST ("Modified National Institute of Standards and Technology") is the de facto “hello world” dataset of computer vision. Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification algorithms. As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike. The MNIST database contains 60,000 training images and 10,000 testing images. Half of the training set and half of the test set were taken from NIST's training dataset, while the other half of the training set and the other half of the test set were taken from NIST's testing dataset.

![image](https://github.com/code4koustav/ERA---First-Neural-Network/assets/92668707/d278c429-63e7-4a2f-9cd1-565e65ca9c0e)

## Usage
### 1 . model.py
This file contains the architechture of the convolution network.
which includes no. of convolution layers , padding no , strides , kernel size and activation function

### 2  . Utils.py
In this file we are doing the following opertaion -
  a. Feedforward
  b. Update weights by backpropogation
  c. Calculate loss
  d. Plot the loss & accurecy

## 3. Notebook.ipynb
In this file we are doing the following opertaion -
  a. Download data
  b. Data tranformation
  c. Calling both model.py and Utils.py
  d. Run the model in multiple epoch
  c. Model summary
  
  
  ![image](https://github.com/code4koustav/ERA---First-Neural-Network/assets/92668707/fde6da01-89b3-479c-a618-4a1a55ae48d1)


