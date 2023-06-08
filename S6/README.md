# This repository contains 1. Backpropagation excel file 2. CNN model having less than 20k parameter on MNIST Data

## 1. Backpropagation excel -
In this file entire backpropgation has been calculated manually of the below fuly connected layer netwrok : 
![image](https://github.com/code4koustav/ERA---First-Neural-Network/assets/92668707/d946cccb-b255-4021-b86a-cea3dca59af7)
Derivatives of E total has been calculated based on all the weights (W8,W7,W6,W5,W4,W3,W2,W1) and loss is calculated.
intially weights has been assigned randomly and loss is calculated for differnet learning rate.
Below is the snapshot of the sheet :
![image](https://github.com/code4koustav/ERA---First-Neural-Network/assets/92668707/a050c381-58be-459f-be8f-4d334e62845c)
Also loss is calculated for different learning rate (0.01,0.02,0.05,1,2) and below is the plot for loss curves :
![image](https://github.com/code4koustav/ERA---First-Neural-Network/assets/92668707/65389e7a-8e63-42c3-8184-2750e67e8f87)

## 2. CNN model having less than 20k parameter on MNIST Data -
Inorder to keep the parameters less than 20,000 we have used the below structure 
![image](https://github.com/code4koustav/ERA---First-Neural-Network/assets/92668707/3bdcfb80-3014-4665-8b35-8994ab240b2a)
After every convolution layers batchnormalization and dropout (5%) has been used.
In the final layers GAP has been used and able to achieve 99% accurecy.
Below is the model summary :
![image](https://github.com/code4koustav/ERA---First-Neural-Network/assets/92668707/6e8fb503-e098-4b17-ab76-f23fc7595c21)

