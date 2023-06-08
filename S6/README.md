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
Block	N_in	Kernel	Stride	Padding	J_in	J_out	RF_in	RF_out	N_out	channel	Params
Conv1	28	3	1	1	1	1	1	3	28	12	120
Conv2	28	3	1	1	1	1	3	5	28	24	2616
Conv3	28	3	1	1	1	1	5	7	28	36	7812
MaxPool 1	28	2	2	1	1	2	7	8	15	36	5220
T1	15	1	1	1	2	2	8	8	17	12	444
Conv4	15	3	1	1	2	2	8	12	15	12	1308
Conv5	15	2	1	1	2	2	12	14	16	24	1176
MaxPool 2	16	2	2	1	2	4	14	16	9	24	2328
T2	9	1	1	1	4	4	16	16	11	12	300
Conv6	11	2	2	1	4	8	16	20	6.5	12	588
Conv7	6.5	2	2	1	8	16	20	28	4.25	24	1176
![image](https://github.com/code4koustav/ERA---First-Neural-Network/assets/92668707/3bdcfb80-3014-4665-8b35-8994ab240b2a)
