## Problem Definition

Task is to train resnet model for 20 epochs on CIFAR10 dataset and observe and plot loss curves, both for test and train datasets. Next task would be to identify 10 misclassified images. And we would have to show the GradCAM output on those 10 misclassified images.

## Repo

https://github.com/code4koustav/ERA_V1/tree/main/S11

## Model

https://github.com/code4koustav/ERA_V1/blob/main/S11/models/resnet.py

## Main.py

https://github.com/code4koustav/ERA_V1/blob/main/S11/Main.py

## Utils.py

https://github.com/code4koustav/ERA_V1/blob/main/S11/Utils.py

## Training Log (model: resnet18, epochs used: 20, best validation accuracy got: 88.19)

EPOCH: 0
Loss=1.273415207862854 LR=0.005172100769491095 Batch_id=97 Accuracy=45.51: 100%|██████████| 98/98 [00:42<00:00, 2.31it/s]
Test set: Average loss: 0.0053, Accuracy: 3298/10000 (32.98%)

EPOCH: 1
Loss=0.8608632683753967 LR=0.017415071608211923 Batch_id=97 Accuracy=58.53: 100%|██████████| 98/98 [00:41<00:00, 2.34it/s]
Test set: Average loss: 0.0027, Accuracy: 5684/10000 (56.84%)

EPOCH: 2
Loss=0.8684700727462769 LR=0.03252700446835656 Batch_id=97 Accuracy=65.52: 100%|██████████| 98/98 [00:41<00:00, 2.33it/s]
Test set: Average loss: 0.0025, Accuracy: 5734/10000 (57.34%)

EPOCH: 3
Loss=0.9037236571311951 LR=0.0447128079348083 Batch_id=97 Accuracy=67.46: 100%|██████████| 98/98 [00:42<00:00, 2.33it/s]
Test set: Average loss: 0.0024, Accuracy: 5976/10000 (59.76%)

EPOCH: 4
Loss=0.8460249304771423 LR=0.049299943712960495 Batch_id=97 Accuracy=68.91: 100%|██████████| 98/98 [00:41<00:00, 2.36it/s]
Test set: Average loss: 0.0026, Accuracy: 5701/10000 (57.01%)

EPOCH: 5
Loss=0.8274180889129639 LR=0.04875038538977839 Batch_id=97 Accuracy=69.95: 100%|██████████| 98/98 [00:41<00:00, 2.36it/s]
Test set: Average loss: 0.0077, Accuracy: 3225/10000 (32.25%)

EPOCH: 6
Loss=0.9425478577613831 LR=0.04714763231091147 Batch_id=97 Accuracy=70.97: 100%|██████████| 98/98 [00:41<00:00, 2.36it/s]
Test set: Average loss: 0.0061, Accuracy: 3339/10000 (33.39%)

EPOCH: 7
Loss=0.7929680347442627 LR=0.044561732476768805 Batch_id=97 Accuracy=71.98: 100%|██████████| 98/98 [00:41<00:00, 2.37it/s]
Test set: Average loss: 0.0045, Accuracy: 4094/10000 (40.94%)

EPOCH: 8
Loss=0.7820583581924438 LR=0.041105702118626505 Batch_id=97 Accuracy=72.79: 100%|██████████| 98/98 [00:41<00:00, 2.37it/s]
Test set: Average loss: 0.0020, Accuracy: 6532/10000 (65.32%)

EPOCH: 9
Loss=0.7733492851257324 LR=0.03693058634700901 Batch_id=97 Accuracy=74.07: 100%|██████████| 98/98 [00:41<00:00, 2.38it/s]
Test set: Average loss: 0.0029, Accuracy: 5314/10000 (53.14%)

EPOCH: 10
Loss=0.7440263032913208 LR=0.03221885775556428 Batch_id=97 Accuracy=75.21: 100%|██████████| 98/98 [00:41<00:00, 2.37it/s]
Test set: Average loss: 0.0022, Accuracy: 6608/10000 (66.08%)

EPOCH: 11
Loss=0.6406944990158081 LR=0.02717644149312068 Batch_id=97 Accuracy=76.13: 100%|██████████| 98/98 [00:41<00:00, 2.38it/s]
Test set: Average loss: 0.0016, Accuracy: 7225/10000 (72.25%)

EPOCH: 12
Loss=0.671959400177002 LR=0.022023715346544757 Batch_id=97 Accuracy=77.98: 100%|██████████| 98/98 [00:40<00:00, 2.39it/s]
Test set: Average loss: 0.0023, Accuracy: 6478/10000 (64.78%)

EPOCH: 13
Loss=0.5167250633239746 LR=0.016985878173965182 Batch_id=97 Accuracy=80.00: 100%|██████████| 98/98 [00:41<00:00, 2.39it/s]
Test set: Average loss: 0.0014, Accuracy: 7608/10000 (76.08%)

EPOCH: 14
Loss=0.506597638130188 LR=0.012283107634048529 Batch_id=97 Accuracy=82.27: 100%|██████████| 98/98 [00:41<00:00, 2.39it/s]
Test set: Average loss: 0.0012, Accuracy: 8000/10000 (80.00%)

EPOCH: 15
Loss=0.495120108127594 LR=0.008120937365785891 Batch_id=97 Accuracy=84.49: 100%|██████████| 98/98 [00:40<00:00, 2.40it/s]
Test set: Average loss: 0.0010, Accuracy: 8241/10000 (82.41%)

EPOCH: 16
Loss=0.41763895750045776 LR=0.004681274182209207 Batch_id=97 Accuracy=87.06: 100%|██████████| 98/98 [00:40<00:00, 2.39it/s]
Test set: Average loss: 0.0009, Accuracy: 8513/10000 (85.13%)

EPOCH: 17
Loss=0.2833939790725708 LR=0.0021144478697759585 Batch_id=97 Accuracy=90.16: 100%|██████████| 98/98 [00:40<00:00, 2.39it/s]
Test set: Average loss: 0.0008, Accuracy: 8721/10000 (87.21%)

EPOCH: 18
Loss=0.1968851536512375 LR=0.000532641055338678 Batch_id=97 Accuracy=93.02: 100%|██████████| 98/98 [00:41<00:00, 2.39it/s]
Test set: Average loss: 0.0007, Accuracy: 8800/10000 (88.00%)

EPOCH: 19
Loss=0.1684059351682663 LR=4.98628703950403e-06 Batch_id=97 Accuracy=94.42: 100%|██████████| 98/98 [00:41<00:00, 2.38it/s]
Test set: Average loss: 0.0007, Accuracy: 8813/10000 (88.13%)

## Validation Loss and Accuracy Progression

![image](https://github.com/code4koustav/ERA_V1/assets/92668707/e5d90100-f0a2-4399-aa6b-8c0f2ba5f5b6)

## Misclassified Images Gallery

![image](https://github.com/code4koustav/ERA_V1/assets/92668707/259bb86a-3b96-4709-9795-2224bfe93c30)

