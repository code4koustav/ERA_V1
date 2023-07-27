from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import seaborn as sns
import matplotlib.pyplot as plt
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def get_device():
    '''
    This method returns the device in use.
    If cuda(gpu) is available it would return that, otherwise it would return cpu.
    '''
    use_cuda = torch.cuda.is_available()
    return torch.device("cuda" if use_cuda else "cpu")

############################################################################# Albumentations for augmentations ######################################################################################################

class Data_Set(torchvision.datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, label

train_transforms = A.Compose(
    [
        #A.PadIfNeeded(min_height = 36,min_width = 36,p = 1),
        A.RandomCrop(height=32,width = 32),
        #A.HorizontalFlip(p = 0.5),
        A.Cutout(num_holes = 1,max_h_size = 16 , max_w_size = 16 , fill_value = 0 , p = 0.5),
        #A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.3),
        #A.CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16,fill_value=0.4734,mask_fill_value = None,p=0.3),
        A.Normalize(mean = (0.4914, 0.4822, 0.4465),std = (0.2470, 0.2435, 0.2616),p =1.0),
        ToTensorV2()
    ], p = 1.0
)

test_transforms = A.Compose(
    [
        #A.HorizontalFlip(p=0.5),
        #A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        #A.CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16,fill_value=0.4734),
        A.Normalize(mean = (0.4914, 0.4822, 0.4465),std = (0.2470, 0.2435, 0.2616),p=1.0),
        ToTensorV2()
    ], p = 1.0
)


############################################################################# Data loaders ######################################################################################################

def dataloader(gpu_batch_size, cpu_batch_size, workers, cuda):

    trainset = Data_Set(root='./data', train=True,download=True, transform = train_transforms)
    testset = Data_Set(root='./data', train=False,download=True, transform = test_transforms)
    #print('No. of images in train : ',trainset.shape[0])
    #print('No. of images in test : ',testset.shape[0])
    dataloader_args = dict(shuffle=True, batch_size=gpu_batch_size, num_workers=workers, pin_memory=True) if cuda else dict(shuffle=True, batch_size=cpu_batch_size)
    trainloader = torch.utils.data.DataLoader(trainset, **dataloader_args)
    testloader = torch.utils.data.DataLoader(testset, **dataloader_args)

    return trainloader, testloader

############################################################################# Train and Test ######################################################################################################

def get_lr(optimizer):
    """"
    for tracking how your learning rate is changing throughout training
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(model, device, train_loader, optimizer, epoch,train_losses,train_acc,schedular,criterion,lrs):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = criterion(y_pred, target)
    train_losses.append(loss)
    lrs.append(get_lr(optimizer))

    # Backpropagation
    loss.backward()
    optimizer.step()
    schedular.step()

    # Update pbar-tqdm
    
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} LR={get_lr(optimizer)} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)


def test(model,device,test_loader,test_losses,test_acc,criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    test_acc.append(100. * correct / len(test_loader.dataset))


def plot_metrics(train_accuracy, train_losses, test_accuracy, test_losses):

    sns.set(font_scale=1)
    plt.rcParams["figure.figsize"] = (25, 6)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(np.array(test_losses), 'b', label="Validation Loss")

    ax1.set_title("Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(np.array(test_accuracy), 'b', label="Validation Accuracy")

    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.show()

def identify_wrong_predictions(model,test_loader):
    '''
    Identifies the wrong predictions and plots them
    :param model: model
    :return: None
    '''

    device = get_device()
    class_names = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    misclassified_max_count = 10
    #current_misclassified_count = 0

    wrong_images, wrong_label, correct_label = [], [], []

    with torch.no_grad():
        for data, target in test_loader:
            # if current_misclassified_count > misclassified_max_count:
            # break

            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True).squeeze()

            # if pred.eq(target.view_as(pred)) == False:
            wrong_pred = (pred.eq(target.view_as(pred)) == False)
            wrong_images.append(data[wrong_pred])
            wrong_label.append(pred[wrong_pred])
            correct_label.append(target.view_as(pred)[wrong_pred])
            wrong_predictions = list(zip(torch.cat(wrong_images), torch.cat(wrong_label), torch.cat(correct_label)))
            # current_misclassified_count += 1

        fig = plt.figure(figsize=(10, 12))
        fig.tight_layout()
        for i, (img, pred, correct) in enumerate(wrong_predictions[:misclassified_max_count]):
            img, pred, target = img.cpu().numpy(), pred.cpu(), correct.cpu()
            img = np.transpose(img, (1, 2, 0)) / 2 + 0.5
            ax = fig.add_subplot(5, 5, i + 1)
            ax.axis('off')
            ax.set_title(f'\nactual : {class_names[target.item()]}\npredicted : {class_names[pred.item()]}',
                         fontsize=10)
            ax.imshow(img)

        plt.show()
    return wrong_predictions

def grad_cam(net, plot_arr):
  target_layers = [net.layer4[0].conv1]
  res = []
  for i in range(0,10):
    input_tensor = preprocess_image(plot_arr[i],
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = plot_arr[i]
    img = np.float32(img) / 255
    with GradCAM(model=net,
                target_layers=target_layers,
                use_cuda=torch.cuda.is_available()) as cam:
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=[ClassifierOutputTarget(i)])[0, :]
        cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

    img_g = Image.fromarray(cam_image)
    res.append(img_g)
  return res

