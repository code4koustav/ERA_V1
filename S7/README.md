### Purpose: Deep dive into coding and applying different blocks in 7 steps.

## Based on MNIST dataset
### Create a simple Convolutional Neural Network model and predict

### Project Setup:
Clone the project as shown below:-
About the file structure</br>
|__era1_S7_0_BasicSetup.ipynb<br/>
|__era1_S7_1_BasicSkeleton.ipynb<br/>
|__era1_S7_2_Batch_Normalization.ipynb<br/>
|__era1_S7_3_Dropout.ipynb<br/>
|__era1_S7_4_Fully_Connected_layer.ipynb<br/>
|__era1_S7_5_Augmentation.ipynb<br/>
|__era1_S7_6_LRScheduler.ipynb<br/>
|__model.py<br/>
|__README.md<br/>
|__requirements.txt<br/>
|__utils.py<br/>

### Step 1:
**File used: era1_S7_1_BasicSkeleton.ipynb**
<p>
Target:
- Establish the basic skeleton in terms of convolution and placement of transition blocks such as max pooling, 1x1's
- Attempting to reduce the number of parameters as low as possible
- Adding GAP and remove the last BIG kernel.

Results:
- Total parameters: 4572
- Best Training accuracy: 98.22
- Best Test accuracy: 98.43

Analysis:
- Structured the model as a new model class 
- The model is lighter with less number of parameters 
- The performace is reduced compared to previous models. Since we have reduced model capacity, this is expected, the model has capability to learn.   
- Next, we will be tweaking this model further and increase the capacity to push it more towards the desired accuracy.
</p>

### Step 2:
**File used: era1_S7_2_Batch_Normalization.ipynb**
<p>
Target:
- Add Batch-norm to increase model efficiency.

Results:
- Parameters: 5,088
- Best Train Accuracy: 99.02%
- Best Test Accuracy: 99.03%

Analysis:
- There is slight increase in the number of parameters, as batch norm stores a specific mean and std deviation for each layer.
- Model overfitting problem is rectified to an extent. But, we have not reached the target test accuracy 99.40%.
</p>

### Step 3:
**File used: era1_S7_3_Dropout.ipynb**
<p>
Target:
- Add Batch-norm to increase model efficiency.

Results:
- Parameters: 5,088
- Best Train Accuracy: 99.02%
- Best Test Accuracy: 99.03%

Analysis:
- There is slight increase in the number of parameters, as batch norm stores a specific mean and std deviation for each layer.
- Model overfitting problem is rectified to an extent. But, we have not reached the target test accuracy 99.40%.
</p>

### Step 4:
**File used: era1_S7_4_Fully_Connected_layer.ipynb**
<p>
Target:
- Add Regularization Dropout to each layer except last layer.

Results:
- Parameters: 5,088
- Best Train Accuracy: 97.94%
- Best Test Accuracy: 98.64%

Analysis:
- There is no overfitting at all. With dropout training will be harder, because we are droping the pixels randomly.
- The performance has droppped, we can further improve it.
- But with the current capacity,not possible to push it further.We can possibly increase the capacity of the model by adding a layer after GAP.
</p>

### Step 5:
**File used: era1_S7_5_Augmentation.ipynb**
<p>
Target:
- Add various Image augmentation techniques, image rotation, randomaffine, colorjitter

### Results:
- Parameters: 6124
- Best Training Accuracy: 97.61
- Best Test Accuracy: 99.32%

### Analysis:
- he model is under-fitting, that should be ok as we know we have made our train data harder. 
- However, we haven't reached 99.4 accuracy yet.
- The model seems to be stuck at 99.2% accuracy, seems like the model needs some additional capacity towards the end.
</p>

### Step 6:
**File used: era1_S7_6_LRScheduler.ipynb**
<p>
Target:
- Add some capacity (additional FC layer after GAP) to the model and added LR Scheduler

Results:
- Parameters: 6720
- Best Training Accuracy: 99.43
- Best Test Accuracy: 99.53

Analysis:
- The model parameters have increased
- The model is under-fitting. This is fine, as we know we have made our train data harder.  
- LR Scheduler and the additional capacity after GAP helped getting to the desired target 99.4, Onecyclic LR is being used, this seemed to perform better than StepLR to achieve consistent accuracy in last few layers
</p>

### Python script files - details:
**model.py** - This has Model_1, Model_2, Model_3, Model_4, Model_5, Model_6, Model_7 <br />
in all 7 models to achieve as the final model Model_7 a train accuracy of 99.44 and test accuracy of 99.31

**utils.py** - This file contains the following main functions
* get_device() - checks for device availability for cuda, if not gives back cpu as set device
* plot_sample_data() - plots a sample grid of random 12 images from the training data
* plot_metrics() - plots the metrics - train and test - losses and accuracies respectively
* show_summary() - displays the model summary with details of each layer
* download_train_data() - downloads train data from MNIST
* download_test_data() - downloads test data from MNIST
* create_data_loader() - common data loader function using which we create both train_loader and test_loader by appropriately passing required arguments
* train_and_predict() - trains the CNN model on the training data and uses the trained model to predict on the test data
