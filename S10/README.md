# Problem Statement
  1. Write a custom Resent architecture that has folllowing architecture :
      #### PrepLayer -
        Conv 3x3 s1, p1) >> BN >> RELU [64k]
      #### Layer1 -
        X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
        R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
        Add(X, R1)
      #### Layer 2 -
        Conv 3x3 [256k]
        MaxPooling2D
        BN
        ReLU
      #### Layer 3 -
        X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
        R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
        Add(X, R2)
     
     MaxPooling with Kernel Size 4
     FC Layer 
     SoftMax
     
  3. Uses One Cycle Policy such that:
        Total Epochs = 24
        Max at Epoch = 5
        LRMIN = FIND
        LRMAX = FIND
        NO Annihilation
4. Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
5. Batch size = 512
6. Use ADAM, and CrossEntropyLoss
7. Target Accuracy: 90%

# Solution 
1. Data Augmentation >> https://github.com/code4koustav/ERA_V1/blob/main/S10/Augmentation.py
2. Data Loader >> https://github.com/code4koustav/ERA_V1/blob/main/S10/Data_loader.py
3. Model >> https://github.com/code4koustav/ERA_V1/blob/main/S10/Model.py
4. Train >> https://github.com/code4koustav/ERA_V1/blob/main/S10/Train.py
5. Test >> https://github.com/code4koustav/ERA_V1/blob/main/S10/Test.py
6. Notebook >> https://github.com/code4koustav/ERA_V1/blob/main/S10/Notebook.ipynb

# Result 
<img width="694" alt="image" src="https://github.com/code4koustav/ERA_V1/assets/92668707/9c029340-431f-41b1-9ff3-94a806ad9004">
<img width="686" alt="image" src="https://github.com/code4koustav/ERA_V1/assets/92668707/db340f85-756b-4cc5-805b-798c6cbd01fd">


  

