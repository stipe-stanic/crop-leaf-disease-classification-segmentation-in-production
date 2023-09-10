# Crop disease classification
- This repository contains the training of a single custom multiclass ResNet classification model and the utilization of the Segment Anything Model (SAM), the pre-trained zero-shot generalization model. It also contains the integration of FastAPI functionalities and the creation of a docker image for deployment purposes.

## Dataset
- The dataset used to train neural network is an augmented and expanded version of the PlantVillage dataset. It contains 92968 images of healthy and diseased crop leaves. Inside this set, there are 10 unique crops: apple, cherry, corn, grape, peach, orange, pepper, potato, strawberry and tomato. This dataset covers 18 different types of diseases.
- The dataset contains 31 crop classes. For each plant, there is one class indicating a healthy crop, except for the orange, and one or more classes indicating crop diseases. Corn images contain real examples from the field, which means that the leaves are present in the image along with the rest of the background.
- After applying preprocessing steps on the dataset, including cleaning, segmentation, augmentation and undersampling, the data distribution has the most significant imbalance ratio of 1:4.

<img alt="The dataset after preprocessing steps" height=50% src="/plot_images/dataset/data_distribution_trimmed_and_augmented.png" width=50%/>

## Segmentation model
- An automatic segmentation system has been implemented utilizing SAM. The prompt used is a modified matrix of foreground points, and an ROI algorithm is employed to select the most suitable leaf for further processing.

<img alt="Foreground point matrix" height=50% src="/plot_images/segmentation/foreground_point_matrix.png" width=50%/>

## Classification model
- Residual network architecture is used, featuring a total of 63 layers, with the last layer having 1024 input units. This architecture consists of 5 convolutional blocks, 5 residual blocks and 2 linear blocks, overall the model has 37 million parameters.

## Methods
- The Focal loss function is used to deal with the imbalance of classes in the dataset. It aims to down-weight the loss contribution of well-classified examples and focus more on the hard examples during the training.
- Contrast Limited Adaptive Histogram Equalization (CLAHE) enhances the contrast in images. It is useful when dealing with uneven lighting conditions and varying levels of contrast.

## Evaluation
- Four methods are used for the evaluation of the classification model: AUC ROC, Cohen kappa score, F1 score and Confusion matrix
- The Model achieves an AUC ROC value of 0.99, which indicates the model's outstanding ability to distinguish between positive and negative instances. The model also achieves a Kohen cappa score of 0.99 which indicates almost a perfect agreement. 

<img alt="Classification report" height=50% src="/plot_images/training/classification_report_heatmap.png" width=50%/>

<img alt="Confusion matrix" height=50% src="/plot_images/training/confusion_matrix.png" width=50%/>

## Deployment
A docker image has been created with all dependencies required for deployment. Testing of the deployment was done on Google Cloud to validate stability and performance. Communication is achieved through the FastAPI application.

