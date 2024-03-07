# Semantic Segmentation for Extracting Geologic Features from Historic Topographic Maps

## Objectives

The objective of this project was to extract specific geologic features from historic topographic USGS maps. Traditional map processing techniques typically rely on manually collected templates representing the feature of interest, followed by template matching through computer vision techniques. Yet, we were faced with thousands of scanned maps that needed to be vectorized and digitized in to a format suitable for efficient extraction and integration into analysis. Moreover, these maps presented additional complexities such as inconsistencies between map labelings and variations in features symbology, overprinting with contour lines, text and labels, and the presence of scanning artifacts like folds, rotations, and other irregularities further complicated the process. Addressing these complexities required techniques and models more advanced than those employed by traditional computer vision methods. We estimated that manually digitizing and extracting features from a single map would take 1-2 weeks for a single human expert. However, by harnessing AI techniques, our goal was to accomplish this task within minutes or hours, significantly reducing the time and effort involved.

## Model Architecture

For this project I employed  a CNN with an encoder–decoder architecture based on UNet. The data were processed with different augmentation techniques and the best network architecture was searched by running several experiments where the important parameters were tuned. I showed that even with a small number of training images promising results can be achieved. 
UNet has demonstrated remarkable potential in effectively modeling intricate spatial patterns and contextual relationships, while also generating data abstractions that generalize well to unseen data. Given the complex nature of topographic maps and the need for robust pattern recognition, UNet is an excellent deep learning architecture to explore for this specific project. 

## Network Configurations and Training

The training algorithm uses Focal Loss as the loss function which is designed to make the network focus on hard examples by giving more weight-age and also to deal with extreme class imbalance. I also tried training the network using Dice Loss (Dice loss is basically same as F1 score, in fact the IOU (Jaccard Index) metric can also be used as the loss function to run the optimization as well). During the training I keep track of the IOU metrics to monitor and measure the performance of the model during training and testing. 
* Training and validation were performed in an iterative fashion. Three parameters were considered: number of layers, number of filters per layer, and kernel size. At each iteration, different combinations of these parameters were chosen, and the training was performed. At the end, the performance of the different trained networks were compared. The average training and validation accuracy achieved over the whole dataset were used as the performance metric. The configurations that achieved the worse results have been discarded.
* number of layers most strongly affects the accuracy
* number of feature maps does not significantly affect the accuracy

## Results

After the model was trained and evaluated on unseen maps, to ensure transparency and provide a comprehensive assessment, I shared side-by-side images of the ground-truth maps/masks alongside the predicted segmented maps with the client. Furthermore, I shared the model's performance metrics on unseen maps, including metrics such as Frequency weighted IOU, F1 score, accuracy, and multi-class confusion matrix plots. These metrics provided quantitative measures of the model's performance, enabling the client to gain a deeper understanding of its accuracy, precision, and overall effectiveness.
By offering both visual and numerical assessments, I aimed to provide the client with a comprehensive and transparent evaluation of the model's performance on the task at hand.
During our discussion, I explained the specific features that the network accurately segmented, as well as those that exhibited some confusion for the network. I delved into how the training data influenced these results and emphasized the benefits of incorporating additional training samples to improve the segmentation of these features.
