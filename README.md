# oz-face-verification-task

## General pipeline:
1. Data preparation
2. Model development and loss function
3. Model training and validation on the CASIA WebFace dataset
4. Model evaluation on the LFW dataset

## 1. Data preparation 
### Training, validation, and test set 
The CASIA WebFace dataset contains 490,623 images of 10,572 persons. I used the initial 10,000 persons as a training set and the remaining 572 persons as a validation set. In this way, the subjects are not overlapping (subject-independent). To automate the process, I wrote ```split_casia_dataset.py``` such that the number of persons in the training and validation sets could be defined. The LFW dataset was used only for testing purposes at the end as a test set.

### Custom Dataloader
The custom dataloader was implemented to efficiently process the CASIA Webface dataset during the training and validation of the model. The dataloader creates positive and negative pairs with an equal probability for training a siamese network. The details can be found in ```dataset/datasets.py```.

## 2. Model development 
In this project, I decided to use a Siamese network. It is the most commonly used neural network architecture for measuring the similarity/dissimilarity of images/faces. Various loss functions have been proposed to efficiently train Siamese networks. The simplest ones are a binary cross-entropy to measure similarity and a contrastive loss to measure dissimilarity. There are more advanced loss functions such as triplet loss, marginal loss, center loss, arc face, etc. Considering that accuracy in this project was not a priority, I decided to use a binary cross-entropy loss function. Also, I have adopted a resnet-18-based siamese network (not pretrained) from the official Pytorh [repository](https://github.com/pytorch/examples/blob/main/siamese_network/main.py). To get more information about the model structure please see ```models/nn.py```. I also tried to use a simple custom siamese network but it did not feed the training data well. The main reason, in my opinion, was the difficulty of the given dataset (grayscale low-resolution faces with a high degree of variability in head pose, emotions, etc.). 

## 3. Model training and validation on the CASIA WebFace dataset
The model was trained on a GeForce RTX 3070 for 10 epochs with a batch size of 128. The learning rate was set to 0.0005. More detailed information including training curves can be found in the ```sigmoid_siamese.ipynb``` notebook. 
The training and validation curves (loss and accuracy) are as follows:

![](https://github.com/akuzdeuov/oz-face-verification-task/blob/main/results/train_val_losses.png)
![](https://github.com/akuzdeuov/oz-face-verification-task/blob/main/results/train_val_acc.png)

According to the training and validation losses/accuracies, the model was trained without overfitting. However, it can be also seen that the model could be trained for more epochs because the losses were still decreasing while accuracies were increasing. After training was done, the model was saved into a disk for the tesing purpose. The checkpoints can be downloaded from a [Google drive folder](https://drive.google.com/drive/folders/13HsqOOko1rxRC4n6oUk1346dPD3VksS7?usp=sharing).

## 4. Model evaluation on the LFW dataset
The evaluation code is given in the ```lfw_evaluate.ipynb``` notebook. I used cosine similarity as metric to measure similarity of two facial embeddings. To extract facial embeddings, I removed the last sequential layers of the model. As a result, the model outputs a feature embedding with the size of 512. To calculate accuracy, I set a threshold of 0.5 for the cosine similarity score. If the score is higher than the threshold then the prediction is accepted as positive otherwise as negative. Confusion matrix and classification report is given below. The model achieved 80% accuracy of the LFW dataset. Also, it is important to note that the model tends to predict more false positives than false negatives. As a result, precision for the negative class is higher than the precision for the positive class. However, recall for the positive classes is higher than the precision for the negative classes. These results are for 0.5 threshold and they will change if set a different threshold. There is always a trade-off between precision and recall. Choosing one them depends on the task. 
 
```
Confusion Matrix:
[[1929 1071]
 [ 131 2869]]

True Positives: 2869
True Negatives: 1929
False Positives: 1071
False Negatives: 131

Classification Report:
              precision    recall  f1-score   support

           0       0.94      0.64      0.76      3000
           1       0.73      0.96      0.83      3000

    accuracy                           0.80      6000
   macro avg       0.83      0.80      0.79      6000
weighted avg       0.83      0.80      0.79      6000
```
To test the model on different thresholds, we can calculate AUC for ROC as shown below. The model achieves 0.9187 AUC.  

![](https://github.com/akuzdeuov/oz-face-verification-task/blob/main/results/roc_lfw.png)

EER is the point at which the FAR and FAR are equal. The calculated EER is 16.33% and the corresponding threshold is 0.6783. 

![](https://github.com/akuzdeuov/oz-face-verification-task/blob/main/results/far_fpr_thr.png)

FAR/FRR plot

![](https://github.com/akuzdeuov/oz-face-verification-task/blob/main/results/far_fpr.png)

