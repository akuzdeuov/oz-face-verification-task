# oz-face-verification-task

## Pipeline:
1. Data preparation 
2. Model development
3. Model training and validation on the CASIA WebFace dataset
4. Model evaluation on the LFW dataset
5. ```TODO```

## 1. Data Preparation 
### Training, Validation, and Testing Set 
I was given two datasets for this task: CASIA Webface for training and LFW for testing. The CASIA WebFace contains 490,623 images of 10,572 persons. Thus, I split this dataset into training and validation sets. I used the initial 10,000 persons for training and the remaining 572 persons for validation. In this way, the subjects are not overlapping (subject-independent). I wrote a simple script (```split_casia_dataset.py```) to automate the dataset splitting process. The LFW dataset was used only for testing purposes at the end.

### Constructing Positive and Negative Pairs
Positive and negative pairs were generated randomly (with an equal probability) for the training and validation sets. The details can be found in ```dataset/datasets.py```. Considering that the faces are aligned and they will be aligned in practice, I applied only horizontal flip augmentation to the training set. 

## 2. Model Development 
I employed a siamese network in this project because it is the most commonly used neural network architecture for measuring the similarity/dissimilarity of images/faces. Numerous loss functions exist to train siamese networks efficiently. The simplest ones are a binary cross-entropy to measure similarity and a contrastive loss to measure dissimilarity. There are many advanced loss functions, such as triplet loss, marginal loss, center loss, arc face, etc. For the sake of simplicity and considering that accuracy was not a priority in the project, I decided to use a binary cross-entropy loss function to measure similarity of image pairs. Also, I have adopted a resnet-18-based siamese network (not pretrained) from the official Pytorch [repository](https://github.com/pytorch/examples/blob/main/siamese_network/main.py). To get more information about the model's structure please see ```models/nn.py```. I also tried to train a simple custom siamese network but it did not feed the training data well. The main reason, in my opinion, was the difficulty of the given dataset (grayscale low-resolution faces with a high degree of variability in head pose, emotions, etc.). In addition, I trained the resnet-18-based siamese network using a contrastive loss function. However, it did not converge well. Thus, I proceeded with the BCE loss. The contrastive loss function can be found in ```losses/losses.py``` and training details in ```contrastive_siamese.ipynb```.

## 3. Model training and validation on the CASIA WebFace dataset
The model was trained for 10 epochs with a batch size of 128 on a NVIDIA GeForce RTX 3070 GPU. The initial learning rate was 0.0005. It is important to note that the data loader for the training set was created in the beginning of each epoch. In this way, the model 'sees' a new set of pairs in each epoch. The details can be found in the ```sigmoid_siamese.ipynb``` notebook. To check the model performance, the training and validation curves (loss and accuracy) were plotted:

![](https://github.com/akuzdeuov/oz-face-verification-task/blob/main/results/train_val_losses.png)
![](https://github.com/akuzdeuov/oz-face-verification-task/blob/main/results/train_val_acc.png)

We can see that the model was trained without overfitting. However, it could be trained it for a few more epochs because the losses were still decreasing for both sets. The trained model was saved in a PTH format for testing purposes. The checkpoints can be downloaded from [Google Drive](https://drive.google.com/drive/folders/13HsqOOko1rxRC4n6oUk1346dPD3VksS7?usp=sharing).

## 4. Model evaluation on the LFW dataset
The evaluation code is given in the ```lfw_evaluate.ipynb``` notebook. The test set (LFW) provides 6,000 pairs of faces. I used the cosine distance metric to measure the similarity of two faces. Considering that the model was trained using the BCE loss, I had to use the backbone network (i.e., without the last sequential layers) to extract facial embeddings. In this case, the model outputs a feature embedding with a dimension of 512. I set a cut-off threshold (0.5) for the cosine distance score. If the cosine distance between two faces is lower than the threshold then the prediction is accepted as positive, else as negative. A confusion matrix and a classification report are given below. The model achieves 80% accuracy on the LFW dataset. The model predicted more false positives than false negatives. Therefore, the precision for the negative class is higher than for the positive class. As a result, recall for the positive class is higher than for the negative class. There is always a trade-off between precision and recall. Prioritizing one of them depends on the application. 
 
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
To evaluate the model across all possible thresholds, we can use Area Under the Curve (AUC), which represents the area under the ROC curve. In this case, the model achieves 0.9187 AUC.  

![](https://github.com/akuzdeuov/oz-face-verification-task/blob/main/results/roc_lfw.png)

Equal Error Rate (EER) is an another important metric. It is the point at which the FAR and FRR are equal. In our case, the calculated EER at the threshold of 0.3216 is 16.33%. 

![](https://github.com/akuzdeuov/oz-face-verification-task/blob/main/results/far_fpr_thr.png)

FRR refers to the probability of a valid user being denied access by the system, whereas FAR represents the chance of an unauthorized user gaining access. Ideally, it's crucial to strike a balance between security and user-friendliness, however, depending on the situation one might be prioritized over the other. 

![](https://github.com/akuzdeuov/oz-face-verification-task/blob/main/results/far_fpr.png)

## 5.TODO
* Normalize the datasets by subtracting the mean and a division by the standard deviation. Mean and Std Dev are computed on the training set of CASIA Webface.
* Use advanced loss functions: triplet loss, marginal loss, center loss, etc.
* Use different model architecture, try a pretrained model
* Set a learning rate scheduler
* Use Weights & Biases to keep track of experiments     

