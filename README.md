# machine-learning-homework
- This is for saving each UCSC 2018 fall& spring machine learning courses' homework, which contains courses: CMPS242 Machine Learning & CMPS290C Advanced Machine Learning
- Teacher: Prof. Manfred K. Warmuth

## 1. homework 1 Description
In this group assignment, what we have is an assemble of training set and an assemble of test set. $k$-fold(in this case $k=10$) cross validation is used in this assignment to find the best value of lambda and report the loss on the test set. In addition, we prove that the regularization parameter $\lambda $ can solve the over-fit problem in 9-degree polynomial curve fitting.

> skip homework 2 & 4 because they are both paper work.

## 2. homework 3 Description
In this homework, we use logistic regression as tools to handle the problem of spam email detection. The method we use have batch gradient descent, stochastic gradient descent and exponential gradient descent. The simulation results show that the logistic regression can effectively classify the spam email and ham email, but different methods have different computation time and convergence speed.

## 2. homework 5 Description
In this homework, we have a class Kaggle contest for classifying tweets between Trump and Clinton, first we use SVM as a baseline model and then implement one-hot vector, word-embedding matrixes and LSTM-RNN model for improving the accuracy.


## 2. CMPS242 final project Description
In this course, we tried to use a new depth learning framework different from Tensorflow, MXNet, and used a transfer learning method to identify pigs in this framework. By using existing data and pre-trained model stitching training, the results shows the superiority of this method.

## 3. CMPS290C Mini-Project Description
In this mini project, we compares three different non-linear dimension reduction
methods, which are [t-SNE](https://github.com/lvdmaaten/bhtsne), [LargerVis](https://github.com/lferry007/LargeVis) and [TriMap](https://github.com/eamid/trimap). The datasets we used is
Fashion-MNIST. We also use mean Precision-Recall and Trustworthiness-Continuity
to analysis the quality of different methods.

## 3. CMPS290C Final-Project Description
In this project, we mainly focus on visual question answering where text-based questions are generated about an given image, and the goal is to give correnct answer. For baseline model, we first implemented a basic MLP model and further tried to use the MCB model to improve the accuracy. At the same time, we tried using mechanism of attention and data augmentation to enhance performance. Experiments conducted on VQA dataset have shown the effectiveness of our models.
