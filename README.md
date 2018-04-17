# Machine-Learning-Training-

In this repository, I go through a NLP, ANN, CNN and Kernel PCA models I learned from practice 

First is NLP, where I convert data from the Restaurant_Reviews.tsv using the nltk package in Python. 
Using the Naive Bayes Classifier we look at sentiment of restauurant reviews. Negative reviews are denoted as 0. Positive reviews are denoted as 1. We train the model to see how well it can predict positive and negative reviews in the test set. Using the confusion matrix to determine the accuracy of prediction.

Second is ANN, using the Churn Modelling.csv dataset to create an input layer, hidden layer and output layer. We make predictions and observe the accuracy using confusion matrix.  

Third is CNN, using training set of images of cats and dogs we train the CNN to predict the test images of cat and dog. This is a classification excercise using image recognition. We use Keras deep learning package for CNN in this example.

Finally, I apply the KPCA algorithm. Using the social_network_Ads.csv dataset, we split the dataset into training and testing data. Using Kernel PCA, the dataset's dimension is reduced. Then you fit the logistic regression model to the truncated datasset to examine the efficiency of the model. 
