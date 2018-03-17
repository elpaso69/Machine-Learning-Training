# NLP 
#.tsv is tab seperated value 
#.csv is comma sperated value 
# we want tsv as there are already comma's in food reviews 

#import libraries 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

# import dataset 
dataset = pd.read_csv('C:/Users/pasan/Desktop/coding/Python training/Part 7 - NLP (text based mostly)/Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) 
# putting quoting = 3 makes you ignore double quotes 

#Cleaning the text in our dataset 
import re 
import nltk
from nltk.corpus import stopwords 
nltk.download('stopwords')
review = re.sub('[^a-zA-Z]',' ',dataset['Review'][0])
review = review.lower() 
#put words in lower case using review.lower() 
#currently the review is a string. We have to make it a list 
# with one line we made it a list. Just below  
review = review.split() 

# time to make a for loop 
review = [word for word in review if not word in set(stopwords.words('english'))]
# if you have large data it is actually better 
# to make review a set instead of a list so 
# you write set(stopwords.words('english').

'We will do stemming now' 
from nltk.stem.porter import PorterStemmer 
ps = PorterStemmer()
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

#finally we return the values back to a string 
review = ' '.join(review)

# now we will go through all 1000 reviews. Currently it is only for one review 
'corpus is a collection of text'
corpus = []
for i in range(0,1000):
   review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
   review = review.lower() 
   review = review.split() 
   from nltk.stem.porter import PorterStemmer 
   ps = PorterStemmer()
   review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
   review = ' '.join(review)
   corpus.append(review)
   
# creating the bag of words model 
# a matrix with lots of zeros are called Sparse Matrix 
# As data scientists we try to minimze sparse matrix 
from sklearn.feature_extraction.text import CountVectorizer 
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values 

# Now time for Machine Learning classfication
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix (to see incorrect predictions)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
