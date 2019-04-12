#! /Users/xiaotongli/anaconda3/bin/python
# -*- coding: utf-8 -*-
# @Time    : 4/11/19 5:44 PM
# @Author  : Xiaotong Li
# @School  : University of California, Santa Cruz
# @FileName: tweet_classfication.py
# @Software: PyCharm

import re
import pandas as pd
import numpy as np
from nltk.stem.porter import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

stemmer = PorterStemmer()


train_path = "/Users/xiaotongli/PycharmProjects/tweet_classification/train.csv"
test_path = "/Users/xiaotongli/PycharmProjects/tweet_classification/test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

def remove_pattern(input_txt, pattern):
    input_txt = re.sub(r"http\S+", "", input_txt)
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt

def preprocess(input_txt):
    input_txt['label'] = input_txt['handle']
    input_txt['tidy_tweet'] = np.vectorize(remove_pattern)(input_txt['tweet'], "@[\w]*")
    # remove special characters, numbers, punctuations
    input_txt['tidy_tweet'] = input_txt['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")
    input_txt['tidy_tweet'] = input_txt['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))
    # Stemming
    tokenized_tweet = input_txt['tidy_tweet'].apply(lambda x: x.split())
    tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

        input_txt['tidy_tweet'] = tokenized_tweet
    return input_txt


def bow(input_txt):
    bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
    # bag-of-words feature matrix
    bow = bow_vectorizer.fit_transform(input_txt['tidy_tweet'])
    return bow

def tf_idf(input_txt):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
    # TF-IDF feature matrix
    tfidf = tfidf_vectorizer.fit_transform(input_txt['tidy_tweet'])
    return tfidf


train_bow = bow(preprocess(train))
test_bow = bow(preprocess(test))


# splitting data into training and validation set
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'], random_state=42, test_size=0.3)

lreg = LogisticRegression()
lreg.fit(xtrain_bow, ytrain) # training the model
test_pred = lreg.predict_proba(test_bow)
length = len(test_pred) * 3
submission = pd.DataFrame(np.arange(length).reshape((len(test), 3)), index=[i for i in range(len(test_pred))], columns=['id', 'realDonaldTrump', 'HillaryClinton'])
submission['realDonaldTrump'] = test_pred[:, 0]
submission['HillaryClinton'] = test_pred[:, 1]
submission['id'] = [i for i in range(len(test_pred))]
submission.to_csv("/Users/xiaotongli/PycharmProjects/tweet_classification/submissiong.csv")