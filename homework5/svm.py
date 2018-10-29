import csv
import re
import numpy as np
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from sklearn import linear_model
from sklearn.svm import SVC
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import opencv




# read the csv file from new_train.csv
def readtrain():
    with open('new_train.csv','r',encoding='latin-1',errors='ignore') as csvfile:
        reader = csv.reader(csvfile)
        column1 = [row for row in reader]
    label_train = [i[1] for i in column1[1:]]
    content_train = [i[2] for i in column1[1:]]
 #   print('there is %s sentences' % len(label_train))
    train = [label_train, content_train]
    return train


# remove the URL of each tweet
def process_word(cont):
    c = []
    for i in cont:
        i = i.lower()
        clean_tweet = re.sub(r"http\S+","",i)
        tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
        clean_tweet = tokenizer.tokenize(clean_tweet)
        a = list(clean_tweet)
        b = " ".join(a)
        c.append(b)
    return c

train = readtrain()
content = process_word(train[1])
print(content)
label = train[0]
# print(label)


train_content = content[:4000]
test_content = content[4000:]
train_label = label[:4000]
test_label = label[4000:]

vectorizer = CountVectorizer()
tfidftransformer = TfidfTransformer()
tfidf = tfidftransformer.fit_transform(vectorizer.fit_transform(train_content))
# print(tfidf.shape)

# print('*************************\nSVM\n*************************')
# # svm
# text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),  ('clf', SVC(C=0.99, kernel = 'linear'))])
# text_clf = text_clf.fit(train_content, train_label)
# predicted = text_clf.predict(test_content)
# accuracy = np.mean(predicted == test_label)
# # print the accuracy
# print ("The accuracy of test is %s" % accuracy)
# # print('SVC',np.mean(predicted == test_label))
# # print(set(predicted))
