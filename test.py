import csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import math
from sklearn.model_selection import KFold
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from scipy import sparse



def batch_gd(x, y, w, lam):
    # Set epsilon = 100
    temp = np.zeros((x.shape[1], 1))  # temp is matrix 6125 x 1 as temporary w
    error_sum = 0
    error0 = 0
    n = 1
    while True:
        exponent = (np.matmul(-x, w))  # matrix 2700 x 1   [2700x6125] x [6125x1]=2700x1
        # now set  n0 = 0.1
        rate = 1 * (n ** (-0.9))
        print("now is conducting  ", n, "   times studying")
        for i in range(len(exponent)):  # i from (0,2700)
            # print("now is selecting   " + str(i) + " e-mail")
            y_hat = 1 / (1 + math.exp(exponent[i]))
            dif = np.matrix((y_hat - y[i][0]) * x[i, :]).transpose()
           # error_sum += y[i] * math.log2(y_hat) + (1 - y[i]) * math.log2(1 - y_hat)
            for j in range(len(temp)):  # j from (0,6125), done one e-mail than update one w
                temp[j][0] = temp[j][0] - rate * dif[j][0]
        error = (1 / len(exponent)) * (error_sum + lam * np.matmul(temp.transpose(), temp))
        if abs(error - error0) < 10000:
            break
        else:
            error0 = error
            w = temp
            n += 1
            print(error0)
    print("Done ,the best w is ", w)
    return error
    # exponent2 = np.matmul(-x, w)
    # error_sum_1 = 0
    # for i in range(len(exponent2)):
    #     y_hat_2 = 1 / (1 + math.exp(exponent2[i]))
    #     error_sum_1 += y[i] * math.log2(y_hat_2) + (1 - y[i]) * math.log2(1 - y_hat_2)
    # error2 = (1 / len(exponent)) * (error_sum_1 + lam * np.matmul(temp.transpose(), temp))
    # print("do the", n, "time study again", "with this lambda", lam, "the error is ", error2)


# read file and tokenize get tf-idf matrix

words = []
labels = []
with open('new_train.csv', 'r') as readfile:
    reader = csv.reader(readfile)
    for line in reader:
        if line:
            words.append(line[1])
            labels += line[0]

vectorizer = CountVectorizer()
transformer = TfidfTransformer()
X = vectorizer.fit_transform(words)
word = vectorizer.get_feature_names()
x = transformer.fit_transform(X).toarray()  # x is matrix 3000 * 6125
y = np.matrix(labels).transpose().astype(int)
w = np.zeros((x.shape[1], 1))  # w is matrix 6125 x 1
kf = KFold(n_splits=10)


# for i in range(20):
#         cv_x = np.zeros((2700, x.shape[1]))
#         cv_y = np.zeros((2700, 1))
#         for train, test in kf.split(x):
#             for i in test:
#                 cv_x[i,:] = x[i, :]
#                 cv_y[i,:] = y[i, :]

params = {'n_estimators':500,'max_depth':4,'min_samples_split':2,'learning_rate':0.01,'loss':'ls'}
clf = ensemble.GradientBoostingRegressor(**params)
clf.fit(x,y)
mse = mean_squared_error(y,clf.predict(x))
print("MSE:%.4f"% mse)
print(sparse.csr_matrix(clf.predict(x)))

