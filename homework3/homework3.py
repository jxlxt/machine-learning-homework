import csv
import numpy
from numpy import *
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# this is batch gradient descent
#class BatchGradient:
    def batch_gd(x, y, w, l):
        w = numpy.zeros((x.shape[1], 1))      # w is matrix 6125 x 1 as temporary w
        error_sum = 0
        error0=0
        n = 1
        while True:
            z = (numpy.matmul(-x, w))  # matrix 2700 x 1   [2700x6125] x [6125x1]=2700x1
            # now set  n0 = 0.3
            learning_rate = 0.3 * (n ** (-0.9))
            for i in range(len(z)):  # i from (0,2700)
                # print("now is selecting   " + str(i) + " e-mail")
                y_tutor = 1 / (1 + math.exp(z[i]))
                dif = numpy.matrix((y_tutor - y[i][0]) * x[i, :]).transpose()
                error_sum += y[i] * math.log2(y_tutor) + (1 - y[i]) * math.log2(1 - y_tutor)
                for j in range(len(w)):  # j from (0,6125), done one e-mail than update one w
                    w[j][0] = w[j][0] - learning_rate * dif[j][0]
            error = (1 / len(z)) * (error_sum + l * numpy.matmul(w.transpose(), w))
            if abs(error-error0) < 0.1:
                break
            else:
                error0 = error
                w = w
                n+=1
                print(w)
        print("Done ,the best w is ", w)
        return error


# this is for EG gradient descent
# class ExponentialGradient:
#     def test_case_1(self):
#         data_set = a.toarray()
#         data_set_1 = numpy.array(data_set)
#         self.weighted_vote_method(data_set_1)
#
#     def weighted_vote_method(self, data_set, learning_rate = 0.3):
#         if len(data_set) <= 0:
#             raise ValueError("Data set length error.")
#         weight_result =  []
#         current_weight = [1. / len(data_set[0,:]) for i in range(len(data_set[0,:]))]
#         for i in range(len(data_set[0])):
#             #print("Current weight=\t\t" + str(current_weight))
#             current_weight_plus = current_weight
#             current_weight_minus = current_weight
#             current_weight = self.exponentiated_gradient(data_set[i,:], current_weight_plus, current_weight_minus, learning_rate)
#             weight_result.append(current_weight)
#             #print("===================")
#     def exponentiated_gradient(self, data_set, previous_weight_plus, previous_weight_minus, learning_rate):
#         if len(data_set) <= 0:
#             raise ValueError("Data set length error.")
#         if len(data_set) != len((previous_weight_plus-previous_weight_minus):
#             raise ValueError("Arguments length not equal.")
#
#         #print("Data set =\t\t" + str(data_set))
#
#         result = []
#         all_weighted_value = numpy.sum([previous_weight[i] * data_set[i] for i in range(len(data_set))])
#         # update the w+ and w-
#         numerator_plus = numpy.sum([previous_weight[i] * numpy.exp((learning_rate * data_set[i]) / all_weighted_value) for i in range(len(data_set))])
#         numerator_minus = numpy.sum([previous_weight[i] * numpy.exp(-(learning_rate * data_set[i]) / all_weighted_value) for i in range(len(data_set))])
#         #print("Numerator=\t\t\t" + str(numerator))
#
#         for i in range(len(data_set)):
#             fractions = previous_weight[i] * numpy.exp((learning_rate * data_set[i]) / all_weighted_value)
#             result.append(fractions / (numerator_plus-numerator_minus)
#         #print("Result=\t\t\t\t" + str(result))
#         return result


# b = ExponentialGradient()
# b.test_case_1()

# this is for stochastic gradient
# class StochasticGradient:
#     def stochastic_gd(tf_idf,y,w,lambda_1):
#         # i don't know what the value of the epsilon
#         # initial value
#         w_value = numpy.zeros(tf_idf.shape[1],1)
#         error_1 = 0
#         error_0 = 0
#         n = 1 # count number
#         while True:
#             h_w = (numpy.matmul(-tf_idf.toarray(),w))
#             learning_rate = 0.3*(n ** (-0.9))
#             print("this is the ", n , "times studying")
#             print("=================================")
#             for i in range(len(h_w))：
#                 y_tutor = 1 / (1+math.exp(h_w[i]))
#                 diverg = numpy.matrix((y_tutor-y[i][0]) * tf_idf[i, :]).transpose()
#                 for j in range(len(w_value)):
#                     w_value[j][0] = w_value[j][0] - learning_rate * diverg[j][0]
#             error = (1 / len(h_w)) * (error_1 + lambda_1 * numpy.matmul(w_value.transpose(), w_value))
#             if abs(error-error_0) < 0.1:
#                 break
#             else:
#                 error_0 = error
#                 w = w_value
#                 n +=1
#                 print(w)
#         print("studying is over and the best w is ", w)
#         return error

vectorizer = CountVectorizer()
transformer = TfidfTransformer()
v = TfidfVectorizer(smooth_idf=False)
stop = stopwords.words('english')
sent_tokenizer = RegexpTokenizer(r'\w+')

# first part of homework3
my_list = []
csv_file = 'train.csv'
Processed_csv = 'new_train.csv'
new_file = open(Processed_csv , 'w')
fr = csv.writer(new_file, dialect="excel")

# we need to process the data in the train.csv first



def process(mail,label):
    if label == "ham" :
        new_label = 1
    else :
        new_label = 0

    mail = mail.lower()
    mail_tokenize = sent_tokenizer.tokenize(mail)
    filtered_words = [' '.join(w for w in mail_tokenize if not w in stop)]
    filtered_words.insert(0,new_label)
    return filtered_words


with open(csv_file, 'r', encoding='latin-1') as csv_reader:
    next(csv_reader)
    # we use csv_reader function to read documents from train.csv
    reader = csv.reader(csv_reader)
    my_list = list(reader)
    # we create two variables and save the content of mail into variable mail
    for row in my_list:
        mail = row[1]
        label = row[0]
        fr.writerow(process(mail,label))
new_file.close()
with open(Processed_csv, 'r', encoding='latin-1') as new_reader:
    new_train = csv.reader(new_reader)
    words = []
    for lines in new_train:
        words.append(lines[1])
words_Frequency = vectorizer.fit_transform(words)
words_bag = vectorizer.get_feature_names()
tf_idf = transformer.fit_transform(words_Frequency)

# normalize the tf-idf matrix
# get the maximum and minimum value
tf_idf_min, tf_idf_max = tf_idf.min(), tf_idf.max()
# (matrix elements-minimum)/(maximum-minimum)
a = (tf_idf - tf_idf_min) / (tf_idf_max - tf_idf_min)
words = []
labels = []
with open('new_train.csv', 'r') as new_csv_reader:
    reader = csv.reader(new_csv_reader)
    for line in reader:
        if line:
            words.append(line[1])
            labels += line[0]

error_sum = 0
lambda_value = []
y = numpy.matrix(labels).transpose().astype(int)
w = numpy.zeros((tf_idf.shape[1],1))
cross_validation = KFold(n_splits=10)

for i in range(20):
    lambda_value.append(random.uniform(0, 1))
    for items in lambda_value:
        lambda_1 = items
        print("with this lambda", lambda_1, "we do 10-fold cross_validation")
        cv_x = numpy.zeros((2700, tf_idf.shape[1]))
        cv_y = numpy.zeros((2700, 1))
        for train, test in cross_validation.split(tf_idf):
            for i in test:
                cv_x[i, :] = tf_idf[i, :]
                cv_y[y, :] = y[i, :]
            error_sum += batch_gd(cv_x, cv_y, w, lambda_1)
        error_average = error_sum / 10
        print("with this lambda", lambda_1, "the average error is ", error_average)


