import csv
import numpy
import spacy
from numpy import *
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import KFold

nlp = spacy.load('en')
stop = stopwords.words('english')
sent_tokenizer = RegexpTokenizer(r'\w+')
my_list = []
csv_file = 'train.csv'
Processed_csv = 'new_train.csv'
new_file = open(Processed_csv , 'w')
fr = csv.writer(new_file, dialect="excel")
def process(mail,label):
    if label == "HillaryClinton" :
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
words = []
labels = []        
with open('new_train.csv', 'r') as new_csv_reader:
    reader = csv.reader(new_csv_reader)
    for line in reader:
        if line:
            words.append(line[1])
            labels += line[0]
