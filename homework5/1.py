import re
import os
import csv
import sys
import time
import spacy
import collections
import numpy as np
import pandas as pd
from cnn_model import *
from sklearn import metrics
from datetime import timedelta
from collections import Counter
from nltk.corpus import stopwords
import tensorflow.contrib.keras as kr
from nltk.tokenize import RegexpTokenizer


def open_file(filename, mode='r'):
    """
    commonly used file reader, change this to switch between

    """
    return open(filename, mode, encoding='latin-1', errors='ignore')


def data_process(input_file, output_file, label, label_name1, label_name2):
    nlp = spacy.load('en')
    print('Loading', input_file, '...')
    print('=========================')
    data_info = pd.read_csv(input_file)
    # stopWords = set(stopwords.words('english'))
    # tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
    i = 0
    print('Start processing data...')
    print('=========================')
    for rows in data_info[label_name1]:
        if rows == label:
            data_info[label_name1][i] = 1
        else:
            data_info[label_name1][i] = 0
        i = i + 1
        # i = 0
        # for content in data_info[label_name2]:
        #     content = content.lower()
        #     clean_tweet = re.sub(r"http\S+", "", content)
    #     clean_tweet = tokenizer.tokenize(clean_tweet)
    #     filtered_words = ' '.join(w for w in tokens if not w in stopWords)
    #     data_info[label_name2][i] = filtered_words
    #     i = i + 1
    df = pd.DataFrame(data_info, columns=[label_name1, label_name2])
    df.to_csv(output_file)
    print('Finish and output in', output_file)


data_process('train.csv', 'new_train.csv', 'HillaryClinton', 'handle', 'tweet')


def readtrain():
    with open('new_train.csv', 'r', encoding='latin-1', errors='ignore') as csvfile:
        reader = csv.reader(csvfile)
        column1 = [row for row in reader]
    label_train = [i[1] for i in column1[1:]]
    content_train = [i[2] for i in column1[1:]]
    #   print('there is %s sentences' % len(label_train))
    train = [label_train, content_train]
    return train


def process_word(cont):
    c = []
    for i in cont:
        i = i.lower()
        clean_tweet = re.sub(r"http\S+", "", i)
        tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
        clean_tweet = tokenizer.tokenize(clean_tweet)
        a = list(clean_tweet)
        b = " ".join(a)
        c.append(b)
    return c


tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
train = readtrain()
data_train = process_word(train[1])
all_data = []


def cout_word():
    result = {}
    for content in data_train:
        for word in content.split():
            if word not in result:
                result[word] = 0
            result[word] += 1
    return result


# def sort_by_count(d):
#     # 字典排序
#     d = collections.OrderedDict(sorted(d.items(), key=lambda t: -t[1]))
#     return d


counter = cout_word()
open_file('vocab.csv', mode='w').write('\n'.join(counter) + '\n')
print("Finish writing into vocab.csv")
print(len(counter))
vocabs = open('vocab.csv', 'r', encoding='latin-1', errors='ignore')
for words in vocabs:
    words = ''.join(words)
# def read_vocab(voca_dir):
#     """read dictionary"""
#     words = open_file(voca_dir).read().strip().split('/n')
#     word_to_id = dict(zip(words,range(words)))
#
#     return words,word_to_id
#
# read_vocab('vocab.csv')
#
#
# def read_category():
#     """read labels"""
#     categories = ['HillaryClition','DonaldTrump']
#     cat_to_id = dict(zip(categories,range(len(categories))))
#
#     return categories,cat_to_id
#
#
# def to_words(content,words):
#     """transfer the content of id into words"""
#     return ''.join(words[x] for x in content)
#
#
# def process_file(word_to_id, cat_to_id, max_length = 600):
#     train = readtrain()
#     content = process_word(train[1])
#     label = train[0]
#     data_id, label_id = [],[]
#     for i in range(len(content)):
#         data_id.append([word_to_id[x] for x in content[i] if x in word_to_id])
#         label_id.append(cat_to_id[label[i]])
#
#     # 使用keras提供的pad_sequences来将文本pad为固定长度
#     x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
#     y_pad = kr.utils.to_categorical(label_id)  # 将标签转换为one-hot表示
#     return x_pad, y_pad
#
#
# def batch_iter(x, y, batch_size=64):
#     """生成批次数据"""
#     data_len = len(x)
#     num_batch = int((data_len - 1) / batch_size) + 1
#
#     indices = np.random.permutation(np.arange(data_len))
#     x_shuffle = x[indices]
#     y_shuffle = y[indices]
#
#     for i in range(num_batch):
#         start_id = i * batch_size
#         end_id = min((i + 1) * batch_size, data_len)
#         yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
#
#
# # base_dir = 'data/cnews'
# # os.path.join(base_dir, 'train.csv')
# train_dir = 'train.csv'
# # os.path.join(base_dir, 'test.csv')
# # test_dir = 'train.csv'
# # os.path.join(base_dir, 'train.csv')
# val_dir = 'new_train.csv'
# # os.path.join(base_dir, 'train.csv')
# vocab_dir = 'vocab.csv'
#
# save_dir = 'checkpoints/textcnn'
# save_path = os.path.join(save_dir, 'best_validation')   # 最佳验证结果保存路径

# def get_time_dif(start_time):
#     """获取已使用时间"""
#     end_time = time.time()
#     time_dif = end_time - start_time
#     return timedelta(seconds=int(round(time_dif)))
#
# def feed_data(x_batch, y_batch, keep_prob):
#     feed_dict = {
#         model.input_x: x_batch,
#         model.input_y: y_batch,
#         model.keep_prob: keep_prob
#     }
#     return feed_dict
#
# def evaluate(sess, x_, y_):
#     """评估在某一数据上的准确率和损失"""
#     data_len = len(x_)
#     batch_eval = batch_iter(x_, y_, 128)
#     total_loss = 0.0
#     total_acc = 0.0
#     for x_batch, y_batch in batch_eval:
#         batch_len = len(x_batch)
#         feed_dict = feed_data(x_batch, y_batch, 1.0)
#         loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
#         total_loss += loss * batch_len
#         total_acc += acc * batch_len
#
#     return total_loss / data_len, total_acc / data_len
#
#
# def train():
#     print("Configuring TensorBoard and Saver...")
#     # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
#     tensorboard_dir = 'tensorboard/textcnn'
#     if not os.path.exists(tensorboard_dir):
#         os.makedirs(tensorboard_dir)
#
#     tf.summary.scalar("loss", model.loss)
#     tf.summary.scalar("accuracy", model.acc)
#     merged_summary = tf.summary.merge_all()
#     writer = tf.summary.FileWriter(tensorboard_dir)
#
#     # 配置 Saver
#     saver = tf.train.Saver()
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#
#     print("Loading training and validation data...")
#     # 载入训练集与验证集
#     start_time = time.time()
#     x_train, y_train = process_file(word_to_id, cat_to_id, config.seq_length)
#     x_val, y_val = process_file( word_to_id, cat_to_id, config.seq_length)
#     time_dif = get_time_dif(start_time)
#     print("Time usage:", time_dif)
#
#     # 创建session
#     session = tf.Session()
#     session.run(tf.global_variables_initializer())
#     writer.add_graph(session.graph)
#
#     print('Training and evaluating...')
#     start_time = time.time()
#     total_batch = 0              # 总批次
#     best_acc_val = 0.0           # 最佳验证集准确率
#     last_improved = 0            # 记录上一次提升批次
#     require_improvement = 1000   # 如果超过1000轮未提升，提前结束训练
#
#     flag = False
#     for epoch in range(config.num_epochs):
#         print('Epoch:', epoch + 1)
#         batch_train = batch_iter(x_train, y_train, config.batch_size)
#         for x_batch, y_batch in batch_train:
#             feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)
#
#             if total_batch % config.save_per_batch == 0:
#                 # 每多少轮次将训练结果写入tensorboard scalar
#                 s = session.run(merged_summary, feed_dict=feed_dict)
#                 writer.add_summary(s, total_batch)
#
#             if total_batch % config.print_per_batch == 0:
#                 # 每多少轮次输出在训练集和验证集上的性能
#                 feed_dict[model.keep_prob] = 1.0
#                 loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
#                 loss_val, acc_val = evaluate(session, x_val, y_val)   # todo
#
#                 if acc_val > best_acc_val:
#                     # 保存最好结果
#                     best_acc_val = acc_val
#                     last_improved = total_batch
#                     saver.save(sess=session, save_path=save_path)
#                     improved_str = '*'
#                 else:
#                     improved_str = ''
#
#                 time_dif = get_time_dif(start_time)
#                 msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},'\
#                     + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
#                 print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))
#
#             session.run(model.optim, feed_dict=feed_dict)  # 运行优化
#             total_batch += 1
#
#             if total_batch - last_improved > require_improvement:
#                 # 验证集正确率长期不提升，提前结束训练
#                 print("No optimization for a long time, auto-stopping...")
#                 flag = True
#                 break  # 跳出循环
#         if flag:  # 同上
#             break
#
#
# if __name__ == '__main__':
#     # if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
#     #     raise ValueError("""usage: python run_cnn.py [train / test]""")
#
#     print('Configuring CNN model...')
#     config = TCNNConfig()
#     if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
#         build_vocab(train_dir, vocab_dir, config.vocab_size)
#     categories, cat_to_id = read_category()
#     words, word_to_id = read_vocab(vocab_dir)
#     config.vocab_size = len(words)
#     model = TextCNN(config)
#     if sys.argv[1] == 'train':
#         train()
#
