import numpy as np
from scipy.misc import imsave
import cv2

def unpickle(file):
    import pickle
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='bytes')
    fo.close()
    return dict




# 生成训练集图片，如果需要png格式，只需要改图片后缀名即可。
# for j in range(1, 6):
#     dataName = "cifar-10-batches-py/data_batch_" + str(j)  # 读取当前目录下的data_batch12345文件，dataName其实也是data_batch文件的路径，本文和脚本文件在同一目录下。
#     Xtr = unpickle(dataName)
#     print(Xtr[b'data'][1].shape)
    # print(dataName + " is loading...")
    #
    # for i in range(0, 10000):
    #     img = np.reshape(Xtr[b'data'][i], (3, 32, 32))  # Xtr['data']为图片二进制数据
    #     img = img.transpose(1, 2, 0)  # 读取image
    #     picName = 'cifar-10-batches-py/' + str(Xtr[b'labels'][i]) + '_' + str(i + (j - 1)*10000) + '.jpg'
    #     # Xtr['labels']为图片的标签，值范围0-9，本文中，train文件夹需要存在，并与脚本文件在同一目录下。
    #     imsave(picName, img)
    # print(dataName + " loaded.")

# dataName = "cifar-10-batches-py/data_batch_" + str(1)  # 读取当前目录下的data_batch12345文件，dataName其实也是data_batch文件的路径，本文和脚本文件在同一目录下。
# Xtr = unpickle(dataName)
# print(Xtr[b'data'][1].shape[0])
# cv2.imshow('image', Xtr[b'data'][1])
# img = np.reshape(Xtr[b'data'][1], (3, 32, 32))
# img = (img[0]+img[1]+img[2])/3
# img = cv2.resize(img, (8, 8))
# print(img.shape)
# img = img.reshape(1, 64)






# for j in range(1, 6):
#     dataName = "cifar-10-batches-py/data_batch_" + str(j)
#     print(dataName + " is loading...")
#     Xtr = unpickle(dataName)
#     for i in range(0, 10000):
#         img = np.reshape(Xtr[b'data'][i], (3, 32, 32))  # Xtr['data']为图片二进制数据
#         img = img.transpose(1, 2, 0)  # 读取image
#         picName = 'cifar-10-batches-py/' + str(Xtr[b'labels'][i]) + '_' + str(i + (j - 1) * 10000) + '.jpg'
#         imsave(picName, img)
#         # img = (img[0] + img[1] + img[2]) / 3
#         # img = cv2.resize(img, (8, 8))
#         # img = img.reshape(1, 64)
#     print(dataName + " loaded.")
file_fashion = 'fashion20k_matrixs.txt'
file_labels = 'fashion20k_labels.txt'


from keras.datasets import fashion_mnist
minist_train = np.zeros((20000, 28*28))
y = np.zeros(20000)
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

for i in range(0,20000):
     minist_train[i] = x_train[i].reshape(1, 28*28)
     y[i] = y_train[i]
with open(file_fashion, 'wb') as h:
    np.savetxt(h, minist_train)
with open(file_labels, 'wb') as f:
    np.savetxt(f, y)
print(" fashion20k_matrixs.txt" + " has already saved! ")
print(" fashion20k_labels.txt" + " has already saved! ")
