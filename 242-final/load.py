from mxnet import ndarray as nd


features = nd.load('./pig/features_train_vgg11.nd')[0]
labels = nd.load('./pig/labels.nd')[0]

# print(features)
print(labels[550:60])
