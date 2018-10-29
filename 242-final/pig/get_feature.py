import mxnet as mx
from mxnet import autograd
from mxnet import gluon
from mxnet import image
from mxnet import init
from mxnet import nd
from mxnet.gluon.data import vision
from mxnet.gluon.model_zoo import vision as models
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import h5py
import os

import matplotlib.pyplot as plt

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

ctx = [mx.cpu(i) for i in range(4)]  # [mx.gpu(i) for i in range(4)]

X_224, X_299, y = nd.load('train.nd')
X_224_test, X_299_test = nd.load('test.nd')


def save_features(model_name, data_train_iter, data_test_iter, ignore=False):
    # 文件已存在
    if os.path.exists('features_train_%s.nd' % model_name) and ignore:
        if os.path.exists('features_test_%s.nd' % model_name):
            return

    net = models.get_model(model_name, pretrained=True, ctx=ctx)

    for prefix, data_iter in zip(['train', 'test'], [data_train_iter, data_test_iter]):
        features = []
        for data in tqdm(data_iter):
            # 并行预测数据
            for data_slice in gluon.utils.split_and_load(data, ctx, even_split=False):
                feature = net.features(data_slice)
                if 'squeezenet' in model_name:
                    feature = gluon.nn.GlobalAvgPool2D()(feature)
                feature = gluon.nn.Flatten()(feature)
                features.append(feature.as_in_context(mx.cpu()))
            nd.waitall()

        features = nd.concat(*features, dim=0)
        nd.save('features_%s_%s.nd' % (prefix, model_name), features)


batch_size = 128

data_iter_224 = gluon.data.DataLoader(X_224, batch_size=batch_size)
data_iter_299 = gluon.data.DataLoader(X_299, batch_size=batch_size)

data_test_iter_224 = gluon.data.DataLoader(X_224_test,
                                           batch_size=batch_size)
data_test_iter_299 = gluon.data.DataLoader(X_299_test,
                                           batch_size=batch_size)

from mxnet.gluon.model_zoo.model_store import _model_sha1

for model in sorted(_model_sha1.keys()):
    print(model)
    if model == 'inceptionv3':
        save_features(model, data_iter_299, data_test_iter_299)
    else:
        save_features(model, data_iter_224, data_test_iter_224)
