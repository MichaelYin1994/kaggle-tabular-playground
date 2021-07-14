#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202106132222
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
本模块(main.py)用于研究深度学习框架tinygrad的自动微分机制的内容。

tinygrad的项目地址：
- https://github.com/geohot/tinygrad

使用的数据地址：
- https://www.kaggle.com/c/tabular-playground-series-jun-2021
'''

import numpy as np
import pandas as pd
from tqdm import tqdm
from tinygrad.tensor import Tensor
import tinygrad.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

np.random.seed(3090)

'''
from tinygrad.tensor import Tensor
import tinygrad.optim as optim

class TinyBobNet:
  def __init__(self):
    self.l1 = Tensor.uniform(784, 128)
    self.l2 = Tensor.uniform(128, 10)

  def forward(self, x):
    return x.dot(self.l1).relu().dot(self.l2).logsoftmax()

model = TinyBobNet()
optim = optim.SGD([model.l1, model.l2], lr=0.001)

# ... and complete like pytorch, with (x,y) data

out = model.forward(x)
loss = out.mul(y).mean()
optim.zero_grad()
loss.backward()
optim.step()
'''

class TinyTabularNet:
    '''使用tinygrad写的小型全连接网络。'''
    def __init__(self, input_dim, n_classes):
        self.layer_0 = Tensor.randn(input_dim, 64)
        self.layer_1 = Tensor.randn(64, 32)
        self.layer_2 = Tensor.randn(32, n_classes)

    def forward(self, x):
        '''网络的前向传播方法，x \in (input_dim, batch_size)。'''
        hidden_0 = x.dot(self.layer_0).relu()
        hidden_1 = hidden_0.dot(self.layer_1).relu()
        hidden_2 = hidden_1.dot(self.layer_2).relu()

        return hidden_2.logsoftmax()


# def build_model(input_shape=None, n_classes=None):
#     '''使用tinygrad build一个全连接的神经网络，并采用softmax作为最后一层。'''
#     pass


if __name__ == '__main__':
    # 全局化的参数
    # ------------------------------
    N_ROWS = 20000
    N_FOLDS = 5
    IS_SHUFFLE = True
    IS_STRATIFIED = False
    GLOBAL_RANDOM_SEED = 3080

    N_EPOCHS = 50
    BATCH_SIZE = 256

    ID_COLS = ['id']
    TARGET_COLS = ['target']

    # 载入tabular playground的所有数据
    # ------------------------------
    train_df = pd.read_csv(
        './data/train.csv', nrows=N_ROWS)
    test_df = pd.read_csv(
        './data/test.csv', nrows=N_ROWS)
    sub_df = pd.read_csv(
        './data/sample_submission.csv', nrows=N_ROWS)

    # 构造K-folds的交叉验证机制，训练神经网络
    # ------------------------------
    if IS_STRATIFIED:
        folds = StratifiedKFold(
            n_splits=N_FOLDS,
            shuffle=IS_SHUFFLE,
            random_state=GLOBAL_RANDOM_SEED)
    else:
        folds = KFold(
            n_splits=N_FOLDS,
            shuffle=IS_SHUFFLE,
            random_state=GLOBAL_RANDOM_SEED)

    # 训练数据与测试数据
    X_train = train_df.drop(ID_COLS + TARGET_COLS, axis=1).values
    X_test = test_df.drop(ID_COLS, axis=1).values

    X_sc = StandardScaler()
    X_sc.fit(X_train)
    X_train = X_sc.transform(X_train)
    X_test = X_sc.transform(X_test)

    encoder = OneHotEncoder(sparse=False)
    y_train = train_df[TARGET_COLS].values
    y_train_oht = encoder.fit_transform(y_train)

    # loop每一个fold的idx，进行交叉验证
    for fold, (tra_id, val_id) in enumerate(folds.split(X_train, y_train_oht)):
        X_train_fold = X_train[tra_id]
        y_train_fold = y_train_oht[tra_id]

        X_valid_fold = X_train[val_id]
        y_valid_fold = y_train_oht[val_id]

        # 构建模型与优化器
        model = TinyTabularNet(
            input_dim=X_train.shape[1], n_classes=y_train_oht.shape[1])
        optimizer = optim.SGD([model.layer_0, model.layer_1, model.layer_2], lr=0.0001)

        # 按step训练模型
        Tensor.training = True
        batch_idx = np.arange(0, len(X_train_fold))
        np.random.shuffle(batch_idx)

        for epoch in (range(N_EPOCHS)):

            n_steps = int(np.ceil(len(X_train_fold) / BATCH_SIZE))
            for step in range(n_steps):
                X_batch = X_train_fold[batch_idx[int(step * BATCH_SIZE):int((step + 1) * BATCH_SIZE)]]
                y_batch = y_train_fold[batch_idx[int(step * BATCH_SIZE):int((step + 1) * BATCH_SIZE)]]

                # Training NN
                out = model.forward(Tensor(X_batch))
                loss = out.mul(y_batch).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                cat = np.argmax(out.cpu().data, axis=-1)
                accuracy = (cat == np.argmax(y_batch, axis=-1)).mean()
                print(accuracy)
        print('----------------------')
