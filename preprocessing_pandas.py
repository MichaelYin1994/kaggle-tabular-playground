#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202107150038
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
本模块(preprocessing_pandas.py)使用pandas构建数据读取与预处理的pipline，并训练神经网络模型。
'''

import gc
import os
import warnings
import numpy as np

import pandas as pd
from tqdm import tqdm
from numba import njit
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_log_error

from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.layers import (Add, BatchNormalization, Dense, Dropout,
                                     Input, PReLU)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import xgboost as xgb
from typing import Tuple

# 设定全局随机种子，并且屏蔽warnings
GLOBAL_RANDOM_SEED = 2022
np.random.seed(GLOBAL_RANDOM_SEED)
warnings.filterwarnings('ignore')
sns.set(style='ticks', font_scale=1.2, palette='deep', color_codes=True)
###############################################################################

def yield_group_range():
    '''group_range数组的生成器'''
    # train range: [201003, 201006), valid range: [201006, 201010)
    group_range_list = [[[201003, 201006], [201006, 201010]],
                        [[201003, 201007], [201007, 201011]],
                        [[201003, 201008], [201008, 201012]],
                        [[201003, 201009], [201009, 201101]],
                        [[201003, 201010], [201010, 201101]],
                        [[201003, 201011], [201011, 201101]]]

    for item in group_range_list:
        yield item


def build_model(verbose=False, is_compile=True, **kwargs):
    '''构建与编译全连接Neural Network。

    @Parameters:
    ----------
    verbose: {bool-like}
        决定是否打印模型结构信息。
    is_complie: {bool-like}
        是否返回被编译过的模型。
    **kwargs:
        其他重要参数。

    @Return:
    ----------
    Keras模型实例。
    '''

    # 网络关键参数与输入
    # -------------------
    n_feats = kwargs.pop('n_feats', 128)
    layer_input = Input(
        shape=(n_feats, ),
        dtype='float32',
        name='layer_input')

    # bottleneck网络结构
    # -------------------
    layer_dense_init = Dense(16, activation='relu')(layer_input)

    layer_dense_norm = BatchNormalization()(layer_dense_init)
    layer_dense_dropout = Dropout(0.3)(layer_dense_norm)
    layer_dense = Dense(8, activation='relu')(layer_dense_dropout)
    layer_dense_prelu = PReLU()(layer_dense)

    layer_dense_norm = BatchNormalization()(layer_dense)
    layer_dense_dropout = Dropout(0.3)(layer_dense_norm)
    layer_dense = Dense(16, activation='relu')(layer_dense_dropout)
    layer_dense_prelu = PReLU()(layer_dense)

    # 局部残差连接
    # -------------------
    layer_total = Add()([layer_dense_init, layer_dense_prelu])
    layer_pred = Dense(
        3, activation='linear', name='layer_output')(layer_total)
    model = Model([layer_input], layer_pred)

    if verbose:
        model.summary()
    if is_compile:
        model.compile(
            loss='mse', optimizer=Adam(0.0003),
            metrics=[tf.keras.losses.MeanSquaredLogarithmicError()])
    return model

@njit
def gradient(predt: np.ndarray, y) -> np.ndarray:
    '''Compute the gradient squared log error.'''
    return (np.log1p(predt) - np.log1p(y)) / (predt + 1)


@njit
def hessian(predt: np.ndarray, y) -> np.ndarray:
    '''Compute the hessian for squared log error.'''
    return ((-np.log1p(predt) + np.log1p(y) + 1) /
            np.power(predt + 1, 2))


# def squared_log(predt: np.ndarray,
#                 dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
#     '''Squared Log Error objective. A simplified version for RMSLE used as
#     objective function.
#     '''
#     predt[predt < -1] = -1 + 1e-6
#     grad = gradient(predt, dtrain)
#     hess = hessian(predt, dtrain)
#     return grad, hess


def squared_log(labels, predt) -> Tuple[np.ndarray, np.ndarray]:
    '''Squared Log Error objective. A simplified version for RMSLE used as
    objective function.
    '''
    predt[predt < -1] = -1 + 1e-6
    grad = gradient(predt, labels)
    hess = hessian(predt, labels)
    return grad, hess


def rmsle(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    ''' Root mean squared log error metric.'''
    y = dtrain.get_label()
    predt[predt < -1] = -1 + 1e-6
    elements = np.power(np.log1p(y) - np.log1p(predt), 2)
    return 'PyRMSLE', float(np.sqrt(np.sum(elements) / len(y)))


if __name__ == '__main__':
    # 全局化的参数
    # -------------------
    PATH = './data/tabular-playground-series-jul-2021'
    IS_VISUALIZING_DATA = False

    # 读取数据进入Memory
    # -------------------
    train_df = pd.read_csv(
        os.path.join(PATH, 'train.csv'))
    test_df = pd.read_csv(
        os.path.join(PATH, 'test.csv'))

    feat_col_names = [
        'deg_C', 'relative_humidity', 'absolute_humidity',
        'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5'
    ]
    target_col_names = [
        'target_carbon_monoxide', 'target_benzene', 'target_nitrogen_oxides'
    ]
    total_col_names = feat_col_names + target_col_names

    if IS_VISUALIZING_DATA:
        fig, ax_objs = plt.subplots(len(total_col_names), 1, figsize=(14, 18))
        ax_objs = ax_objs.ravel()

        for idx, feat_name in enumerate(feat_col_names):
            ax = ax_objs[idx]

            if 'target' not in feat_name:
                ax.plot(train_df[feat_name].values,
                        linestyle='-', color='k', linewidth=1.5)
            else:
                ax.plot(train_df[feat_name].values,
                        linestyle='-', color='r', linewidth=1.5)

            ax.grid(True)
            ax.set_xlim(0, len(train_df))
            ax.set_ylim(0, )
            ax.set_xlabel(feat_name, fontsize=10)
            ax.tick_params(axis='both', labelsize=10)
        plt.tight_layout()
        plt.savefig('./plots/ts_visualizing.png', dpi=600, bbox_inches='tight')

    # 数据预处理
    # -------------------
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    total_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    del train_df, test_df

    total_df['date_time'] = pd.to_datetime(total_df['date_time'])
    total_df['year'] = total_df['date_time'].dt.year
    total_df['month'] = total_df['date_time'].dt.month
    total_df['day'] = total_df['date_time'].dt.day
    total_df['hour'] = total_df['date_time'].dt.hour

    # 特征工程
    # -------------------
    shift_size_hour_list = [1, 2, 3, 4, 5, 6, 12, 24, 72]
    rolling_window_size_hour_list = [6, 12, 24, 48, 168]

    # shift features
    # ************
    for feat_name in feat_col_names:
        for shift_size in shift_size_hour_list:
            total_df['{}_shift_{}'.format(feat_name, shift_size)] = \
                total_df[feat_name].shift(shift_size)

    # rolling window features
    # ************
    for feat_name in feat_col_names:
        for window_size in rolling_window_size_hour_list:
            total_df['{}_window_{}_mean'.format(feat_name, window_size)] = \
                total_df[feat_name].rolling(window_size, min_periods=1).mean()
            total_df['{}_window_{}_std'.format(feat_name, window_size)] = \
                total_df[feat_name].rolling(window_size, min_periods=1).std()
            total_df['{}_window_{}_median'.format(feat_name, window_size)] = \
                total_df[feat_name].rolling(window_size, min_periods=1).median()

    xgb_params = {'n_estimators': 5000,
                  'max_depth': 4,
                  'learning_rate': 0.05,
                  'verbosity': 0,
                  'booster': 'gbtree',
                  'colsample_bytree': 0.98,
                  'colsample_bylevel': 0.98,
                  'subsample': 0.98,
                  'objective': squared_log,
                  'disable_default_eval_metric': 1,
                  'random_state': GLOBAL_RANDOM_SEED}

    # 划分folds
    # -------------------
    train_df = total_df.query('is_train == 1').drop(
        ['is_train'] + target_col_names, axis=1).reset_index(drop=True)
    train_target_df = total_df.query(
        'is_train == 1')[target_col_names].reset_index(drop=True)
    train_df.drop(['date_time'], axis=1, inplace=True)
    train_df.fillna(0, inplace=True)

    test_df = total_df.query('is_train == 0').drop(
        ['is_train'] + target_col_names, axis=1).reset_index(drop=True)
    test_df.drop(['date_time'], axis=1, inplace=True)

    group_array = train_df['year'].values * 100 + train_df['month'].values

    test_pred_val_list, valid_score_list = [], []
    for fold, group_range_list in enumerate(yield_group_range()):
        train_group_bool = (group_array >= group_range_list[0][0]) & \
            (group_array < group_range_list[0][1])
        valid_group_bool = (group_array >= group_range_list[1][0]) & \
            (group_array < group_range_list[1][1])

        X_train = train_df.values[train_group_bool]
        y_train = train_target_df.values[train_group_bool]

        X_val = train_df.values[valid_group_bool]
        y_val = train_target_df.values[valid_group_bool]

        X_test = test_df.values
        '''
        # 训练残差网络
        # --------------
        # 对训练数据进行归一化
        X_sc = StandardScaler()
        X_sc.fit(X_train)

        X_train_scaled = X_sc.transform(X_train)
        X_val_scaled = X_sc.transform(X_val)
        X_test_scaled = X_sc.transform(X_test)

        # 对target数据进行归一化
        y_sc = StandardScaler()
        y_sc.fit(y_train)

        y_train_scaled = y_sc.transform(y_train)
        y_val_scaled = y_sc.transform(y_val)

        model = build_model(n_feats=X_train.shape[1])

        early_stop = EarlyStopping(
            monitor='val_mean_squared_logarithmic_error', mode='min',
            verbose=0, patience=150,
            restore_best_weights=True)

        history = model.fit(
            x=[X_train_scaled], y=y_train_scaled,
            batch_size=1024,
            epochs=600,
            validation_data=([X_val_scaled], y_val_scaled),
            callbacks=[early_stop],
            verbose=0)

        valid_pred = model.predict(X_val_scaled)
        valid_pred = y_sc.inverse_transform(valid_pred).clip(0, )
        test_pred = model.predict(X_test_scaled)
        test_pred = y_sc.inverse_transform(test_pred).clip(0, )

        '''
        # 训练XGBoost
        # --------------
        valid_pred, test_pred = [], []
        for i in range(3):
            xgb_reg = xgb.XGBRegressor(**xgb_params)
            xgb_reg.fit(
                X_train, y_train[:, i],
                eval_set=[(X_val, y_val[:, i])],
                early_stopping_rounds=150,
                eval_metric=rmsle,
                verbose=False)
            valid_pred.append(xgb_reg.predict(
                X_val, ntree_limit=xgb_reg.best_iteration+1).reshape(-1, 1))
            test_pred.append(xgb_reg.predict(
                X_test, ntree_limit=xgb_reg.best_iteration+1).reshape(-1, 1))
        valid_pred = np.hstack(valid_pred).clip(0, )
        test_pred = np.hstack(test_pred).clip(0, )

        # 评估valid结果
        print('-- fold: {}, msle: {}, mean msle: {:.5f}'.format(
            fold, mean_squared_log_error(y_val, valid_pred, multioutput='raw_values'),
            mean_squared_log_error(y_val, valid_pred)
        ))
        valid_score_list.append(
            mean_squared_log_error(y_val, valid_pred, multioutput='raw_values'))
        test_pred_val_list.append(test_pred)

    print('-- fold valid mean msle: {}, mean msle accross cols: {:.5f}'.format(
        np.mean(valid_score_list, axis=0), np.mean(valid_score_list)))

    # 保存需要提交的submissions
    # -------------------
    test_pred_val = np.mean(test_pred_val_list, axis=0)
    test_pred_val = test_pred_val.clip(0, )

    sub_df = pd.read_csv(
        os.path.join(PATH, 'sample_submission.csv'))
    sub_df['target_carbon_monoxide'] = test_pred[:, 0]
    sub_df['target_benzene'] = test_pred[:, 1]
    sub_df['target_nitrogen_oxides'] = test_pred[:, 2]

    sub_df.to_csv('./submissions/sub.csv', index=False, encoding='utf-8')
