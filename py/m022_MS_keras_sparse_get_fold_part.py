is_debug = 0
from tqdm import tqdm
import os
import gc
import re
import sys
import glob
import datetime
import pandas as pd
import numpy as np
HOME = os.path.expanduser('~')
sys.path.append(f"{HOME}/kaggle/data_analysis/library/")
sys.path.append(f"../py/")
import MS_utils
import utils, ml_utils, kaggle_utils
from utils import logger_func
try:
    if not logger:
        logger=logger_func()
except NameError:
    logger=logger_func()
import time
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.metrics import mean_squared_error, roc_auc_score
#========================================================================
# Keras 
# Corporación Favorita Grocery Sales Forecasting
sys.path.append(f'{HOME}/kaggle/data_analysis/model')
from nn_keras import MS_NN
from keras import callbacks
from keras import optimizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
#========================================================================

start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

# Columns
key, target, ignore_list = MS_utils.get_basic_var()
comment = sys.argv[1]

base = utils.read_df_pkl('../input/base_group*')[[key, target, 'country_group']]
base_train = base[~base[target].isnull()]

from scipy.sparse import vstack, csr_matrix, save_npz, load_npz, hstack
train = load_npz('../input/sp_train_8114.npz')
x_test = load_npz('../input/sp_test_8114.npz')
train = train.tocsr()
x_test = x_test.tocsr()


if is_debug:
    train = train[:10000, :]
    x_test = x_test[:1000, :]
    base_train = base_train.head(10000)
    base = base.head(11000)

Y = base_train[target]
print(f"Train: {train.shape} | Test: {x_test.shape}")
# ========================================================================

#========================================================================
# CVの準備
seed = 605
fold_n = 5
kfold = MS_utils.get_kfold(base=base_train, fold_seed=seed)
kfold = zip(*kfold)
del base_train
gc.collect()
#========================================================================

#========================================================================
# NN Setting
from nn_keras import MS_NN
# NN Model Setting 
if is_debug:
    N_EPOCHS = 2
else:
    N_EPOCHS = 20
# learning_rate = 1e-4
learning_rate = 1e-3
first_batch=8 # 7: 128
from adabound import AdaBound

adabound = AdaBound(lr=learning_rate,
                final_lr=0.1,
                gamma=1e-03,
                weight_decay=0.,
                amsbound=False)


#========================================================================

#========================================================================
# Result Box
model_list = []
result_list = []
score_list = []
val_pred_list = []

oof_pred = np.zeros(train.shape[0])
y_test = np.zeros(x_test.shape[0])
#========================================================================

model_type = 'NN'

n_feature = train.shape[1]
metric = "accuracy"

#========================================================================
# Train & Prediction Start

for num_fold, (trn_idx, val_idx) in enumerate(kfold):

    #========================================================================
    # 複数スレッドでkfoldを取得する
    get_fold = sys.argv[2].split('_')
    get_fold_list = []
    for fold in get_fold:
        get_fold_list.append(np.int(fold))
    if num_fold not in get_fold_list:
        continue
    #========================================================================

    #========================================================================
    # Make Dataset
    x_train, y_train = train[trn_idx, :], Y.iloc[trn_idx]
    x_val, y_val = train[val_idx, :], Y.iloc[val_idx]

    #  x_train, x_val = csr_matrix(x_train, dtype='float32'), csr_matrix(x_val, dtype='float32')
    print(x_train.shape, x_val.shape)

    #========================================================================
    # Model
    model = MS_NN(input_cols=train.shape[1])
    opt = optimizers.Adam(lr=learning_rate)
    model.compile(loss="binary_crossentropy", optimizer=adabound, metrics=[metric])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4)
    ]

    model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val)
              , batch_size=2**first_batch, epochs=N_EPOCHS
              , verbose=2, callbacks=callbacks)

    #========================================================================
    # OOF
    y_pred = np.squeeze(model.predict(x_val))
    oof_pred[val_idx] = y_pred
    # Test
    y_test += np.squeeze(model.predict(x_test))
    #========================================================================

    del x_train, x_val, model
    gc.collect()

    #========================================================================
    # Scorring
    score = roc_auc_score(y_val, y_pred)
    print(f'AUC: {score}')
    score_list.append(score)
    #========================================================================

    # 念のため
    tmp = y_test / (num_fold+1)
    utils.to_pkl_gzip(obj=tmp, path=f'../output/{start_time[4:12]}_TMP_TEST_NN_FOLD{num_fold}_CV{score}')


y_test /= len(get_fold_list)
pred_col = 'prediction'
base[pred_col] = np.hstack((oof_pred, y_test))
base = base[[key, pred_col]]
print(f"DF Stack Shape: {base.shape}")
#========================================================================
# Saving
utils.to_pkl_gzip(obj=base, path=f'../output/{start_time[4:12]}_{comment}_stack_{model_type}_FOLD-{sys.argv[2]}_feat{n_feature}')
#========================================================================
