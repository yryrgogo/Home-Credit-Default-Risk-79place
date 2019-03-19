is_debug = 0
is_down = 0
learning_rate = 0.05
#  learning_rate = 0.5
num_leaves = 2**12-1
import os
import re
import gc
import sys
import glob
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

#========================================================================
# original library 
HOME = os.path.expanduser('~')
sys.path.append(f"{HOME}/kaggle/data_analysis/library/")
sys.path.append(f"{HOME}/kaggle/data_analysis/model/")
import MS_utils
from params_MS import params_lgb
import utils, ml_utils
from utils import logger_func
try:
    if not logger:
        logger=logger_func()
except NameError:
    logger=logger_func()
#========================================================================

"""
argv[1]: comment
argv[2]: feature_key
"""
# Basic Args
seed = 605
#  seed = 328
set_type = 'all'
fold_n = 5
key, target, ignore_list = MS_utils.get_basic_var()
ignore_list = [key, target, 'country_group', 'down_flg']
comment = sys.argv[1]

# Base
base_path = '../input/base_exclude*'
base_path = '../input/base_Av*'
base_path = '../input/base_group*'
base = utils.read_df_pkl(base_path)[[key, target, 'country_group']]
base_train = base[~base[target].isnull()]

from scipy.sparse import vstack, csr_matrix, save_npz, load_npz, hstack
train = load_npz('../input/sp_train_8114.npz')
x_test = load_npz('../input/sp_test_8114.npz')
#  train = load_npz('../input/sp_train_1032.npz')
#  x_test = load_npz('../input/sp_test_1032.npz')
train = train.tocsr()
x_test = x_test.tocsr()
x_test = csr_matrix(x_test, dtype='float32')

if is_debug:
    train = train[:10000, :]
    x_test = x_test[:500, :]
    base_train = base_train.head(10000)
    base = base.head(10500)

Y = base_train[target]

#========================================================================
# Make Validation
kfold = MS_utils.get_kfold(base=base_train, fold_seed=seed)
kfold = zip(*kfold)
#========================================================================

#========================================================================
# PreSet
model_type = 'lgb'
metric = 'auc'
params = params_lgb()
params['num_threads'] = 46
params['num_leaves'] = num_leaves
params['learning_rate'] = learning_rate

feim_list = []
score_list = []
oof_pred = np.zeros(train.shape[0])
y_test = np.zeros(x_test.shape[0])
#========================================================================

start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
logger.info(f"{model_type} Train Start!!")
get_feim = False
n_features = train.shape[1]

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

    x_train, y_train = train[trn_idx, :], Y.iloc[trn_idx]
    x_val, y_val = train[val_idx, :], Y.iloc[val_idx]

    x_train, x_val = csr_matrix(x_train, dtype='float32'), csr_matrix(x_val, dtype='float32')

    logger.info(f"Fold{num_fold} | Train:{x_train.shape} | Valid:{x_val.shape}")

    score, tmp_oof, tmp_pred, feim, _ = ml_utils.Classifier(
        model_type=model_type
        , x_train=x_train
        , y_train=y_train
        , x_val=x_val
        , y_val=y_val
        , x_test=x_test
        , params=params
        , seed=seed
        , get_score=metric
        , get_feim=get_feim
    )

    logger.info(f"Fold{num_fold} CV: {score}")
    score_list.append(score)
    oof_pred[val_idx] = tmp_oof
    y_test += tmp_pred

y_test /= len(get_fold_list)
pred_col = 'prediction'
base[pred_col] = np.hstack((oof_pred, y_test))
base = base[[key, pred_col]]
#========================================================================
# Saving
utils.to_pkl_gzip(obj=base, path=f'../output/{start_time[4:12]}_stack_{model_type}_FOLD-{get_fold}_feat{n_features}_seed{seed}_{comment}')
#========================================================================
