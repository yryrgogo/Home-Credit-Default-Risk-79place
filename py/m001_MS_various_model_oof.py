is_debug = 0
is_submit = 1
learning_rate = 0.1
learning_rate = 0.01
num_threads = -1
import os
import re
import gc
import sys
import glob
import shutil
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold

#========================================================================
# original library 
HOME = os.path.expanduser('~')
sys.path.append(f"{HOME}/kaggle/data_analysis/library/")
sys.path.append(f"{HOME}/kaggle/data_analysis/model/")
from params_HC import params_lgb
from feature_manage import FeatureManage
import utils, ml_utils
from utils import logger_func
try:
    if not logger:
        logger=logger_func()
except NameError:
    logger=logger_func()
#========================================================================

#========================================================================
"""
argv[1]: comment
argv[2]: feature_key
"""
comment = sys.argv[1]
try:
    rank = np.int(sys.argv[2])
except IndexError:
    rank = 50000
#========================================================================


#========================================================================
# Global Variable
COMPETITION_NAME = 'home-credit-default-risk'
sys.path.append(f"../py")
from info_home_credit import hcdr_key_cols
key, target, ignore_list = hcdr_key_cols()
#========================================================================

#========================================================================
# Data Load
feim_path = glob.glob('../valid/use_feim/*.csv')[0]
base = utils.read_df_pkl('../input/base0*')[[key, target]].set_index(key)
manage = FeatureManage(key, target)
manage.set_base(base)
train, test = manage.feature_matrix(feim_path=feim_path, rank=rank)

if is_debug:
    train = train.head(10000)
    test = test.head(500)
Y = train[target]
print(train.shape, test.shape)
#========================================================================



# Basic Args
seed = 1208
set_type = 'all'
n_fold = 5

folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
kfold = folds.split(train, Y)

model_type_list = ['lgb', 'rmf', 'lgr']
model_type_list = ['lgb']
model_type = 'lgb'
metric = 'auc'

feim_list = []
score_list = []
oof_pred = np.zeros(len(train))
y_test = np.zeros(len(test))

use_cols = [col for col in train.columns if col not in ignore_list]
x_test = test[use_cols]
start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

for model_type in model_type_list:

    if model_type=='lgb':
        params = params_lgb()
        #  params['num_leaves'] = 4
        params['num_threads'] = num_threads
        #  params['num_threads'] = 60
        params['learning_rate'] = learning_rate
    else:
        params = {}

    logger.info(f"{model_type} Train Start!!")

    for num_fold, (trn_idx, val_idx) in enumerate(kfold):
        x_train, y_train = train[use_cols].iloc[trn_idx, :], Y.iloc[trn_idx]
        x_val, y_val = train[use_cols].iloc[val_idx, :], Y.iloc[val_idx]

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
        )
        feim_list.append(feim.set_index('feature').rename(columns={'importance':f'imp_{num_fold}'}))

        logger.info(f"Fold{num_fold} CV: {score}")
        score_list.append(score)
        oof_pred[val_idx] = tmp_oof
        y_test += tmp_pred

    n_feature = len(x_train.columns)
    del x_train, y_train, x_val, y_val
    gc.collect()

    feim = pd.concat(feim_list, axis=1)
    feim_cols = [col for col in feim.columns if col.count('imp_')]
    feim['importance'] = feim[feim_cols].mean(axis=1)
    feim.drop(feim_cols, axis=1, inplace=True)
    feim.sort_values(by='importance', ascending=False, inplace=True)
    feim['rank'] = np.arange(len(feim))+1

    cv_score = np.mean(score_list)
    logger.info(f'''
    #========================================================================
    # Model: {model_type}
    # CV   : {cv_score}
    #========================================================================''')

    y_test /= (num_fold+1)

    pred_col = 'prediction'
    train[pred_col] = oof_pred
    test[pred_col] = y_test
    train[key] = manage.base_train.index.tolist()
    test[key] = manage.base_test.index.tolist()
    train[target] = manage.base_train[target].values
    test[target] = manage.base_test[target].values
    stack_cols = [key, target, pred_col]

    df_stack = pd.concat([train[stack_cols], test[stack_cols]], ignore_index=True, axis=0)

    #========================================================================
    # Saving
    feim.to_csv(f'../valid/{start_time[4:12]}_valid_{model_type}_SET-{set_type}_feat{n_feature}_{comment}_CV{str(cv_score)[:7]}_LB.csv', index=True)
    utils.to_pkl_gzip(obj=df_stack, path=f'../stack/{start_time[4:12]}_stack_{model_type}_SET-{set_type}_feat{n_feature}_{comment}_CV{str(cv_score)[:7]}_LB')

    submit = pd.read_csv('../input/sample_submission.csv').set_index(key)
    submit[target] = test[pred_col].values
    submit_path = f'../submit/{start_time[4:12]}_submit_{model_type}_SET-{set_type}_feat{n_feature}_{comment}_CV{str(cv_score)[:7]}_LB.csv'
    submit.to_csv(submit_path, index=True)

    if is_submit:
        utils.submit(file_path=submit_path, comment=comment, COMPETITION_NAME=COMPETITION_NAME)
        shutil.move(submit_path, '../log_submit/')

    #========================================================================
