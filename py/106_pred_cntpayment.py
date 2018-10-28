import gc
import numpy as np
import pandas as pd
import sys
import re
from glob import glob
import os
HOME = os.path.expanduser('~')
import datetime
sys.path.append(f'{HOME}/kaggle/data_analysis/model')
from params_lgbm import xgb_params_0814, params_home_credit
from xray_wrapper import Xray_Cal
sys.path.append(f'{HOME}/kaggle/data_analysis')
from model.lightgbm_ex import lightgbm_ex as lgb_ex

sys.path.append(f"{HOME}/kaggle/data_analysis/library/")
import utils
from make_file import make_feature_set
from utils import logger_func
logger=logger_func()
start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

#========================================================================
# Global Variable
from info_home_credit import hcdr_key_cols
key, target, ignore_list = hcdr_key_cols()
#========================================================================

prev_key = 'SK_ID_PREV'
target = 'CNT_PAYMENT'
ignore_list.remove('TARGET')
ignore_list.append(target)

acr = 'AMT_CREDIT'
aan = 'AMT_ANNUITY'
adp = 'AMT_DOWN_PAYMENT'
cpy = 'CNT_PAYMENT'
co_type = 'NAME_CONTRACT_TYPE'
dd = 'DAYS_DECISION'

#========================================================================
# Data Load
#========================================================================
app = utils.read_df_pkl('../input/clean_app*')
amt_list = [col for col in app.columns if col.count('AMT_')]

prev = utils.read_df_pkl('../input/clean_prev*')[[key, dd, acr, aan, cpy]]
df = prev.merge(app.drop(amt_list, axis=1), on=key, how='inner')

days_list = [col for col in df.columns if col.count('DAYS') and not(col.count('DECISION'))]
for col in days_list:
    df[col] = df[col].values - df[dd].values
df.drop([dd, 'TARGET'], axis=1, inplace=True)

train = df[~df[cpy].isnull()]
test = app

model_type='lgb'
learning_rate = 0.05
early_stopping_rounds = 100
num_boost_round = 10000
metric = 'l2'
fold=5
fold_type='kfold'
group_col_name=''
dummie=1
oof_flg=False

params = params_home_credit()
params['learning_rate'] = learning_rate
params['objective'] = 'regression'
params['metric'] = metric

LGBM = lgb_ex(logger=logger, metric=metric, model_type=model_type, ignore_list=ignore_list)

train, _ = LGBM.data_check(df=train)
test, drop_list = LGBM.data_check(df=test, test_flg=True)
if len(drop_list):
    train.drop(drop_list, axis=1, inplace=True)
    test.drop(drop_list, axis=1, inplace=True)

#========================================================================
# Train & Prediction Start
#========================================================================
LGBM = LGBM.cross_prediction(
    train=train
    ,test=test
    ,key=key
    ,target=target
    ,fold_type=fold_type
    ,fold=fold
    ,group_col_name=group_col_name
    ,params=params
    ,num_boost_round=num_boost_round
    ,early_stopping_rounds=early_stopping_rounds
    ,oof_flg=oof_flg
)

xray=False
if xray:
    for fold_num in range(fold):
        model = LGBM.fold_model_list[fold_num]
        tmp_xray = Xray_Cal(ignore_list=ignore_list, model=model).get_xray(base_xray=train)
        tmp_xray.to_csv('../output/xray.csv')
        sys.exit()

cv_score = LGBM.cv_score
result = LGBM.prediction
cv_feim = LGBM.cv_feim
feature_num = len(LGBM.use_cols)

cv_feim.to_csv(f'../valid/{start_time[4:12]}_feim_feat{feature_num}_CV{cv_score}.csv', index=False)

test[target] = result.astype(np.int64)

utils.to_df_pkl(df=test, path='../input/', fname='clean_cpy_application')
