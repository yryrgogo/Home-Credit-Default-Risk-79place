win_path = f'../features/bureau/*'
win_path = f'../features/4_winner/*'
stack_name='add_nest'
fname='bureau'
#========================================================================
# argv[1] : model_type 
# argv[2] : learning_rate
# argv[3] : early_stopping_rounds
# argv[4] : pred_type
#========================================================================
import gc
import sys

try:
    model_type=sys.argv[1]
except IndexError:
    model_type='lgb'
try:
    learning_rate = float(sys.argv[2])
except IndexError:
    learning_rate = 0.02
try:
    early_stopping_rounds = int(sys.argv[3])
except IndexError:
    early_stopping_rounds = 150
num_boost_round = 10000

try:
    pred_type = int(sys.argv[4])
except IndexError:
    pred_type = 1

import numpy as np
import pandas as pd
import datetime
import glob
import os
HOME = os.path.expanduser('~')

sys.path.append(f'{HOME}/kaggle/data_analysis/model/')
from params_lgbm import xgb_params_0814, params_home_credit

sys.path.append(f"{HOME}/kaggle/data_analysis/library/")
import utils
from make_file import make_feature_set
from submission import make_submission
from utils import logger_func
logger=logger_func()

if model_type=='lgb':
    params = params_home_credit()
    params['learning_rate'] = learning_rate
elif model_type=='xgb':
    params = xgb_params_0814()
    params['eta'] = learning_rate
elif model_type=='extra':
    params = extra_params()

start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

key = 'SK_ID_CURR'
target = 'TARGET'
ignore_list = [key, 'SK_ID_BUREAU', 'SK_ID_PREV', target]


def main():

    #  base = pd.read_csv('../input/base.csv')[[key, target]]

    #========================================================================
    # Data Load
    #========================================================================
    win_path_list = glob.glob(win_path)
    train_path_list = []
    test_path_list = []
    for path in win_path_list:
        if path.count('train'):
            train_path_list.append(path)
        elif path.count('test'):
            test_path_list.append(path)

    train_feature_list = utils.pararell_load_data(path_list=train_path_list, delimiter='gz')
    test_feature_list = utils.pararell_load_data(path_list=test_path_list, delimiter='gz')
    train = pd.concat(train_feature_list, axis=1)
    test = pd.concat(test_feature_list, axis=1)

    train.set_index(key, inplace=True)
    test.set_index(key, inplace=True)
    use_cols = [col for col in train.columns if col not in ignore_list]
    feature_num = len(use_cols)

    metric = 'auc'
    judge_flg = False
    fold=5
    fold_type='stratified'
    group_col_name=''
    dummie=1
    seed_num=1
    train_args = {
        'logger' : logger,
        'train' : train,
        'key' : key,
        'target' : target,
        'fold_type' : fold_type,
        'fold' : fold,
        'group_col_name' : group_col_name,
        'params' : params,
        'early_stopping_rounds' : early_stopping_rounds,
        'num_boost_round' : num_boost_round,
        'metric' : metric,
        'judge_flg' : judge_flg,
        'model_type' : model_type,
        'dummie' : dummie,
        'ignore_list' : ignore_list
    }

    #========================================================================
    # Variables
    #========================================================================
    train_args['test'] = test
    train_args['seed_num'] = seed_num
    submit = pd.read_csv('../input/sample_submission.csv')
    #  submit = []

    #========================================================================
    # Train & Prediction Start
    #========================================================================
    result, score, result_stack, use_cols = make_submission(
        params = train_args
    )
    #========================================================================
    # STACKING
    #========================================================================
    if len(stack_name)>0:
        logger.info(f'result_stack shape: {result_stack.shape}')
        utils.to_pkl(path=f"../stack/{start_time[4:12]}_{stack_name}_{model_type}_CV{str(score).replace('.', '-')}_{feature_num}features.fp", obj=result_stack)
    logger.info(f'FEATURE IMPORTANCE PATH: {HOME}/kaggle/home-credit-default-risk/output/cv_feature{feature_num}_importances_auc_{score}.csv')

    #========================================================================
    # Submission
    #========================================================================
    if len(submit)>0:
        if stack_name=='add_nest':
            test[target] = result
            submit[key].to_frame().merge(test.reset_index()[[key, target]], on=key, how='left')
            submit[target].fillna(0, inplace=True)
            submit.to_csv(f'../submit/{start_time[4:12]}_submit_{fname}_{model_type}_rate{learning_rate}_{feature_num}features_CV{score}_LB.csv', index=False)
        else:
            submit[target] = result
            submit.to_csv(f'../submit/{start_time[4:12]}_submit_{model_type}_rate{learning_rate}_{feature_num}features_CV{score}_LB.csv', index=False)

if __name__ == '__main__':
    main()
