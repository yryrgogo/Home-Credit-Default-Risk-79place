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
num_iterations = 20000

try:
    pred_type = int(sys.argv[4])
except IndexError:
    pred_type = 1

import numpy as np
import pandas as pd
import datetime
import os
HOME = os.path.expanduser('~')

sys.path.append(f'{HOME}/kaggle/github/model/')
from params_lgbm import xgb_params_0814, main_params

sys.path.append(f"{HOME}/kaggle/github/library/")
import utils
from make_file import make_feature_set
from make_submission import make_submission
from utils import logger_func
logger=logger_func()

if model_type=='lgb':
    params = main_params()
    params['learning_rate'] = learning_rate
    params['num_iterations'] = num_iterations
    params['early_stopping_rounds'] = early_stopping_rounds
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

    pickle_name = 'add_clean_bureau'
    pickle_name = 'add_clean_pre'
    #  pickle_name = 'add_clean_ccb'
    #  pickle_name = 'add_clean_ins'
    #  pickle_name = 'add_clean_pos'
    #  pickle_name = sys.argv[1]

    base = pd.read_csv('../input/base.csv')[[key, target]]
    win_path = f'../features/4_winner/*'

    #========================================================================
    # Data Load
    #========================================================================
    win_path_list = glob.glob(win_path)
    win_feature_list = utils.pararell_load_data(path_list=win_path_list, delimiter='gz')
    train = pd.concat(win_feature_list, axis=1)
    #  base_train = utils.read_pkl_gzip('../input/base_train.gz')
    #  train = pd.concat([base_train, train], axis=1)

    data.set_index(key, inplace=True)
    use_cols = [col for col in data.columns if col not in ignore_list]
    feature_num = len(use_cols)

    stack_name=''
    stack_name='add_nest'
    fold=5
    fold_type='group'
    fold_type='stratified'
    dummie=0
    seed_num=1

    score, result, result_stack = make_submission(
        logger=logger
        ,data=data
        ,key=key
        ,target=target
        ,fold=fold
        ,fold_type=fold_type
        ,params=params
        ,model_type=model_type
        ,dummie=dummie
        ,seed_num=seed_num
        ,pred_type=1
        ,ignore_list=ignore_list
        ,stack_name=stack_name
    )

    submit = pd.read_csv('../input/sample_submission.csv')
    if len(submit)>0:
        submit[target] = result
        submit.to_csv(f'../submit/{start_time[:12]}_submit_{model_type}_rate{learning_rate}_{feature_num}features_CV{score}_LB_early{early_stopping_rounds}_iter{num_iterations}.csv', index=False)
    #========================================================================
    # STACKING
    #========================================================================
    if len(stack_name)>0:
        #  result_stack = base[key].to_frame().merge(result_stack, on=key, how='inner')
        utils.to_pickle(path=f"../stack/{start_time[:12]}_{stack_name}_{model_type}_CV{str(score).replace('.', '-')}_{feature_num}features.fp", obj=result_stack)
        #  result_stack.to_csv(f'../output/{start_time[:12]}_stack_{model_type}_rate{learning_rate}_{feature_num}features_CV{score_avg}_LB_early{early_stopping_rounds}_iter{num_iterations}.csv', index=False)
        logger.info(f'result_stack shape: {result_stack.shape}')
    logger.info(f'FEATURE IMPORTANCE: {HOME}/kaggle/home-credit-default-risk/output/cv_feature{feature_num}_importances_auc_{score}.csv')


if __name__ == '__main__':
    main()
