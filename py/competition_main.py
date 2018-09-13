import gc
import sys

try:
    model_type=sys.argv[2]
except IndexError:
    model_type='lgb'
try:
    learning_rate = float(sys.argv[3])
except IndexError:
    learning_rate = 0.02
try:
    early_stopping_rounds = int(sys.argv[4])
except IndexError:
    early_stopping_rounds = 150
num_iterations = 20000

try:
    pred_type = int(sys.argv[5])
except IndexError:
    pred_type = 1

import numpy as np
import pandas as pd
import datetime
import os
HOME = os.path.expanduser('~')

sys.path.append(f'{HOME}/kaggle/github/model/')
from params_lgbm import xgb_params_0814, train_params_dima

sys.path.append(f"{HOME}/kaggle/github/library/")
import utils
from make_file import make_feature_set
from make_submission import make_submission
from utils import logger_func
logger=logger_func()

if model_type=='lgb':
    params = train_params_dima()
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
ignore_list = [key, 'SK_ID_BUREAU', 'SK_ID_PREV', target, 'valid_no_4']


def main():

    pickle_name = 'add_clean_bureau'
    pickle_name = 'add_clean_pre'
    #  pickle_name = 'add_clean_ccb'
    #  pickle_name = 'add_clean_ins'
    #  pickle_name = 'add_clean_pos'
    #  pickle_name = sys.argv[1]

    #  base = pd.read_csv('../input/base.csv')[[key, target]]
    #  base[target] = base[target].replace(-1, np.nan)

    #  data = utils.read_df_pickle(path=f'../input/{pickle_name}*')
    #  if pickle_name.count('ins') or pickle_name.count('pos'):
    #      data.drop(target, axis=1, inplace=True)
    #  data = data.merge(base, on=key, how='inner')

    base = pd.read_csv('../input/base.csv')[[key, target]]
    path = f'../features/3_winner/*'
    #  path = f'../features/dima/*.npy'
    #  path = f'../features/go_dima/*.npy'
    data = make_feature_set(base, path)
    data[target] = data[target].replace(-1, np.nan)

    #  keras_1 = pd.read_csv('../output/20180829_105454_442features_auc0.71133_keras_prediction.csv')
    #  keras_1.fillna(0, inplace=True)
    #  t_value_1 = keras_1[target].values
    #  p_value_1 = keras_1['prediction'].values
    #  keras_1['prediction'] = t_value_1 + p_value_1
    #  data['emb_buro_prev'] = keras_1['prediction']

    data.set_index(key, inplace=True)

    use_cols = [col for col in data.columns if col not in ignore_list]
    feature_num = len(use_cols)

    submit = pd.read_csv('../input/sample_submission.csv')

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


    if len(submit)>0:
        submit[target] = result
        submit.to_csv(f'../submit/{start_time[:12]}_submit_{model_type}_rate{learning_rate}_{feature_num}features_CV{score}_LB_early{early_stopping_rounds}_iter{num_iterations}.csv', index=False)
    if len(stack_name)>0:
        #  result_stack = base[key].to_frame().merge(result_stack, on=key, how='inner')
        utils.to_pickle(path=f"../stack/{start_time[:12]}_{stack_name}_{model_type}_CV{str(score).replace('.', '-')}_{feature_num}features.fp", obj=result_stack)
        #  result_stack.to_csv(f'../output/{start_time[:12]}_stack_{model_type}_rate{learning_rate}_{feature_num}features_CV{score_avg}_LB_early{early_stopping_rounds}_iter{num_iterations}.csv', index=False)
        logger.info(f'result_stack shape: {result_stack.shape}')
    logger.info(f'FEATURE IMPORTANCE: {HOME}/kaggle/kaggle_private/home-credit-default-risk/output/cv_feature{feature_num}_importances_auc_{score}.csv')


if __name__ == '__main__':
    main()
