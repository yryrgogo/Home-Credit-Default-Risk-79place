#  from time import sleep
#  sleep(360)
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

try:
    dir_num = int(sys.argv[6])
except IndexError:
    dir_num = 3

try:
    p_num = int(sys.argv[7])
except IndexError:
    p_num = 1


sys.path.append('../model/')
from lgbm_clf import prediction, cross_prediction
from params_lgbm import train_params_0816, train_params_0729, train_params_dart, xgb_params_0814, train_params_dima

if model_type=='lgb':
    if p_num==0:
        #  submit_params = train_params_0729()
        submit_params = train_params_dima()
        #  submit_params = train_params_dart()
    #  submit_params = train_params_0816()
    elif p_num==1:
        submit_params = train_params_dima()
    submit_params['learning_rate'] = learning_rate
elif model_type=='xgb':
    submit_params = xgb_params_0814()
    submit_params['eta'] = learning_rate
elif model_type=='extra':
    submit_params = extra_params()


import numpy as np
import pandas as pd
import glob, os
import datetime


sys.path.append('../../../github/module/')
from preprocessing import set_validation, split_dataset, factorize_categoricals, get_dummies
from convinience_function import get_categorical_features, get_numeric_features
from make_file import make_feature_set
from logger import logger_func


start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
logger = logger_func()

unique_id = 'SK_ID_CURR'
target = 'TARGET'
ignore_features = [unique_id, target, 'valid_no', 'is_train', 'is_test']



def make_submission(data, dummie=0, path='', val_col='valid_no'):

    data = make_feature_set(data, path)
    keras_1 = pd.read_csv('../output/20180829_105454_442features_auc0.71133_keras_prediction.csv')
    keras_1.fillna(0, inplace=True)
    t_value_1 = keras_1[target].values
    p_value_1 = keras_1['prediction'].values
    #  keras_1['prediction'] = t_value_1 + p_value_1
    data['emb_buro_prev'] = keras_1['prediction']
    #  logger.info(data['emb_buro_prev'].sort_values())
    data.set_index(unique_id, inplace=True)

    categorical = get_categorical_features(data, [])

    if dummie==0:
        data = factorize_categoricals(data, categorical)
        categorical=[]
    elif dummie==1:
        data = get_dummies(data, categorical)
        categorical=[]

    train = data.query('is_train==1')
    test = data.query('is_test==1')
    logger.info(f'train shape: {train.shape}')
    logger.info(f'test shape: {test.shape}')
    train.drop(['is_train', 'is_test'], axis=1, inplace=True)
    test.drop(['is_train', 'is_test'], axis=1, inplace=True)

    for col in test.columns:
        length = len(test[col].drop_duplicates())
        if length <=1:
            logger.info(f'***********WARNING************* LENGTH {length} COLUMN: {col}')

    seed_list = [
        #  1208,
        #  308,
        #  1012,
        #  1128,
        605,
        #  1222,
        #  405,
        #  503,
        #  2018,
        #  1212
    ]

    tmp_result = np.zeros(len(test))
    score_list = []

    for i, seed in enumerate(seed_list):
        if model_type=='lgb':
            submit_params['bagging_seed'] = seed
            submit_params['data_random_seed'] = seed
            submit_params['feature_fraction_seed'] = seed
            submit_params['random_seed'] = seed
        elif model_type=='xgb':
            submit_params['seed'] = seed
        elif model_type=='extra':
            submit_params['seed'] = seed

        if pred_type==0:
            ' 予測 '
            result,feature_num = prediction(
                logger=logger,
                train=train,
                test=test,
                target=target,
                categorical_feature=categorical,
                params = submit_params,
                num_iterations=num_iterations,
                learning_rate=learning_rate,
                model_type=model_type
            )
            score = '?'
        elif pred_type==1:
            ' 予測 '
            y_pred, score, feature_num, stack = cross_prediction(
                logger=logger,
                train=train,
                test=test,
                target=target,
                categorical_feature=categorical,
                val_col=val_col,
                params = submit_params,
                num_iterations=num_iterations,
                learning_rate=learning_rate,
                early_stopping_rounds=early_stopping_rounds,
                model_type=model_type
            )

            ' for stacking pred_value '
            if i==0:
                result_stack = stack.copy()
            else:
                pred_stack = stack[target].values
                result_stack[target] = result_stack[target].values + pred_stack

            tmp_result += y_pred
            score_list.append(score)
            score_avg = np.mean(score_list)
            logger.info(f'CURRENT AUC AVERAGE: {score_avg}')

    result_stack[target] = result_stack[target].values / len(seed_list)
    result = tmp_result / len(seed_list)

    submit = pd.read_csv('../data/sample_submission.csv')
    submit[target] = result
    if pred_type==0:
        submit.to_csv(f'../submit/{start_time[:12]}_submit_{model_type}_rate{learning_rate}_{feature_num}features_CV{score}_LB_early{early_stopping_rounds}_iter{num_iterations}.csv', index=False)
    elif pred_type==1:
        submit.to_csv(f'../submit/{start_time[:12]}_submit_{model_type}_rate{learning_rate}_{feature_num}features_CV{score_avg}_LB_early{early_stopping_rounds}_iter{num_iterations}.csv', index=False)

    base = pd.read_csv('../data/base.csv')
    logger.info(f'base shape: {base.shape}')
    result_stack = base[unique_id].to_frame().merge(result_stack, on=unique_id, how='inner')
    logger.info(f'result_stack shape: {result_stack.shape}')
    result_stack.to_csv(f'../output/{start_time[:12]}_stack_{model_type}_rate{learning_rate}_{feature_num}features_CV{score_avg}_LB_early{early_stopping_rounds}_iter{num_iterations}.csv', index=False)


def check_length():
    path_list = glob.glob('../features/3_winner/*.npy')
    for path in path_list:
        data = np.load(path)
        print(path)
        print(data.shape)


def main():
    #  check_length()
    #  sys.exit()
    #  data = pd.read_csv('../data/sample_submission.csv')
    data = pd.read_csv('../data/base.csv')

    #  nn_train = pd.read_csv('../output/oof_preds_NN_go_1195_dr_01_l2_0005_561_dr_01_l2_0005_0.797315952118905.csv')
    #  nn_test = pd.read_csv('../output/submission_NN_go_1195_dr_01_l2_0005_561_dr_01_l2_0005_0.797315952118905.csv')
    #  nn = pd.concat([nn_train, nn_test], axis=0).set_index(unique_id)
    #  data['nn_07973'] = nn['oof_preds']

    dummie=0
    val_col='valid_no_4'
    #  val_col='valid_no_2'
    #  val_col='valid_no'
    #  path = f'../features/{dir_num}_winner/*.npy'
    path = f'../features/rawgo/*.npy'
    path = f'../features/3_winner/*.npy'
    #  path = f'../features/CV08028/*.npy'
    #  if model_type=='xgb':
    #      path = f'../features/f_app/*.npy'
    #  path = '../features/select_target_feature_CV0.8036_LB0.804/*.npy'
    #  path = '../features/ec2/*.npy'

    make_submission(data, path=path, dummie=dummie, val_col=val_col)


if __name__ == '__main__':
    main()
