import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
import datetime
from tqdm import tqdm
import sys
sys.path.append('../../../github/module')
from load_data import load_data, x_y_split
from preprocessing import set_validation, split_dataset

start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())


' データセットからそのまま使用する特徴量 '
unique_id = 'SK_ID_CURR'
target = 'TARGET'
ignore_features = [unique_id, target, 'valid_no', 'is_train', 'is_test']


"""Model Parameter"""
#  metric = 'logloss'
metric = 'auc'

early_stopping_rounds = 200
all_params = {
    'min_child_weight': [9, 13],
    'subsample': [0.7, 0.8],
    'bagging_freq': [1],
    'seed': [1208],
    'n_estimators': [100, 1000],
    'colsample_bytree': [0.7, 0.8],
    'silent': [True],
    'learning_rate': [0.1],
    'max_depth': [-1],
    'min_data_in_bin': [8, 10],
    'min_split_gain': [0],
    'clf_alpha': [1],
    'max_bin': [255, 511],
    'num_leaves': [31, 63],
    'objective': ['binary'],
}


def cross_validation(logger, train, test, target, categorical_feature):
    '''
    Explain:
        交差検証を行う.
        必要な場合はグリッドサーチでパラメータを探索する.
    Args:
    Return:
    '''

    list_score = []
    list_best_iterations = []
    best_params = None

    if metric == 'auc':
        best_score = 0
    elif metric == 'logloss':
        best_score = 100

    x_train, y_train = x_y_split(train, target)
    x_val, y_test = x_y_split(test, target)

    use_cols = list(x_train.columns)

    now = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

    for params in tqdm(list(ParameterGrid(all_params))):
        logger.info('params: {}'.format(params))

        lgb_train = lgb.Dataset(data=x_train, label=y_train)
        lgb_eval = lgb.Dataset(data=x_val, label=y_val)

        ' 学習 '
        clf = lgb.train(fix_params,
                        lgb_train,
                        valid_sets=lgb_eval,
                        early_stopping_rounds=150,
                        verbose_eval=200
                        )

        y_pred = clf.predict(x_val)
        sc_score = sc_metrics(y_test, y_pred)

        list_score.append(sc_score)
        list_best_iterations.append(clf.current_iteration())
        logger.info('{}: {}'.format(metric, sc_score))

        params['n_estimators'] = int(np.mean(list_best_iterations))
        sc_score = np.mean(list_score)
        if metric == 'logloss':
            if best_score > sc_score:
                best_score = sc_score
                best_params = params
        elif metric == 'auc':
            if best_score < sc_score:
                best_score = sc_score
                best_params = params

        logger.info('current {}: {}  best params: {}'.format(
            metric, best_score, best_params))

    logger.info('CV best score : {}'.format(best_score))
    logger.info('CV best params: {}'.format(best_params))

    # params output file
    df_params = pd.DataFrame(best_params, index=['params'])
    df_params.to_csv(f'../output/{start_time[:11]}_best_params_{metric}_{best_score}.csv', index=False)

    lgb_train = lgb.Dataset(data=x_train, label=y_train)
    lgb_eval = lgb.Dataset(data=x_val, label=y_val)

    clf = lgb.train(fix_params,
                    lgb_train,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=150,
                    verbose_eval=200
                    )

    y_pred = clf.predict(x_val)

    # feature importance output file
    feim_result = pd.Series(clf.feature_importance, name='importance')
    feature_name = pd.Series(use_cols, name='feature')
    features = pd.concat([feature_name, feim_result], axis=1)
    features.sort_values(by='importance', ascending=False, inplace=True)

    sc_score = sc_metrics(y_test, y_pred)
    list_score.append(sc_score)
    features.to_csv('../output/{}_feature_importances_{}_{}.csv'.format(
        start_time[:11], metric, sc_score), index=False)

    mean_score = np.mean(list_score)
    logger.info('CV & TEST mean {}: {}  best_params: {}'.format(
        metric, mean_score, best_params))


