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
    early_stopping_rounds = int(sys.argv[5])
except IndexError:
    early_stopping_rounds = 150
num_iterations = 20000
try:
    experience_code = int(sys.argv[1])
except IndexError:
    experience_code = 0
try:
    seed = int(sys.argv[4])
except IndexError:
    seed = 1208

truncate_flg = 0
if experience_code>=2:
    truncate_flg = 1

iter_no = 1000
decrease_word = ''
decrease_path = '../output/use_feature/feature333_importance_auc0.79948108783529.csv'
rank_list = [450, 350, 300, 250]
val_col = 'valid_no_4'

import gc
import numpy as np
import pandas as pd
import datetime
from datetime import date, timedelta
import glob

import re
import shutil
from sklearn.metrics import log_loss, roc_auc_score
from itertools import combinations
from multiprocessing import Pool
import multiprocessing
import lightgbm as lgb
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from tqdm import tqdm

sys.path.append('../engineering')
from select_feature import move_to_second_valid

sys.path.append('../model')
from lgbm_clf import validation, prediction, cross_validation
from incremental_train import exploratory_train, incremental_train
from params_lgbm import train_params, valid_params, train_params_0729, train_params_0811, train_params_0815, xgb_params_0814, extra_params, lgr_params, train_params_dima

sys.path.append('../../../github/module/')
from preprocessing import set_validation, split_dataset, factorize_categoricals, get_dummies, data_regulize
from load_data import pararell_load_data, x_y_split
from convinience_function import get_categorical_features, get_numeric_features
from logger import logger_func
from make_file import make_feature_set, make_npy
from statistics_info import correlation


start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
logger = logger_func()

' validatoin_featureを組み合わせで試す場合はこちら。単一の場合は＜1 '
combi_no = 1

""" データセット関連 """

' データセットからそのまま使用する特徴量 '
unique_id = 'SK_ID_CURR'
target = 'TARGET'
ignore_features = [unique_id, target, 'valid_no', 'valid_no_2', 'valid_no_3', 'valid_no_4', 'is_train', 'is_test']

metric='auc'
if model_type=='lgb':
    train_params = train_params_0729()
    train_params = train_params_dima()
    #  train_params = train_params_0816()
    #  train_params = train_params_0815()
    train_params['learning_rate'] = learning_rate
    train_params['random_seed'] = seed
    train_params['bagging_seed'] = seed
    train_params['feature_fraction_seed'] = seed
    train_params['data_random_seed'] = seed
elif model_type=='xgb':
    train_params = xgb_params_0814()
    train_params['learning_rate'] = learning_rate
elif model_type=='ext':
    train_params = extra_params()
elif model_type=='lgr':
    train_params = lgr_params()


def incremental_decrease(base, path, decrease_path, decrease_word='', dummie=0, val_col='valid_no_4', iter_no=20):
    global iteration

    dataset = make_feature_set(base, path)
    dataset = dataset.set_index(unique_id)

    train = dataset.query('is_train==1')
    logger.info(f'train shape: {train.shape}')
    train.drop(['is_train', 'is_test'], axis=1, inplace=True)

    decrease_list = []

    df_feature = pd.read_csv(decrease_path)
    #  feature_arr_0 = df_feature.query('rank>100').query('rank<=150')['feature'].values
    #  feature_arr_1 = df_feature.query('rank>150').query('rank<=200')['feature'].values
    #  feature_arr_2 = df_feature.query('rank>200').query('rank<=300')['feature'].values
    #  feature_arr_3 = df_feature.query('rank>200')['feature'].values
    feature_arr_9 = df_feature.query('rank>50')['feature'].values
    feature_arr_9 = [col for col in feature_arr_9 if col.count('TARGET')]

    best_score = 0
    score_list = []
    decrease_set_list = []
    iter_no = iter_no
    raw_cols = list(train.columns)

    for i in range(iter_no):

        np.random.seed(np.random.randint(100000))
        #  decrease_list_0 = list(np.random.choice(feature_arr_3, 2))
        #  decrease_list = list(np.random.choice(feature_arr_9, 3)) + decrease_list_0
        decrease_list = list(np.random.choice(feature_arr_9, 4))

        #  target_flg = 0
        #  while target_flg == 0 and is_flg == 0:
        #  while target_flg <15:
        #      target_flg = 0
        #      np.random.seed(np.random.randint(100000))
            #  decrease_list_0 = list(np.random.choice(feature_arr_0, 1))
            #  decrease_list_0 = []
            #  decrease_list_1 = list(np.random.choice(feature_arr_1, 1))
            #  decrease_list_2 = list(np.random.choice(feature_arr_2, 2))
            #  decrease_list_3 = list(np.random.choice(feature_arr_2, 2))
            #  decrease_list_3 = []
            #  decrease_list = decrease_list_0 + decrease_list_1 + decrease_list_2 + decrease_list_3
            #  decrease_list = list(np.random.choice(feature_arr_9, 15))
            #  index_list = np.random.randint(0, len(feature_arr), size=12)

            #  for decrease in decrease_list:
            #      if decrease.count('TARGET'):
            #          target_flg += 1
                #  else:
                #      target_flg = 0
                #  if decrease.count('TARGET'):
                #      target_flg=1

        if len(decrease_word)>0:
            ' decrease_wordで指定した文字を含むfeatureのみ残す '
            decrease_list = [col for col in decrease_list if col.count(decrease_word)]

        use_cols = raw_cols.copy()

        error_list = []
        for decrease in decrease_list:
            try:
                use_cols.remove(decrease)
            except ValueError:
                error_list.append(decrease)
            logger.info(f'decrease feature: {decrease}')

        logger.info(f'\n** LIST REMOVE ERROR FEATURE: {error_list} **')

        tmp_result, col_length = get_cv_result(train=train[use_cols],
                                   target=target,
                                   val_col=val_col,
                                   logger=logger,
                                   params=train_params
                                   )

        if len(tmp_result)<=1:
            logger.info(f'\nLOW SCORE is truncate.')
            continue

        ' 追加したfeatureが分かるように文字列で入れとく '
        sc_score = tmp_result['cv_score'].values[0]
        score_list.append(sc_score)
        decrease_set_list.append(str(decrease_list))

        if sc_score > best_score:
            best_score = sc_score
            logger.info(f'\ndecrease: {str(decrease_list)}')
            logger.info(f'\nBest Score Update!!!!!')

            tmp_result['remove_feature'] = str(decrease_list)
            tmp_result.to_csv(f"../output/use_feature/feature{len(use_cols)}_rate{train_params['learning_rate']}_auc{sc_score}.csv", index=False)
        else:
            logger.info(f'\ndecrease: {decrease_list} \nNO UPDATE AUC: {sc_score}')

        logger.info(f'\n\n***** CURRENT BEST_SCORE/ AUC: {best_score} *****')

        if (i+1)%10 == 0 :
            result = pd.Series(data=score_list, index=decrease_set_list, name='score')
            result.sort_values(ascending=False, inplace=True)
            logger.info(f"\n*******Now Feature validation Result*******\n{result.head(10)}\n**************")
            result.to_csv(f'../output/{start_time[:12]}_decrease_feature_validation.csv')

        elif i+1==iter_no:
            result = pd.Series(data=score_list, index=decrease_set_list, name='score')
            result.sort_values(ascending=False, inplace=True)
            logger.info(f"\n*******Feature validation Result*******\n{result.head(20)}\n**************")


def incremental_increase(base, path, input_path, move_path, dummie=0, val_col='valid_no'):

    best_score, train, importance = first_train(base, path, dummie=0, val_col=val_col)

    ' 追加していくfeature '
    feature_path = glob.glob(input_path)
    ' 各学習の結果を格納するDF '
    result = pd.DataFrame([])
    df_idx = base['is_train']
    del base

    for number, path in enumerate(feature_path):

        ' npyを読み込み、file名をカラム名とする'
        feature_name = re.search(r'/([^/.]*).npy', path).group(1)
        feature = pd.Series(np.load(path), name=feature_name)

        ' 結合し学習データのみ取り出す '
        dataset = pd.concat([df_idx, feature], axis=1)
        train[feature_name] = dataset.query('is_train==1')[feature_name]
        del feature
        del dataset
        gc.collect()

        logger.info(f'\niteration no: {number}\nvalid feature: {feature_name}')

        tmp_result, col_length = get_cv_result(train=train, target=target, val_col=val_col, logger=logger)

        ' 追加したfeatureが分かるように文字列で入れとく '
        tmp_result['add_feature'] = feature_name

        sc_score = tmp_result['cv_score'].values[0]

        ' 前回のiterationよりスコアが落ちたら、追加した特徴量を落とす '
        if metric == 'auc':
            if sc_score <= best_score:
                train.drop(feature_name, axis=1, inplace=True)
                logger.info(f'\nExtract Feature: {feature_name}')
                shutil.move(path, move_path)

        if metric == 'auc':
            if best_score < sc_score:
                best_score = sc_score
                logger.info(f'\nBest Score Update!!!!!')
                logger.info(f'\nAdd Feature: {feature_name}')
                shutil.move(path, '../features/3_winner/')
                tmp_result.to_csv( f'../output/use_feature/feature_importance_auc{sc_score}.csv', index=False)

        elif metric == 'logloss':
            if best_score > sc_score:
                best_score = sc_score
        logger.info(f'\nCurrent best_score: {best_score}')


def much_feature_validation(base, path, move_path, dummie=0, val_col='valid_no'):

    feature_list = np.array(glob.glob('../features/1_first_valid/*.npy'))
    feature_list = np.sort(feature_list)
    if len(feature_list)>1000:
        feature_list = feature_list[:1000]

    for feature in feature_list:
        shutil.move(feature, '../features/3_winner/')

    logger.info(f'move feature:{len(feature_list)}')

    key_list = ['len1']
    for rank in rank_list:
        logger.info(f'rank:{rank}')
        _, _, importance = first_train(base, path, dummie=0, val_col=val_col)
        move_to_second_valid(best_select=importance, rank=rank, key_list=key_list)


def validation(base, path, path_list, move_path, dummie=0, val_col='valid_no'):

    best_score, train, importance = first_train(base, path, dummie=0, val_col=val_col)
    sys.exit()


def all_validation(base, path, path_list, move_path, dummie=0, val_col='valid_no'):

    best_score, train, importance = first_train(base, path, dummie=0, val_col=val_col)
    ' 各学習の結果を格納するDF '
    result = pd.DataFrame([])
    df_idx = base['is_train'].to_frame()
    del base

    for feature_dir_path in path_list:

        feature_path_list = glob.glob(feature_dir_path)

        if combi_no>1:
            feature_path_list = list(combinations(feature_path_list, r=combi_no))

        iter_no = len(feature_path_list)

        feature_list = []
        score_list = []
        rank_list = []

        for number, path in enumerate(feature_path_list):

            if combi_no>1:
                combi_columns = []
                for feature_path in path:
                    col = re.search(r'/([^/.]*).npy', feature_path).group(1)
                    df_idx[col]  = np.load(feature_path)
                    train[col] = df_idx.query('is_train==1')[col]
                    df_idx.drop(col, axis=1, inplace=True)
                    combi_columns.append(col)
                col = str(combi_columns)

            else:
                col = re.search(r'/([^/.]*).npy', path).group(1)
                df_idx[col]  = np.load(path)
                train[col] = df_idx.query('is_train==1')[col]
                df_idx.drop(col, axis=1, inplace=True)

            logger.info(f'\niteration no: {number}\nvalid feature: {col}')

            tmp_result, col_length = get_cv_result(train=train,
                                       target=target,
                                       val_col=val_col,
                                       logger=logger,
                                       params=train_params
                                       )

            ' 追加したfeatureが分かるように文字列で入れとく '
            tmp_result['add_feature'] = str(col)

            sc_score = tmp_result['cv_score'].values[0]
            feature_list.append(str(col))
            score_list.append(sc_score)
            if combi_no<=1:
                rank = tmp_result[tmp_result['feature']==col]['rank'].values[0]
                rank = f'{rank}/{len(tmp_result)}'
                rank_list.append(rank)

            if metric == 'auc':
                if best_score < sc_score:
                    best_score = sc_score
                    logger.info(f'\nBest Score Update!!!!!')
                    logger.info(f'\nAdd Feature: {col}')
                    tmp_result.to_csv(
                        f'../output/use_feature/feature_importance_auc{sc_score}.csv', index=False)
                else:
                    if combi_no>1:
                        print('combi feature no move')
                        #  for feature_path in path:
                            #  shutil.move(feature_path, move_path)
                    else:
                        shutil.move(path, move_path)

            elif metric == 'logloss':
                if best_score > sc_score:
                    best_score = sc_score

            if combi_no>1:
                train.drop(combi_columns, axis=1, inplace=True)
            else:
                train.drop(col, axis=1, inplace=True)

            logger.info(f'\nCurrent best_score: {best_score}')

            if iter_no == number+1:
                result = pd.Series(data=score_list, index=feature_list, name='score')
                result.sort_values(ascending=False, inplace=True)
                if combi_no<=1:
                    result.to_frame()['rank'] = rank_list
                logger.info(f"\n*******All Feature validation Fisnish!! Result*******\n{result.head(50)}\n**************")
                result.to_csv(f'../output/{start_time}all_feature_validation.csv')

            elif number%20==0:
                result = pd.Series(data=score_list, index=feature_list, name='score')
                result.sort_values(ascending=False, inplace=True)
                if combi_no<=1:
                    result['rank'] = rank_list
                logger.info(f"\n*******En route{iter_no} Feature validation Top10*******\n{result.head(10)}\n**************")


def parameter_tune(base, feature_set_path, all_params={}, dummie=0, val_col='valid_no'):

    best_score = first_train(base, feature_set_path, dummie=0, val_col='valid_no')

    for params in tqdm(list(ParameterGrid(all_params))):
        logger.info(f'params: {params}')

        tmp_result, col_length = get_cv_result(train=train, target=target, val_col=val_col, params=params, logger=logger)

        sc_score = tmp_result['cv_score'].values[0]
        score_list.append(sc_score)

        if metric == 'auc':
            if best_score < sc_score:
                best_score = sc_score
                logger.info(f'\nBest Score Update!!!!!')
                logger.info(f'\nBest Params: {params}')
                tmp_result.to_csv(f'../output/params/LGBM_params_auc{sc_score}.csv', index=False)

        elif metric == 'logloss':
            if best_score > sc_score:
                best_score = sc_score

        logger.info(f'\nCurrent Best_Score: {best_score}')


def scoring(params):
    global dataset
    if model_type=='lgb':
        params["max_bin"] = int(params["max_bin"])
        params["min_data_in_bin"] = int(params["min_data_in_bin"])
        params["min_child_samples"] = int(params["min_child_samples"])
        params["num_leaves"] = int(params["num_leaves"])
        params["min_child_weight"] = int(params["min_child_weight"])
    elif model_type=='xgb':
        params["min_child_weight"] = int(params["min_child_weight"])
        params["max_depth"] = int(params["max_depth"])
    elif model_type=='ext':
        params["max_leaf_nodes"] = int(params["max_leaf_nodes"])
        params["min_samples_leaf"] = int(params["min_samples_leaf"])
        params["min_samples_split"] = int(params["min_samples_split"])
    elif model_type=='lgr':
        params["max_iter"] = int(params["max_iter"])

    logger.info(f'dataset shape: {dataset.shape}')
    logger.info(f'optimize current params: {params}')
    result, col_length = get_cv_result(train=dataset,
                                       target=target,
                                       val_col=val_col,
                                       params=params,
                                       metric='auc',
                                       logger=logger,
                                       dummie=0
                                       )
    if len(result)<=1:
        return result[0] * -1
        #  except IndexError:
        #      return -0.8030

    score = result['cv_score'].values[0] * -1

    logger.info(f'optimize current score: {score}')

    return score


def optimize(base, path, metric):
    global dataset

    dataset = make_feature_set(base, path)

    dataset = dataset.set_index(unique_id)
    dataset = dataset.query('is_train==1')
    dataset.drop(['is_train', 'is_test'], axis=1, inplace=True)

    trials = Trials()

    if model_type=='lgb':

        search_params = {
            #  'boosting': 'dart',
            'num_threads': 35,
            'metric': 'auc',
            'objective': 'binary',
            'learning_rate': 0.02,
            #  'subsample': 0.9,
            'subsample': hp.quniform("subsample", 0.95, 1.0, 0.05),
            'num_leaves': hp.quniform("num_leaves", 9, 13, 1),
            #  'num_leaves': 14,
            'max_bin': hp.quniform("max_bin", 200, 500, 50),
            'max_depth': 5,
            'min_child_samples': hp.quniform("min_child_samples", 10, 100, 4),
            'min_child_weight': hp.quniform("min_child_weight", 10, 100, 4),
            #  'min_child_weight': 18,
            'min_data_in_bin': hp.quniform("min_data_in_bin", 10, 100, 4),
            'min_split_gain': 0.01,
            'bagging_freq': 1,
            'colsample_bytree': 0.01,
            #  'colsample_bytree': hp.quniform("colsample_bytree", 0.1, 0.14, 0.01),
            #  'lambda_l1': hp.quniform('lambda_l1', 0.1, 2.0, 0.1),
            #  'sigmoid': hp.quniform("sigmoid", 0.8, 1.2, 0.1),
            'random_seed': seed,
            'bagging_seed':seed,
            'feature_fraction_seed':seed,
            'data_random_seed':seed,
            #  'random_seed': 605,
            #  'bagging_seed':605,
            #  'feature_fraction_seed':605,
            #  'data_random_seed':605,
            'lambda_l1': 0.1,
            'lambda_l2': hp.quniform("lambda_l2", 28.0, 58.0, 6.0),
        }

    elif model_type=='xgb':
        search_params = {
            'objective': "binary:logistic",
            'booster': "gbtree",
            'eval_metric': 'auc',
            'eta': 0.02,
            'max_depth': hp.quniform("max_depth", 4, 5, 1),
            #  'max_depth': 5,
            'gamma': hp.quniform("gamma", 0.01, 0.51, 0.05),
            'min_child_weight': hp.quniform("min_child_weight", 10, 30, 2),
            'subsample': 0.9,
            #  'colsample_bytree': hp.quniform("colsample_bytree", 0.2, 0.85, 0.05),
            #  'colsample_bytree': hp.quniform("colsample_bytree", 0.01, 0.04, 0.01),
            'colsample_bytree': 0.01,
            #  'alpha': hp.quniform("alpha", 0.1, 0.5, 0.1),
            #  'lambda': hp.quniform("lambda", 70, 90, 10),
            'alpha': 0.1,
            'lambda': 70,
            #  'lambda': 3,
            #  'alpha': 5,
            'seed': 1208
        }

    elif model_type=='ext':
        search_params = {
            'criterion': 'gini',
            'max_depth': None,
            'max_features': hp.quniform("max_features", 0.2, 0.4, 0.1),
            'max_leaf_nodes': hp.quniform("max_leaf_nodes", 10, 400, 30), # 10~5000
            'min_impurity_decrease': hp.quniform("min_impurity_decrease", 0.1, 1, 0.1),
            'min_samples_leaf': hp.quniform("min_samples_leaf", 5, 35, 3),
            'min_samples_split': hp.quniform("min_samples_split", 10, 50, 5),
            'min_weight_fraction_leaf': hp.quniform("min_weight_fraction_leaf", 0.01, 0.33, 0.04),
            'n_jobs': -1,
            'random_state': 1208
        }
        dataset = data_regulize(dataset, sc_flg=0, float16_flg=1, ignore_feature_list=ignore_features)
        print(dataset[dataset.isnull()])
        sys.exit()

    elif model_type=='lgr':
        search_params = {
            'penalty': 'l2',
            'C': hp.quniform("C", 0.1, 0.9, 0.1),
            'max_iter': hp.quniform("max_iter", 100, 1000, 100),
            'n_jobs': -1,
            'random_state': 1208
        }
        dataset = data_regulize(dataset, sc_flg=1, float16_flg=1, ignore_feature_list=ignore_features)

    best = fmin(scoring, search_params, algo=tpe.suggest, trials=trials, max_evals=250)
    logger.info(f"best parameters: {best}")


def first_train(dataset, path, dummie=0, val_col='valid_no'):

    dataset = make_feature_set(dataset, path)
    dataset = dataset.set_index(unique_id)

    #  for col in dataset.columns:
    #      dataset[col] = dataset.replace(dataset>0, 4)
    #  print(dataset.head())
    #  sys.exit()

    ' テストデータ行にデータが入ってるか？ '
    #  for col in dataset.columns:
    #      if col.count('b_') and col.count('bin'):
    #          print(col)
    #          print(dataset[col].query('TARGET==-1'))
    #  sys.exit()

    if experience_code!=4:
        train = dataset.query('is_train==1')
        logger.info(f'train shape: {train.shape}')
        train.drop(['is_train', 'is_test'], axis=1, inplace=True)
    elif experience_code==4:
        dataset = dataset.reset_index()
        train = dataset
        train.drop(['is_train', 'is_test'], axis=1, inplace=True)
        ' Trainセットダウンロード用 '
        train.drop(['valid_no', 'valid_no_2', 'valid_no_3'], axis=1, inplace=True)
        logger.info(f'train shape: {train.shape}')
        #  train.to_csv(f'../output/{start_time[:11]}_home_credit_train_{len(train.columns)}features_Go_best_model_CV08028.csv')
        train.to_csv(f'../output/{start_time[:11]}_home_credit_train_{len(train.columns)}features_Go_best_model_CV08036_regularize_for_NN.csv')
        sys.exit()

    #  logger.info(f'First Feature: {train.columns.values}')

    first, col_length = get_cv_result(train=train, target=target, val_col=val_col, params=train_params, logger=logger, dummie=0)

    ' 最初のスコア '
    first_score = first['cv_score'].values[0]
    logger.info( f'\nFirst Score: {first_score}')
    first.to_csv( f'../output/use_feature/feature{col_length}_importance_auc{first_score}.csv', index=False)
    importance = first[['feature', 'avg_importance', 'rank']].sort_values(by='avg_importance')

    return first_score, train, importance


def get_cv_result(train, target, val_col, params, metric='auc', logger=False, dummie=0):

    #  if len(train.columns)>=1000:
    #      params['learning_rate'] = 0.1

    ' もしカテゴリカルなカラムが残っていたら、ラベル or HOTエンコーディング '
    categorical = get_categorical_features(train, [])
    if dummie == 0:
        train = factorize_categoricals(train, categorical)
        logger.info(f'LABEL REMAING CATEGORICAL: {categorical}')
    elif dummie == 1:
        train = get_dummies(train, categorical)
        logger.info(f'OHE REMAING CATEGORICAL: {categorical}')

    result, col_length = cross_validation(
        logger=logger,
        dataset=train,
        target=target,
        val_col=val_col,
        params=params,
        metric=metric,
        truncate_flg=truncate_flg,
        num_iterations=num_iterations,
        learning_rate=learning_rate,
        early_stopping_rounds=early_stopping_rounds,
        model_type=model_type
    )

    return result, col_length


def main():
    global iteration

    experience_list = [
        ''
        ,'much'
        ,'decrease'
        ,'params'
    ]

    path = '../features/3_winner/*.npy'
    base = pd.read_csv('../data/base.csv')

    ' 特徴量セットの確認 '
    #  path = '../features/1_first_valid/*.npy'
    #  dataset = make_feature_set(base, path)

    ' 特徴量の組み合わせでnanがどれだけあるかを見る '
    #  dataset = dataset.set_index(unique_id)
    #  dataset.fillna(0, inplace=True)
    #  dataset = dataset.where(dataset==0, 1)
    #  columns = [col for col in dataset.columns if col.count('@')]
    #  combi_list = combinations(columns, 10)
    #  best = 1000000
    #  cnt = 0
    #  for combi in tqdm(combi_list):
    #      combi = list(combi)
    #      tmp = dataset[combi].sum(axis=1)
    #      tmp_result = tmp[tmp==0]
    #      nan_cnt = len(tmp_result)
    #      if nan_cnt<best:
    #          best = nan_cnt
    #          best_combi = combi
    #          logger.info(best)
    #          cnt=0
    #      else:
    #          cnt+=1
    #          if cnt>10000:
    #              break
    #  logger.info(best)
    #  logger.info(best_combi)
    #  sys.exit()
    path_list = glob.glob('../features/1_first_valid/*.npy')

    ' 相関を見る '
    #  corr = correlation(dataset)
    #  corr.to_csv('../output/correlation.csv')
    #  sys.exit()

    if experience_list[experience_code]=='much':
        ' 多量のfeatureを繰り返し検証してスコアリング '
        much_feature_validation(base, path='../features/1_first_valid/', move_path='../features/1_second_valid/', dummie=0, val_col=val_col)

    if experience_list[experience_code]=='':
        ' 特徴セットをそのまま検証する '
        validation(base, path=path, path_list=path_list, move_path='../features/1_second_valid/', dummie=0, val_col=val_col)
        sys.exit()

    # 特徴量の組み合わせを乱数処理 or STEP WISE で検証する
    if experience_list[experience_code]=='increase':
        ' STEP WISE INCREASE ONE '
        incremental_increase(base, input_path='../features/1_first_valid/*.npy', move_path='../features/1_second_valid/', dummie=0)

    if experience_list[experience_code]=='decrease':
        ' STEP WISE DECREASE '
        incremental_decrease(base, path=path, decrease_path=decrease_path, decrease_word=decrease_word, dummie=0, val_col=val_col, iter_no=iter_no)

    if experience_list[experience_code]=='params':
        ' パラメータのベイズ最適化 '
        optimize(base=base, path=path, metric=metric)
        sys.exit()


if __name__ == '__main__':

    main()
