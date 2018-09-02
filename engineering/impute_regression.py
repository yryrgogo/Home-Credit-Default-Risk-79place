import gc
import numpy as np
import pandas as pd
import datetime
import glob
import sys
import re
import lightgbm as lgb
from sklearn.model_selection import train_test_split

sys.path.append('../model')
from lgbm_reg import prediction, cross_prediction

sys.path.append('../../../github/module/')
from preprocessing import set_validation, split_dataset, factorize_categoricals
from convinience_function import get_categorical_features, get_numeric_features
from make_file import make_feature_set
from logger import logger_func
from load_data import pararell_load_data, x_y_split

start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
logger = logger_func()


' データセットからそのまま使用する特徴量 '
unique_id = 'SK_ID_CURR'
target = 'TARGET'
val_col = 'valid_no_imp'
ignore_features = [unique_id, target, 'valid_no', 'valid_no_2', 'valid_no_3', 'valid_no_4', 'is_train', 'is_test']


def impute_regression(base, level, dataset, value, prefix=''):
    '''
    Explain:
    Args:
        base : 最後に特徴量をマージしてインデックスを揃える為のDF
        level: baseをマージする粒度
    Return:
    '''

    logger.info(f'\nimpute feature: {value}')

    ' 目的変数にマイナスが入っていた場合、分布の最小値が0となるように加算（対数変換のため） '
    values = dataset[value].values
    min_val = values[~np.isnan(values)].min()
    if min_val<0:
        dataset[value] = dataset[value].values + min_val*-1

    dataset[target] = dataset[target].map(lambda x:None if x==-1 else x)

    ' 目的のカラムにおいてNullとなっている行がTestで、値が入ってる行がTrain '
    ' is Noneの使い方がわからんので、これでNull判別 '
    dataset['is_train'] = dataset[value].map(lambda x: 1 if np.abs(x)>=0  else 0)

    ' カテゴリ変数があったらとりあえず整数値にしとく '
    categorical = get_categorical_features(dataset, [])
    dataset = factorize_categoricals(dataset, categorical)

    train = dataset.query('is_train==1')
    test = dataset.query('is_train==0')

    ' カラムにNullがなかったら抜ける '
    if len(train)==0 or len(test)==0:
        return

    #  train.drop(['is_train', 'is_test'], axis=1, inplace=True)
    #  test.drop(['is_train', 'is_test'], axis=1, inplace=True)
    train.drop(['is_train'], axis=1, inplace=True)
    test.drop(['is_train'], axis=1, inplace=True)

    train[target] = train[target].fillna(-1)

    ' ターゲットが全て-1のとき '
    if len(train[target].drop_duplicates())==1:
        train[f'bin10_{value}'] = pd.qcut(x=train[value], q=2, duplicates='drop')
        train = factorize_categoricals(train, [f'bin10_{value}'])
        validation = set_validation(train, target=f'bin10_{value}' , unique_id=unique_id, val_col=val_col)
        train.drop( f'bin10_{value}' , axis=1, inplace=True)
    else:
        validation = set_validation(train, target , unique_id=unique_id, val_col=val_col)

    train = train.merge(validation, on=unique_id, how='left')

    train[val_col] = train[val_col].fillna(-1)

    ' imputeするfeatureのノイズとなりそうなfeatureはデータセットから除く '
    for col in train.columns:
        #  if col==target or (col.count('impute') and not(col.count('EXT'))):
        if col==target:
            logger.info(f'extract feature: {col}')
            train.drop(col, axis=1, inplace=True)

    #  x, y = train_test_split(train, test_size=0.2)
    #  x['valid_no'] = 0
    #  y['valid_no'] = 1
    #  valid_no = 1
    #  train = pd.concat([x, y], axis=0)

    logger.info(f'train shape: {train.shape}')

    ' testに対する予測結果(array)が戻り値 '
    impute_value, cv_score = cross_prediction(
        logger=logger,
        train=train,
        test=test,
        target=value,
        #  categorical_feature=categorical,
        val_col=val_col
    )
    if cv_score<0.25:
        return 0
    ' 目的変数にマイナスが入っていた場合、対数変換の関係で行った前処理を元に戻す '
    if min_val<0:
        train[value] = train[value].values + min_val
        impute_value = impute_value + min_val

    ' データセットにJoinする際のインデックスソートを揃える為、unique_idをカラムに戻す '
    train.reset_index(inplace=True)
    test.reset_index(inplace=True)
    train = train[level+[value]]
    test = test[level+[value]]
    test[value] = impute_value
    result = pd.concat([train, test], axis=0)

    result = base.merge(result, on=level, how='left')
    print(result.shape)
    print(result.head())
    print(result.tail())

    #  result.set_index(unique_id, inplace=True)
    #  print(result.loc[check_id, :])
    #  print(result.query('is_test==1').head(10))
    #  sys.exit()

    np.save(f'../features/1_first_valid/{prefix}{value}_impute', result[value].values)

    return cv_score


def main():

    prefix = ''
    level = [unique_id]
    base = pd.read_csv('../data/base.csv')

    ' 学習に使うfeature_setをmerge '
    path = '../features/3_winner/*.npy'
    dataset = make_feature_set(base, path)
    dataset_columns = list(dataset.columns)
    #  dataset.set_index(level, inplace=True)

    #  logger.info(f'\nconcat end\ndataset shape: {dataset.shape}')

    ' imputeする連続値のカラムリスト '
    value_path_list = glob.glob('../features/*.csv')

    ' 特徴量を欠損値補完してnpyに保存する '
    score_list = []
    impute_list = []
    null_len_list = []

    ' 作成済featureが格納されたパス（同じfeatureは除くため） '
    path_list = glob.glob('../features/1_first_valid/*.npy')
    extract_list = []
    for path in path_list:
        if path.count('impute'):

            ' _imputeを除いたfeature_name '
            filename = re.search(r'/([^/.]*).npy', path).group(1)[:-7]
            extract_list.append(filename)

    for value_path in value_path_list:
        if value_path.count('npy'):
            value = re.search(r'/([^/.]*).npy', value_path).group(1)
        elif value_path.count('csv'):
            value = re.search(r'/([^/.]*).csv', value_path).group(1)

        ' 既に作成済のfeatureは作らない '
        if value in extract_list:
            logger.info(f'{value} is already exist.')
            continue

        ' データセットに含まれない特徴量は追加する '
        if value not in dataset_columns:
            if value_path.count('npy'):
                dataset[value] = np.load(value_path)
            elif value_path.count('csv'):
                base = pd.read_csv('../data/base.csv')
                tmp = pd.read_csv(value_path)
                dataset[value] = base[unique_id].to_frame().merge(tmp, on=unique_id, how='left')['TARGET']

        null_len = len(dataset[value]) - len(dataset[value].dropna())
        cv_score = impute_regression(base, level, dataset, value, prefix)

        if value not in dataset_columns:
            dataset.drop(value, axis=1, inplace=True)
        impute_list.append(value)
        score_list.append(cv_score)
        null_len_list.append(null_len)

        if len(impute_list)>2:
            result = pd.Series(impute_list, name='feature').to_frame()
            result['r2_score'] = score_list
            result['null_len'] = null_len_list
            result.to_csv(f'../output/{start_time[:11]}_impute_reg_feature_score.csv', index=False)


if __name__ == '__main__':
    main()
