import numpy as np
import pandas as pd
import datetime
import glob
import sys
import re
from multiprocessing import Pool
import multiprocessing
from itertools import combinations
from select_feature import select_feature

sys.path.append('../../../github/module/')
from preprocessing import set_validation, split_dataset, get_categorical_features, get_numeric_features, dframe_dtype, get_dummies, factorize_categoricals, contraction, outlier
from load_data import pararell_load_data
from logger import logger_func
from convinience import list_check
from feature_engineering import make_npy, base_aggregation


logger = logger_func()
start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

unique_id = 'SK_ID_CURR'
target = 'TARGET'
ignore_features = [unique_id, target, 'valid_no', 'is_train', 'is_test', 'SK_ID_BUREAU', 'SK_ID_PREV']

pd.set_option('display.max_rows', 300)


def feature_contraction(data):

    limit = 74625

    columns = [col for col in data.columns if col.count('DAYS')]
    columns = ['SELLERPLACE_AREA']

    for col in columns:
        #  data = contraction(data, col, limit, 1, 1)
        #  data = contraction(data, col, limit, 0, 1)
        data = contraction(data, col, limit, 1, 0)

        print(f'columns: {col}')
        #  print(data.query(f"{col} > {limit}").count())
        #  print(data.query(f"{col} < {limit}").count())

    data.to_csv('../data/previous_cleansing.csv', index=False)
    sys.exit()


def extract_outlier(data):

    num_list = get_numeric_features(data, ignore_features)
    for num in num_list:
        #  if num.count('AMT_') or num.count('DAYS'):
        #  if num == 'CNT_DRAWINGS_OTHER_CURRENT':continue
        print(f'\ncolumn name: {num}\n')
        data = outlier(data, [], num, 1.96, 0)

    data.to_csv('../data/previous_cleansing.csv', index=False)


def cats_replace(data):

    categorical = get_categorical_features(data, ignore_features)

    for cat in categorical:
        data[cat] = data[cat].map(lambda x: None if x=='XNA' or x=='XAP' else x)
        print(data[cat].drop_duplicates())

    data.to_csv('../data/previous_application.csv', index=False)
    sys.exit()


def set_nan(data, feature_list, value):
    '''
    Explain:
        feature_listに入った各カラムについて、valueで受け取った値をnp.nanに置換する
    Args:
    Return:
    '''
    for feature in feature_list:
        data[feature] = data[feature].map(lambda x:np.nan if x==value else x)
    return data


def trans_month(data):

    days_list = [col for col in data.columns if col.count('_DAY_') or col.count('DAYS')]

    for col in days_list:
        data[col] = data[col]/30
        data[col].fillna(99999, inplace=True)
        data[col] = data[col].astype('int')
        data[col] = data[col].map(lambda x:np.nan if x==99999 else x)

    return data


def main():

    " データの読み込み "
    ' bureauのロード '
    data = pd.read_csv('../data/previous_cleansing.csv')
    #  data = pd.read_csv('../data/rawdata/previous_application.csv')
    #  train = pd.read_csv('../data/application_train_test.csv')

    ' 大きな外れ値を値を指定して収縮する '
    #  feature_contraction(data)
    ' DAYS変数を日でなく月に直す '
    #  data = trans_month(data)
    ' NaNが0埋めされているのをNaNにする '
    #  nan_list = ['AMT_APPLICATION', 'AMT_CREDIT', 'AMT_DOWN_PAYMENT', 'AMT_GOODS_PRICE', 'RATE_DOWN_PAYMENT']
    #  nan_list = ['SELLERPLACE_AREA']
    #  data = set_nan(data, nan_list, -1)

    #  data.to_csv('../data/previous_cleansing.csv', index=False)
    #  sys.exit()

    ' カラムの値ごとに行数カウント '
    #  for col in data.columns:
    #      if col in ignore_features:continue
    #      print(f'columns: {col}')
    #      print(data[col].value_counts().sort_index())
    #  sys.exit()

    ' XNAなどでNULL埋めされてる変数をNULLに戻す '
    #  cats_replace(data)

    extract_outlier(data)


if __name__ == '__main__':

    main()
