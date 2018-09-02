import numpy as np
import pandas as pd
import datetime
import glob
import sys
import re
from itertools import combinations
from multiprocessing import Pool
import multiprocessing
from categorical_encoding import make_win_set, cat_to_target_bin_enc
from target_encoding import target_encoding
from tqdm import tqdm

sys.path.append('../../../github/module/')
from load_data import pararell_load_data
from feature_engineering import diff_feature, division_feature, product_feature, cat_to_target_bin_enc
from convinience_function import get_categorical_features, get_numeric_features, row_number
from make_file import make_feature_set, make_npy
from logger import logger_func

logger = logger_func()
start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

unique_id = 'SK_ID_CURR'
target = 'TARGET'
ignore_features = [unique_id, target, 'valid_no', 'valid_no_1', 'valid_no_2', 'valid_no_3', 'valid_no_4', 'is_train', 'is_test', 'SK_ID_PREV', 'SK_ID_BUREAU']


def num_cat_encoding(df, bins=0):

    if bins>0:

        bin_list = get_numeric_features(data=df, ignore=ignore_features)

        logger.info(df.shape)
        for col in bin_list:
            df[col] = df[col].replace(np.inf, np.nan)
            df[col] = df[col].replace(-1*np.inf, np.nan)
            df[col] = df[col].fillna(df[col].median())
            length = len(df[col].drop_duplicates())
            #  print(df[col].drop_duplicates())
            #  continue
            #  sys.exit()
            if length<bins:
                continue
            df[f'bin{bins}_{col}'] = pd.qcut(x=df[col], q=bins, duplicates='drop')
            df.drop(col, axis=1, inplace=True)
            #  df.rename(columns={col:f'bin{bin}_{col}'}, inplace=True)

    app = pd.read_csv('../data/application_summary_set.csv')
    #  print(app.columns)
    #  print(app['bin10_a_ORGANIZATION_TYPE'].drop_duplicates())
    #  sys.exit()

    label_list = ['a_REGION_RATING_CLIENT_W_CITY', 'a_HOUSE_HOLD_CODE@']
    cat_list = get_categorical_features(data=app, ignore=ignore_features) + label_list
    cat_list = [col for col in cat_list if not(col.count('bin')) or (col.count('TION_TYPE'))]
    #  cat_list = ['a_HOUSE_HOLD_CODE@']
    #  cat_list = [col for col in cat_list if not(col.count('FLAG')) and not(col.count('GEND'))]
    bin_list = [col for col in df.columns if col.count('bin')]
    #  bin_list = [col for col in df.columns if (col.count('bin20') or col.count('bin10') )]
    df = df.merge(app, on=unique_id, how='inner')

    categorical_list = []
    for cat in cat_list:
        for num in bin_list:
            #  encode_list = [cat, elem_3, elem, elem_2]
            encode_list = [cat, num, 'a_CODE_GENDER']

            length = len(df[encode_list].drop_duplicates())
            cnt_id = len(df[unique_id].drop_duplicates())
            if length>100 or length<60 or cnt_id/length<3000:
                continue
            categorical_list.append(encode_list)

    method_list = ['mean', 'std']
    select_list = []
    val_col = 'valid_no_4'

    base = pd.read_csv('../data/base.csv')
    for cat in tqdm(categorical_list):
        length = len(df[cat].drop_duplicates())
        prefix = f'new_len{length}_'
        #  prefix = f'abp_vc{length}_'
        target_encoding(base=base, data=df, unique_id=unique_id, level=cat, method_list=method_list,
                        prefix=prefix, select_list=select_list, test=1, impute=1208, val_col=val_col, npy_key=target)


def main():
    '''
    集計粒度であるカテゴリカラムをfeature_ext_sourceにわたし、
    そのカテゴリ粒度をext_sourceでターゲットエンコーディングする
    '''

    base = pd.read_csv('../data/base.csv')
    data = make_feature_set(base[unique_id].to_frame(), '../features/dima/*.npy')

    #  bins = int(sys.argv[1])
    bins = 20
    bins_list = [10, 15, 20]
    for bins in bins_list:
        num_cat_encoding(data, bins)


if __name__ == '__main__':
    main()
