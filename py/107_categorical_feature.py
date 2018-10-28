prefix = '107_'
import gc
import numpy as np
import pandas as pd
from itertools import combinations
import sys
import re
from glob import glob
import os
HOME = os.path.expanduser('~')
sys.path.append(f"{HOME}/kaggle/data_analysis/library/")
#  sys.path.append(f"/mnt/c/Git/go/kaggle/github/library/")
import utils
from utils import logger_func, mkdir_func
logger = logger_func()
import eda
from utils import get_categorical_features, get_numeric_features, pararell_process
from feature_engineering import base_aggregation, diff_feature, division_feature, product_feature, cnt_encoding, select_category_value_agg, exclude_feature, target_encoding
from tqdm import tqdm

#========================================================================
# Global Variable
from info_home_credit import hcdr_key_cols
key, target, ignore_list = hcdr_key_cols()
#========================================================================

app = utils.read_df_pkl(path='../input/clean_app*.p')

def make_cat_features(df, filekey):
    mkdir_func(f'../features/{filekey}')
    train = df[~df[target].isnull()]
    test = df[df[target].isnull()]
    categorical_features = get_categorical_features(df=train, ignore_list=ignore_list)

    #========================================================================
    # Categorical Feature Encode
    #========================================================================
    # Factorize
    logger.info("Factorize Start!!")
    for col in categorical_features:
        for col in categorical_features:
            train[f"lbl_{col}@"], indexer = pd.factorize(train[col])
            test[f"lbl_{col}@"] = indexer.get_indexer(test[col])

    # Count Encoding
    #  logger.info("Count Encoding Start!!")
    #  for col in categorical_features:
    #      train = cnt_encoding(train, col, ignore_list=ignore_list)
    #      test = cnt_encoding(test, col, ignore_list=ignore_list)

    #========================================================================
    # Categorical Feature Save
    #========================================================================
    for col in train.columns:
        logger.info("Saving Features...")
        if col.count('@'):
            result_train = train[col].values
            result_test = test[col].values
            logger.info(f"COL: {col} | LENGTH: {len(result_train)}")
            utils.to_pkl_gzip(obj=result_train, path=f'../features/{filekey}/train_{col}')
            utils.to_pkl_gzip(obj=result_test, path=f'../features/{filekey}/test_{col}')

method_list = ['sum', 'mean', 'var', 'max', 'min']


def one_level_agg(df, prefix):
    # =======================================================================
    # 集計するカラムリストを用意
    # =======================================================================
    method_list = ['mean', 'var']
    num_list = ['EXT_SOURCE_2']
    cat_list = get_categorical_features(df=df, ignore_list=ignore_list)
    #  amt_list = [col for col in num_list if col.count('AMT_')]
    #  days_list = [col for col in num_list if col.count('DAYS_')]

    # 直列処理
    for cat in cat_list:
        if len(df[cat].unique())<=3:
            continue
        for num in num_list:
            for method in method_list:
                base = df[[key, cat, target]].drop_duplicates()
                tmp = df[[cat, num]]
                tmp_result = base_aggregation(
                    df=tmp, level=cat, method=method, prefix=prefix, feature=num)
                result = base.merge(tmp_result, on=cat, how='left')

                for col in result.columns:
                    if not(col.count('@')) or col in ignore_list:
                        continue

                    train_file_path = f"../features/1_first_valid/train_{col}"
                    test_file_path = f"../features/1_first_valid/test_{col}"

                    utils.to_pkl_gzip(obj=result[result[target]>=0][col].values, path=train_file_path)
                    utils.to_pkl_gzip(obj=result[result[target].isnull()][col].values, path=test_file_path)

                    logger.info(f'''
                    #========================================================================
                    # COMPLETE MAKE FEATURE : {train_file_path}
                    #========================================================================''')
                del result, tmp_result
                gc.collect()


def two_level_agg(df, prefix):
    # =======================================================================
    # 集計するカラムリストを用意
    # =======================================================================
    method_list = ['mean']
    num_list = ['EXT_SOURCE_2']
    cat_list = get_categorical_features(df=df, ignore_list=ignore_list)
    cat_combi = combinations(cat_list, 2)
    #  amt_list = [col for col in num_list if col.count('AMT_')]
    #  days_list = [col for col in num_list if col.count('DAYS_')]

    # 直列処理
    for com in cat_combi:
        for num in num_list:
            for method in method_list:
                base = df[[key, target] + list(com)].drop_duplicates()
                tmp = df[list(com)+[num]]
                tmp_result = base_aggregation(
                    df=tmp, level=list(com), method=method, prefix=prefix, feature=num)
                result = base.merge(tmp_result, on=list(com), how='left')

                for col in result.columns:
                    if not(col.count('@')) or col in ignore_list:
                        continue

                    train_feat = result[result[target]>=0][col].values
                    test_feat = result[result[target].isnull()][col].values
                    col = col.replace('[', '_').replace(']', '_').replace(' ', '').replace(',', '_')
                    train_file_path = f"../features/1_first_valid/train_{col}"
                    test_file_path = f"../features/1_first_valid/test_{col}"

                    utils.to_pkl_gzip(obj=train_feat, path=train_file_path)
                    utils.to_pkl_gzip(obj=test_feat, path=test_file_path)

                    logger.info(f'''
                    #========================================================================
                    # COMPLETE MAKE FEATURE : {train_file_path}
                    #========================================================================''')
                del result, tmp_result
                gc.collect()


#  make_cat_features(df=app, filekey='1_first_valid')

two_level_agg(app, prefix)
