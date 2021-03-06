feat_no = '111_'
import numpy as np
import pandas as pd
import datetime
import sys
from tqdm import tqdm

import os
HOME = os.path.expanduser('~')
sys.path.append(f"{HOME}/kaggle/data_analysis/library/")
import utils
from utils import logger_func, get_categorical_features, get_numeric_features, pararell_process
logger = logger_func()
pd.set_option('max_columns', 200)
pd.set_option('max_rows', 200)
from feature_engineering import base_aggregation, diff_feature, division_feature, product_feature, cnt_encoding, select_category_value_agg, exclude_feature, target_encoding


start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

# ===========================================================================
# global variables
# ===========================================================================
key = 'SK_ID_CURR'
target = 'TARGET'
ignore_list = [key, target, 'SK_ID_BUREAU', 'SK_ID_PREV']

# ===========================================================================
# DATA LOAD
# ===========================================================================
base = utils.read_df_pkl(path='../input/base_app*')
fname = 'app'
prefix = f'{feat_no}{fname}_'
df = utils.read_df_pkl(path=f'../input/clean_{fname}*.p')

train = df[~df[target].isnull()]
test = df[df[target].isnull()]

neighbor = '110_app_neighbor81@'
train[neighbor] = utils.read_pkl_gzip('../input/train_110_app_neighbor81@.gz')
test[neighbor] = utils.read_pkl_gzip('../input/test_110_app_neighbor81@.gz')
combi = [neighbor, cat]
cat_list = get_categorical_features(df=df, ignore_list=ignore_list)

#========================================================================
# TARGET ENCODING
#========================================================================
for cat in cat_list:
    combi = cat
    feat_train, feat_test = target_encoding(logger=logger, train=train, test=test, key=key, level=combi, target=target, fold_type='stratified', group_col_name='', prefix='', ignore_list=ignore_list)

    col = f"'TE@{str(combi).replace('[', '').replace(' ', '').replace(',', '_')}"
    train_file_path = f"../features/1_first_valid/train_{prefix}{col}"
    test_file_path = f"../features/1_first_valid/test_{prefix}{col}"
    utils.to_pkl_gzip(obj=feat_train, path=train_file_path)
    utils.to_pkl_gzip(obj=feat_test, path=test_file_path)
