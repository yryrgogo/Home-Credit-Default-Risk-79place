feat_no = '108_'
prefix = feat_no
ext_feat=False
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
prefix = feat_no + f'{fname}_'
df = utils.read_df_pkl(path=f'../input/clean_{fname}*.p')


cat_list = get_categorical_features(df=df, ignore_list=ignore_list)

#========================================================================
# TARGET ENCODING
#========================================================================
for cat in cat_list:
    feat_train, feat_test = target_encoding(logger=logger, base=base, train=tmp_train, test=tmp_test, key=key, level=cat, target=target, fold_type='group', group_col_name=fvid, prefix=feature_group, ignore_list=ignore_list)


# Feature Save
for col in df.columns:
    if not(col.count('@')) or col in ignore_list:
        continue

    train_file_path = f"../features/1_first_valid/train_{prefix}{col}"
    test_file_path = f"../features/1_first_valid/test_{prefix}{col}"
    utils.to_pkl_gzip(obj=df[~df[target].isnull()][col].values, path=train_file_path)
    utils.to_pkl_gzip(obj=df[df[target].isnull()][col].values, path=test_file_path)

    logger.info(f'''
    #========================================================================
    # COMPLETE MAKE FEATURE : {train_file_path}
    #========================================================================''')
