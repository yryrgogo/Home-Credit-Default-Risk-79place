import numpy as np
import pandas as pd
import sys
import re
import gc
import glob
import os
HOME = os.path.expanduser('~')
sys.path.append(f"{HOME}/kaggle/data_analysis/library/")
import utils
from utils import logger_func, get_categorical_features, get_numeric_features, pararell_process, mkdir_func
from preprocessing import get_dummies
from feature_engineering import base_aggregation, diff_feature, division_feature, product_feature, cnt_encoding, select_category_value_agg, exclude_feature, target_encoding

logger = logger_func()
pd.set_option('max_columns', 200)
pd.set_option('max_rows', 200)

#========================================================================
# Global Variable
from info_home_credit import hcdr_key_cols
key, target, ignore_list = hcdr_key_cols()
#========================================================================

app = utils.read_df_pkl(path='../input/clean_app*.p')[[key, target]]

filekey='bureau'
filepath = f'../input/clean_{filekey}*.p'
df = utils.read_df_pkl(path=filepath)
df = df.merge(app, on=key, how='inner')

train = df[~df[target].isnull()]
test = df[df[target].isnull()]

categorical_features = get_categorical_features(df=train, ignore_list=ignore_list)

mkdir_func(f'../features/{filekey}')

#========================================================================
# Numeric Feature Save
#========================================================================
for col in train.columns:
    if col in categorical_features:continue

    utils.to_pkl_gzip(obj=train[col].values, path=f'../features/{filekey}/train_{col}')
    if col != target:
        utils.to_pkl_gzip(obj=test[col].values, path=f'../features/{filekey}/test_{col}')

#========================================================================
# Categorical Feature Encode
#========================================================================
# Factorize
logger.info("Factorize Start!!")
for col in categorical_features:
    for col in categorical_features:
        logger.info(train[col].value_counts().head())
        train[f"lbl_{col}@"], indexer = pd.factorize(train[col])
        test[f"lbl_{col}@"] = indexer.get_indexer(test[col])

# Count Encoding
logger.info("Count Encoding Start!!")
for col in categorical_features:
    train = cnt_encoding(train, col, ignore_list=ignore_list)
    test = cnt_encoding(test, col, ignore_list=ignore_list)

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
