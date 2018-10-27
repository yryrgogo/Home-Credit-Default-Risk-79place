import gc
import numpy as np
import pandas as pd
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


make_cat_features(df=app, filekey='1_first_valid')
