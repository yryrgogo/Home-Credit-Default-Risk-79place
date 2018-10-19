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
from utils import logger_func, get_categorical_features, get_numeric_features, pararell_process
from preprocessing import get_dummies
logger = logger_func()
pd.set_option('max_columns', 200)
pd.set_option('max_rows', 200)

#========================================================================
# Global Variable
from info_home_credit import hcdr_key_cols
key, target, ignore_list = hcdr_key_cols()
#========================================================================

fname = 'bureau'
app = utils.read_df_pkl(path='../input/clean_app*.p')[[key, target]]
df = utils.read_df_pkl(path=f'../input/clean_{fname}*')
df = df.merge(app, on=key, how='inner')
train = df[~df[target].isnull()]
test = df[df[target].isnull()]

def trans_to_dummies():
    global df

    cat_list = get_categorical_features(df=df, ignore_list=ignore_list)
    tmp_dummie_list = list(set(cat_list) - set(no_dummie_list))
    dummie_list = []
    for col in tmp_dummie_list:
        try:
            df[col] = df[col].astype('float64')
            logger.info(f'Remove: {col}')
            continue
        except ValueError:
            logger.info(f'Categorical: {col}')
            pass
        except TypeError:
            logger.info(f'Categorical: {col}')
            pass
        dummie_list.append(col)

    logger.info('# Get Dummies Start.')
    for col in dummie_list:
        logger.info(f'# {col} to Dummies...')
        df = get_dummies(df=df, cat_list=[col], drop=True)

def feature_check(col):
    global df

    if col in ignore_list:
        pass
    if col in no_dummie_list:
        pass
    if str(type(df[col].values[0])).count('time'):
        logger.info(f"TIme Column: {col}")
        pass
    try:
        df[col] = df[col].astype('float64')
        return False
    except ValueError:
        pass
    except TypeError:
        pass

    return True

#  trans_to_dummies()

def make_raw_feature(df, is_train):

    columns = df.columns
    for col in columns:
        value = df[col].values
        if str(type(value[0])).count('object') or str(type(value[0])).count('str'):continue

        logger.info(f'# {col} To Pickle & GZIP. | LENGTH: {len(value)}')
        col = col.replace('.', '_').replace('/', '_')
        if is_train:
            utils.to_pkl_gzip(obj=value, path=f'../features/4_winner/train_{col}')
        else:
            utils.to_pkl_gzip(obj=value, path=f'../features/4_winner/test_{col}')

make_raw_feature(df=train, is_train=True)
make_raw_feature(df=test, is_train=False)
