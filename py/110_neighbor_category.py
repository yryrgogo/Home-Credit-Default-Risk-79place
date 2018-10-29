feat_no = '110_'
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
df = utils.read_df_pkl(path=f'../input/clean_{fname}*.p')[[key, target, 'EXT_SOURCE_2']]

train_ir = utils.read_pkl_gzip('../features/4_winner/train_stan_ir_mean@.gz')
test_ir = utils.read_pkl_gzip('../features/4_winner/test_stan_ir_mean@.gz')
ir_mean = np.hstack((train_ir, test_ir))
df['stan_ir_mean@'] = ir_mean
df['stan_ir_mean@'].fillna('ir_nan', inplace=True)

num_split = 9
df['EXT_bin'] = pd.qcut(x=df['EXT_SOURCE_2'], q=num_split)
df['ir_bin'] = pd.qcut(x=df['stan_ir_mean@'], q=num_split)

col = f'neighbor{num_split**2}@'
df[col] = df[['EXT_bin','ir_bin']].apply(lambda x: str(x[0]) + '_' + str(x[1]) if str(x[0])!=str(np.nan) else 'ext_nan', axis=1)

train_feat = df[df[target]>=0][col].values
test_feat = df[df[target].isnull()][col].values
train_file_path = f"../features/1_first_valid/train_{prefix}{col}"
test_file_path = f"../features/1_first_valid/test_{prefix}{col}"
utils.to_pkl_gzip(obj=train_feat, path=train_file_path)
utils.to_pkl_gzip(obj=test_feat, path=test_file_path)

logger.info(f'''
#========================================================================
# COMPLETE MAKE FEATURE : {train_file_path}
#========================================================================''')
