import numpy as np
import pandas as pd
import sys
import re
import gc
import glob

import os
HOME = os.path.expanduser('~')
sys.path.append(f"{HOME}/kaggle/github/library/")
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

app = utils.read_df_pkl(path='../input/application_train_test*.p')[[key, target]]

train_path = '../input/timediff_train*.p'
test_path = '../input/timediff_test*.p'
train = utils.read_df_pkl(path=train_path)
test = utils.read_df_pkl(path=test_path)

train = train.merge(app, on=key, how='inner')
test = test.merge(app, on=key, how='inner')

for col in train.columns:
    print(col)
