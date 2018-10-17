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
#  logger = logger_func()
pd.set_option('max_columns', 200)
pd.set_option('max_rows', 200)

key = 'SK_ID_CURR'
target = 'TARGET'
key_list = [key, target]
ignore_list = key_list + ['SK_ID_BUREAU', 'SK_ID_PREV']


def hcdr_key_cols():
    return key, target, ignore_list
