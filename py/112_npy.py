prefix = '112_'
" Interest_Rate "
import gc
import numpy as np
import pandas as pd
import sys
import re
import glob
import os
HOME = os.path.expanduser('~')
sys.path.append(f"{HOME}/kaggle/github/library/")
#  sys.path.append(f"/mnt/c/Git/go/kaggle/github/library/")
import utils
from utils import logger_func, mkdir_func
logger = logger_func()

#========================================================================
# Global Variable
from info_home_credit import hcdr_key_cols
key, target, ignore_list = hcdr_key_cols()
#========================================================================

# npy
npy_path_list = glob.glob('../features/4_winner/*.npy')
if len(npy_path_list)>0:
    npy_list = utils.pararell_load_data(path_list=npy_path_list)
    df_npy = pd.concat(npy_list, axis=1)

base = utils.read_df_pkl('../input/base_app*')
if len(npy_path_list)>0:
    base = pd.concat([base, df_npy], axis=1)
    base_train = base[~base[target].isnull()].reset_index(drop=True)
    base_test = base[base[target].isnull()].reset_index(drop=True)

for col in base_train.columns:
    if col not in ignore_list and col.count('@'):
        train_file_path = f"../features/1_first_valid/train_{prefix}{col}"
        test_file_path = f"../features/1_first_valid/test_{prefix}{col}"
        utils.to_pkl_gzip(obj=base_train[col].values, path=train_file_path)
        utils.to_pkl_gzip(obj=base_test[col].values, path=test_file_path)
