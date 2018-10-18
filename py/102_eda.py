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
from utils import logger_func
logger = logger_func()
import eda
from utils import get_categorical_features

key = 'SK_ID_CURR'
target = 'TARGET'

utils.start(sys.argv[0])

fname_list = [
    'app'
    ,'bur'
    ,'pre'
    ,'pos'
    ,'ins'
    ,'ccb'
]

result = pd.DataFrame()
for fname in fname_list:
    df = utils.read_df_pkl(path=f'../input/clean_{fname}*.p')
    logger.info(f'EDA of {fname} Start!!')
    df_eda = eda.df_info(df)
    logger.info(f'EDA of {fname} End!!')
    df_eda['fname'] = fname

    if len(result):
        result = pd.concat([result, df_eda], axis=0)
    else:
        result = df_eda.copy()

result.to_csv(f'../eda/1018_home_credit_all_eda.csv', index=True)

utils.end(sys.argv[0])
