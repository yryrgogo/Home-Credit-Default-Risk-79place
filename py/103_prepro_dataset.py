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

'''
#========================================================================
# BUREAU
CREDIT_ACTIVE
Closed      1079273
Active       630607
Sold           6527
Bad debt         21
Name: CREDIT_ACTIVE, dtype: int64
CREDIT_CURRENCY
currency 1    1715020
currency 2       1224
currency 3        174
currency 4         10
Name: CREDIT_CURRENCY, dtype: int64
CREDIT_TYPE
Consumer credit                                 1251615
Credit card                                      402195
Car loan                                          27690
Mortgage                                          18391
Microloan                                         12413
Loan for business development                      1975
Another type of loan                               1017
Unknown type of loan                                555
Loan for working capital replenishment              469
Cash loan (non-earmarked)                            56
Real estate loan                                     27
Loan for the purchase of equipment                   19
Loan for purchase of shares (margin lending)          4
Mobile operator loan                                  1
Interbank credit                                      1
Name: CREDIT_TYPE, dtype: int64
#======================================================================== '''
bur = utils.read_df_pkl(path='../input/clean_bur*.p')

cat_list = get_categorical_features(df=bur, ignore_list=[])

for cat in cat_list:
    print(cat)
    print(bur[cat].value_counts())


sys.exit()
pre = utils.read_df_pkl(path='../input/clean_prev*.p')
pos = utils.read_df_pkl(path='../input/clean_pos*.p')
ins = utils.read_df_pkl(path='../input/clean_ins*.p')
ccb = utils.read_df_pkl(path='../input/clean_ccb*.p')

utils.end(sys.argv[0])

#  pre_eda = eda.df_info(pre)
