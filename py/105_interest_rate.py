" Interest_Rate "
import gc
import numpy as np
import pandas as pd
import sys
import re
from glob import glob
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

prev_key = 'SK_ID_PREV'
acr = 'AMT_CREDIT'
aan = 'AMT_ANNUITY'
adp = 'AMT_DOWN_PAYMENT'
cpy = 'CNT_PAYMENT'
co_type = 'NAME_CONTRACT_TYPE'
dd = 'DAYS_DECISION'

prev_ir = False
if prev_ir:
    app = utils.read_df_pkl('../input/clean_app*')[[key, target]]
    df = utils.read_df_pkl('../input/clean_prev*')
    df = df[[key, prev_key, dd, acr, aan, cpy, adp, co_type]].merge(app, on=key, how='inner')
    df = df[~df[cpy].isnull()]

    # 金利合計（分割数考慮前）
    tmp_ir = (df[aan].values * df[cpy].values) / df[acr].values

    for cnt in range(3, 91, 3):
        ir = ( tmp_ir * (cnt / df[cpy]) ) - 1.0
        df[f'ir_{cnt}'] = ir
        df[f'ir_{cnt}'] = df[f'ir_{cnt}'].map(lambda x: x if (0.08<x) and (x<0.25) else np.nan)
        print(f"{cnt} :", len(df[f'ir_{cnt}'].dropna()))
        if len(df[f'ir_{cnt}'].dropna())<len(df)*0.001:
            df.drop(f'ir_{cnt}', axis=1, inplace=True)

    ir_cols = [col for col in df.columns if col.count('ir_')]
    df['ir_max'] = df[ir_cols].max(axis=1)
    df['ir_min'] = df[ir_cols].min(axis=1)
    df['ir_std'] = df[ir_cols].std(axis=1)
    utils.to_df_pkl(df=df, path='../eda/', fname='1024_prev_ir')


df = utils.read_df_pkl('../input/clean_cpy*')


# CNT_PAYMENT
col = cpy
train_file_path = f"../features/1_first_valid/train_{col}"
test_file_path = f"../features/1_first_valid/test_{col}"

utils.to_pkl_gzip(obj=df[~df[target].isnull()][cpy].values, path=train_file_path)
utils.to_pkl_gzip(obj=df[df[target].isnull()][cpy].values, path=test_file_path)
sys.exit()

# 金利合計（分割数考慮前）
tmp_ir = (df[aan].values * df[cpy].values) / df[acr].values

for cnt in range(3, 61, 3):
    ir = ( tmp_ir * (cnt / df[cpy]) ) - 1.0
    df[f'ir_{cnt}@'] = ir
    df[f'ir_{cnt}@'] = df[f'ir_{cnt}@'].map(lambda x: x if (0.08<x) and (x<0.25) else np.nan)
    print(f"{cnt} :", len(df[f'ir_{cnt}@'].dropna()))
    if len(df[f'ir_{cnt}@'].dropna())<len(df)*0.001:
        df.drop(f'ir_{cnt}@', axis=1, inplace=True)

ir_cols = [col for col in df.columns if col.count('ir_') or col.count('CNT_PAY')]

for col in ir_cols:
    if not(col.count('@')) or col in ignore_list:
        continue
    train_feat = df[df[target]>=0][col].values
    test_feat = df[df[target].isnull()][col].values
    col = col.replace('[', '_').replace(']', '_').replace(' ', '').replace(',', '_')
    train_file_path = f"../features/1_first_valid/train_{col}"
    test_file_path = f"../features/1_first_valid/test_{col}"

    utils.to_pkl_gzip(obj=train_feat, path=train_file_path)
    utils.to_pkl_gzip(obj=test_feat, path=test_file_path)

    logger.info(f'''
    #========================================================================
    # COMPLETE MAKE FEATURE : {train_file_path}
    #========================================================================''')

sys.exit()
df['ir_max'] = df[ir_cols].max(axis=1)
df['ir_min'] = df[ir_cols].min(axis=1)
df['ir_std'] = df[ir_cols].std(axis=1)
#  utils.to_df_pkl(df=df, path='../eda/', fname='1024_prev_ir')
