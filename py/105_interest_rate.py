standard=True
standard=False
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

#========================================================================
# Previous ApplicationのInterest Rateを計算
#========================================================================
prev_ir = False
if prev_ir:
    app = utils.read_df_pkl('../input/clean_app*')[[key, target]]
    df = utils.read_df_pkl('../input/clean_prev*')
    df = df[[key, prev_key, dd, acr, aan, cpy, adp, co_type]].merge(app, on=key, how='inner')
    df = df[~df[cpy].isnull()]

    for cnt in range(3, 64, 3):
        if cnt<=60:
            ir = ( (df[aan].values * cnt) / df[acr].values ) - 1.0
            df[f'ir_{cnt}@'] = ir
            df[f'ir_{cnt}@'] = df[f'ir_{cnt}@'].map(lambda x: x if (0.0<x) and (x<0.5) else np.nan)
            print(f"{cnt} :", len(df[f'ir_{cnt}@'].dropna()))
            if len(df[f'ir_{cnt}@'].dropna())<len(df)*0.001:
                df.drop(f'ir_{cnt}@', axis=1, inplace=True)
                continue
        else:
            ir = ( (df[aan].values * df[cpy].values) / df[acr].values ) - 1.0
            df[f'ir_pred@'] = ir
            df[f'ir_pred@'] = df[f'ir_pred@'].map(lambda x: x if (0.0<x) and (x<0.5) else np.nan)
            cnt = 'pred'

    ir_cols = [col for col in df.columns if col.count('ir_')]
    df['ir_mean'] = df[ir_cols].mean(axis=1)
    df['ir_max'] = df[ir_cols].max(axis=1)
    df['ir_min'] = df[ir_cols].min(axis=1)
    df['ir_std'] = df[ir_cols].std(axis=1)
    utils.to_df_pkl(df=df, path='../eda/', fname='1024_prev_ir')


# Curren Applicationに対するCNT_PAYMENTの予測値
df = utils.read_df_pkl('../input/clean_cpy*')
df['Pred_CPY_diff_Cal_CPY@'] = df['CNT_PAYMENT'].values - (df['AMT_CREDIT'].values / df['AMT_ANNUITY'].values)


if standard:
    #========================================================================
    # SK_ID_BUREAUが
    #========================================================================
    kb = 'SK_ID_BUREAU'
    bur = utils.read_df_pkl('../input/clean_bur*')[[key, kb]].groupby(key)[kb].max().reset_index()
    df = df.reset_index().merge(bur, on=key, how='left')

    # TrainでSK_ID_BUREAUをもつレコード
    train =df[~df[target].isnull()]
    df_not = train[~train[kb].isnull()]
    bur_bin = pd.qcut(x=df_not[kb], q=10)
    df_not['bur_bin'] = bur_bin

    # TrainでSK_ID_BUREAUをもたないレコード
    df_null = train[train[kb].isnull()]
    df_null['bur_bin'] = 'train_no_bureau'

    train = pd.concat([df_not, df_null], axis=0).sort_index()

    # Testはまとめて標準化する
    test =df[df[target].isnull()]
    test['bur_bin'] = 'test'
    df = pd.concat([train, test], axis=0).sort_index()

#========================================================================
# Current ApplicationのInterest Rateを計算
#========================================================================
# CNT_PAYMENT
train_file_path = f"../features/1_first_valid/train_{cpy}"
test_file_path = f"../features/1_first_valid/test_{cpy}"

# Current Application CNT_PAYMENT Save as Feature
#  utils.to_pkl_gzip(obj=df[~df[target].isnull()][cpy].values, path=train_file_path)
#  utils.to_pkl_gzip(obj=df[df[target].isnull()][cpy].values, path=test_file_path)
#  utils.to_pkl_gzip(obj=df[~df[target].isnull()][ 'Pred_CPY_diff_Cal_CPY@' ].values, path=train_file_path)
#  utils.to_pkl_gzip(obj=df[df[target].isnull()][ 'Pred_CPY_diff_Cal_CPY@' ].values, path=test_file_path)

# 金利が何回分の支払いに対して発生しているか不明なので、3回刻みで一通り作る
for cnt in range(3, 64, 3):
    if cnt<=60:
        ir = ( (df[aan].values * cnt) / df[acr].values ) - 1.0
        df[f'ir_{cnt}@'] = ir
        df[f'ir_{cnt}@'] = df[f'ir_{cnt}@'].map(lambda x: x if (0.0<x) and (x<0.5) else np.nan)
        print(f"{cnt} :", len(df[f'ir_{cnt}@'].dropna()))
        if len(df[f'ir_{cnt}@'].dropna())<len(df)*0.001:
            df.drop(f'ir_{cnt}@', axis=1, inplace=True)
            continue
    else:
        ir = ( (df[aan].values * df[cpy].values) / df[acr].values ) - 1.0
        df[f'ir_pred@'] = ir
        df[f'ir_pred@'] = df[f'ir_pred@'].map(lambda x: x if (0.0<x) and (x<0.5) else np.nan)
        cnt = 'pred'

    #========================================================================
    # 時系列で標準化
    #========================================================================
    if standard:
        tmp_mean = df.groupby('bur_bin')[f'ir_{cnt}@'].mean().reset_index().rename(columns={f'ir_{cnt}@':'mean'})
        tmp_std = df.groupby('bur_bin')[f'ir_{cnt}@'].std().reset_index().rename(columns={f'ir_{cnt}@':'std'})
        df = df.merge(tmp_mean, on='bur_bin', how='inner')
        df = df.merge(tmp_std, on='bur_bin', how='inner')
        df[f'stan_ir_{cnt}@'] = (df[f'ir_{cnt}@'].values - df['mean'].values) / df['std'].values
        df.drop([f'ir_{cnt}@', 'mean', 'std'], axis=1, inplace=True)

ir_cols = [col for col in df.columns if col.count('ir_') or col.count('CNT_PAY')]

if standard:
    df['stan_ir_mean@'] = df[ir_cols].mean(axis=1)
    df['stan_ir_std@'] = df[ir_cols].std(axis=1)
    df['stan_ir_max@'] = df[ir_cols].max(axis=1)
    df['stan_ir_min@'] = df[ir_cols].min(axis=1)
else:
    df['ir_mean@'] = df[ir_cols].mean(axis=1)
    df['ir_std@'] = df[ir_cols].std(axis=1)
    df['ir_max@'] = df[ir_cols].max(axis=1)
    df['ir_min@'] = df[ir_cols].min(axis=1)


ir_cols = [col for col in df.columns if col.count('ir_') or col.count('CNT_PAY')]
# Feature Save
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

#  utils.to_df_pkl(df=df, path='../eda/', fname='1024_prev_ir')
