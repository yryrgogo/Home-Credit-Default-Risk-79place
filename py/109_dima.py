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


# Curren Applicationに対するCNT_PAYMENTの予測値
df = utils.read_df_pkl('../input/clean_cpy*').reset_index()[[key, 'AMT_CREDIT', 'AMT_ANNUITY', target]]


#========================================================================
# Current ApplicationのInterest Rateを計算
#========================================================================
#========================================================================
# Dima's Interest_Rate
#========================================================================
df['3']=df['AMT_ANNUITY']*3-df['AMT_CREDIT']
df['3']=df['3']/df['AMT_CREDIT']
df['dima_ir_3@']=df['3'].apply(lambda x: x if 0<=x<=0.5 else np.nan)
df['6']=df['AMT_ANNUITY']*6-df['AMT_CREDIT']
df['6']=df['6']/df['AMT_CREDIT']
df['dima_ir_6@']=df['6'].apply(lambda x: x if 0<=x<=0.5 else np.nan)
df['9']=df['AMT_ANNUITY']*9-df['AMT_CREDIT']
df['9']=df['9']/df['AMT_CREDIT']
df['dima_ir_9@']=df['9'].apply(lambda x: x if 0<=x<=0.5 else np.nan)
df['12']=df['AMT_ANNUITY']*12-df['AMT_CREDIT']
df['12']=df['12']/df['AMT_CREDIT']
df['dima_ir_12@']=df['12'].apply(lambda x: x if 0<=x<=0.5 else np.nan)
df['18']=df['AMT_ANNUITY']*18-df['AMT_CREDIT']
df['18']=df['18']/df['AMT_CREDIT']
df['dima_ir_18@']=df['18'].apply(lambda x: x if  0<=x<=0.5 else np.nan)
df['24']=df['AMT_ANNUITY']*24-df['AMT_CREDIT']
df['24']=df['24']/df['AMT_CREDIT']
df['dima_ir_24@']=df['24'].apply(lambda x: x if 0<=x<=0.5 else np.nan)
df['36']=df['AMT_ANNUITY']*36-df['AMT_CREDIT']
df['36']=df['36']/df['AMT_CREDIT']
df['dima_ir_36@']=df['36'].apply(lambda x: x if  0<=x<=0.5 else np.nan)
df['48']=df['AMT_ANNUITY']*48-df['AMT_CREDIT']
df['48']=df['48']/df['AMT_CREDIT']
df['dima_ir_48@']=df['48'].apply(lambda x: x if  0<=x<=0.5 else np.nan)
df['10']=df['AMT_ANNUITY']*10-df['AMT_CREDIT']
df['10']=df['10']/df['AMT_CREDIT']
df['dima_ir_10@']=df['10'].apply(lambda x: x if 0<=x<=0.5 else np.nan)
df['15']=df['AMT_ANNUITY']*15-df['AMT_CREDIT']
df['15']=df['15']/df['AMT_CREDIT']
df['dima_ir_15@']=df['15'].apply(lambda x: x if 0<=x<=0.5 else np.nan)
df_rate=df[['SK_ID_CURR','AMT_ANNUITY','AMT_CREDIT', 'dima_ir_3@', 'dima_ir_6@', 'dima_ir_9@','dima_ir_10@','dima_ir_12@','dima_ir_15@','dima_ir_18@','dima_ir_24@','dima_ir_36@','dima_ir_48@']]
df_rate['AMT_ANNUITY'].replace(0,np.nan,inplace=True)
df_rate.dropna(subset=['AMT_ANNUITY'], inplace=True)
df_rate['AMT_CREDIT'].replace(0,np.nan,inplace=True)
df_rate.dropna(subset=['AMT_CREDIT'], inplace=True)
df_rate['3x']=df_rate['dima_ir_3@'].apply(lambda x: 3 if  x>=0 else 0)
df_rate['6x']=df_rate['dima_ir_6@'].apply(lambda x: 6 if  x>=0 else 0)
df_rate['9x']=df_rate['dima_ir_9@'].apply(lambda x: 9 if  x>=0 else 0)
df_rate['10x']=df_rate['dima_ir_10@'].apply(lambda x: 10 if  x>=0 else 0)
df_rate['12x']=df_rate['dima_ir_12@'].apply(lambda x: 12 if  x>=0 else 0)
df_rate['15x']=df_rate['dima_ir_15@'].apply(lambda x: 15 if  x>=0 else 0)
df_rate['18x']=df_rate['dima_ir_18@'].apply(lambda x: 18 if  x>=0 else 0)
df_rate['24x']=df_rate['dima_ir_24@'].apply(lambda x: 24 if  x>=0 else 0)
df_rate['36x']=df_rate['dima_ir_36@'].apply(lambda x: 36 if  x>=0 else 0)
df_rate['48x']=df_rate['dima_ir_48@'].apply(lambda x: 48 if  x>=0 else 0)
df_rate['dima_length@']=df_rate[['10x','12x','15x','18x','24x','36x','48x']].max(axis=1)
df_rate['dima_length@'].replace(0,np.nan,inplace=True)
df_rate.dropna(subset=['dima_length@'], inplace=True)

df_rate['dima_newrate@']=np.rate(nper=df_rate['dima_length@'], pmt=-df_rate['AMT_ANNUITY'],pv=df_rate['AMT_CREDIT'],fv=0.0)
df_rate[['dima_newrate@','dima_length@']]
df_rate['dima_lengthX@']=df_rate['dima_length@']
df=df.merge(right=df_rate[['SK_ID_CURR','dima_newrate@','dima_lengthX@']].reset_index(), how='left', on='SK_ID_CURR')
####


ir_cols = [col for col in df.columns if col.count('dima') and col.count('ir')]
df['dima_ir_mean@'] = df[ir_cols].mean(axis=1)
df['dima_ir_std@'] = df[ir_cols].std(axis=1)
df['dima_ir_max@'] = df[ir_cols].max(axis=1)
df['dima_ir_min@'] = df[ir_cols].min(axis=1)
ir_cols = [col for col in df.columns if col.count('dima') and col.count('ir')]

# CNT_PAYMENT系->過学習してるっぽい？
#  df['dima_Pred_CPY_diff_lengthX@'] = df['CNT_PAYMENT'].values - df['dima_lengthX@'].values
#  df['dima_Cal_CPY_diff_lengthX@'] = df['dima_lengthX@'].values - (df['AMT_CREDIT'].values / df['AMT_ANNUITY'].values)
#  train_file_path = f"../features/1_first_valid/train_{cpy}"
#  test_file_path = f"../features/1_first_valid/test_{cpy}"

# Feature Save
for col in ir_cols:
    if not(col.count('@')) or col in ignore_list:
        continue
    if not(col.count('ir_3@')) and not(col.count('ir_6@')) and not(col.count('ir_9@')):
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

