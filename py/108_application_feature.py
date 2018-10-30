feat_no = '108_'
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
fname = 'app'
prefix = feat_no + f'{fname}_'
df = utils.read_df_pkl(path=f'../input/clean_{fname}*.p')


# Null Count
df['Null_Cnt@'] = df.isnull().sum(axis=1)
# Document
doc_list = [col for col in df.columns if col.count('DOCUMENT')]
df['DOCUMENT_SUM@'] = df[doc_list].sum(axis=1)
# Document 3,5,6,8,11,18
doc_list2 = [col for col in doc_list if col.count('DOCUMENT') and ( col.count('_3') or col.count('_5') or col.count('_6') or col.count('_8') or col.count('_11') or col.count('_18') ) ]
df['DOCUMENT_SUM2@'] = df[doc_list2].sum(axis=1)

#  cat_list = get_categorical_features(df=df, ignore_list=ignore_list)

# ボツ
#  df['NEW_REGION_POPULATION_RELATIVE@'] = (df['REGION_POPULATION_RELATIVE']*10000).astype('int')
#  df.drop('REGION_POPULATION_RELATIVE', axis=1, inplace=True)

# INCOME
#  df['INCOME_per_CHILD@'] = df['AMT_INCOME_TOTAL'] / df['CNT_CHILDREN']
#  df['INCOME_per_FAMILY@'] = df['AMT_INCOME_TOTAL'] * df['CNT_FAM_MEMBERS']

# ボツ
#  df['HOUSE_HOLD_CODE@'] = df[['CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'CODE_GENDER']].apply(lambda x:
#                                                                                       '1' if x[0]==0 and x[1]==1 and x[2]=='M'
#                                                                                       else '2' if x[0]>=1 and x[1]==2 and x[2]=='M'
#                                                                                       else '3' if x[0]>=1 and x[1]==2 and x[2]=='F'
#                                                                                       else '4' if x[0]>=0 and x[1]<=5 and x[2]=='M'
#                                                                                       else '5' if x[0]>=0 and x[1]> 5 and x[2]=='M'
#                                                                                       else '6' if x[0]==0 and x[1]==1 and x[2]=='F'
#                                                                                       else '7' if x[0]>=0 and x[1]<=5 and x[2]=='F'
#                                                                                       else '8' if x[0]>=0 and x[1]> 5 and x[2]=='F'
#                                                                                       else 9
#                                                                                       , axis=1)

# INCOMEを使ったカテゴリのエンコーディング（あんま効かない）
#  for cat in cat_list:
#      if (cat.count('NAME') and not(cat.count('CONTRACT'))) or cat.count('TION_TYPE') or cat.count('HOUSE_HOLD'):

#          df_feat = df.groupby(cat)['AMT_INCOME_TOTAL'].mean().reset_index().rename(columns={'AMT_INCOME_TOTAL':f'INCOME_mean@{cat}'})
#          df = df.merge(df_feat, on=cat, how='inner')
#          df_feat = df.groupby(cat)['AMT_INCOME_TOTAL'].std().reset_index().rename(columns={'AMT_INCOME_TOTAL':f'INCOME_std@{cat}'})


# Feature Save
for col in df.columns:
    if not(col.count('@')) or col in ignore_list:
        continue

    train_file_path = f"../features/1_first_valid/train_{prefix}{col}"
    test_file_path = f"../features/1_first_valid/test_{prefix}{col}"
    utils.to_pkl_gzip(obj=df[~df[target].isnull()][col].values, path=train_file_path)
    utils.to_pkl_gzip(obj=df[df[target].isnull()][col].values, path=test_file_path)

    logger.info(f'''
    #========================================================================
    # COMPLETE MAKE FEATURE : {train_file_path}
    #========================================================================''')


#  EXT average
if ext_feat:
    col = 'EXT_SOURCE_avg@'
    ext_list = [col for col in df.columns if col.count('EXT_')]
    df[col] = df[ext_list].mean(axis=1)
    train_file_path = f"../features/1_first_valid/train_{prefix}{col}"
    test_file_path = f"../features/1_first_valid/test_{prefix}{col}"
    utils.to_pkl_gzip(obj=df[~df[target].isnull()][col].values, path=train_file_path)
    utils.to_pkl_gzip(obj=df[df[target].isnull()][col].values, path=test_file_path)

