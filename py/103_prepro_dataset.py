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
from utils import logger_func, mkdir_func
logger = logger_func()
import eda
from utils import get_categorical_features, get_numeric_features, pararell_process
from feature_engineering import base_aggregation, diff_feature, division_feature, product_feature, cnt_encoding, select_category_value_agg, exclude_feature, target_encoding
from tqdm import tqdm

#========================================================================
# Global Variable
from info_home_credit import hcdr_key_cols
key, target, ignore_list = hcdr_key_cols()
#========================================================================

def fe_bureau(df):
    amt_list = [col for col in df.columns if col.count('AMT')]
    day_list = [col for col in df.columns if col.count('DAYS')]

    def combi_caliculate(df, num_list):
        used_list = []
        for f1 in num_list:
            for f2 in num_list:
                if f1==f2:continue
                if sorted([f1, f2]) in used_list:continue
                used_list.append(sorted([f1, f2]))
                df = division_feature(df=df, first=f1, second=f2)
                df = diff_feature(df=df, first=f1, second=f2)
        return df

    df = combi_caliculate(df, amt_list)
    df = combi_caliculate(df, day_list)

    return df


def make_num_features(df, filekey):
    mkdir_func(f'../features/{filekey}')

    if filekey.count('bur'):
        df = fe_bureau(df)

    #========================================================================
    # カテゴリの内容別にNumeric Featureを切り出す
    #========================================================================
    num_list = get_numeric_features(df=df, ignore_list=ignore_list)
    cat_list = get_categorical_features(df=df, ignore_list=[])

    few_list = []
    for cat in tqdm(cat_list):
        for val in tqdm(df[cat].drop_duplicates()):
            length = len(df[df[cat]==val])
            if length < len(df)*0.002:
                few_list.append(val)
                continue
            for num in num_list:
            #  pararell_process(, num_list)
                df[f'{num}_{cat}-{val}@'] = df[num].where(df[cat]==val, np.nan)
                df[f'{num}_{cat}-fewlist@'] = df[num].where(df[cat].isin(few_list), np.nan)

    logger.info(f'{fname} SET SHAPE : {df.shape}')

    #========================================================================
    # Feature Save & Categorical Encoding & Feature Save 
    #========================================================================
    train = df[~df[target].isnull()]
    test = df[df[target].isnull()]

    categorical_features = get_categorical_features(df=train, ignore_list=ignore_list)

    #========================================================================
    # Numeric Feature Save
    #========================================================================
    for col in train.columns:
        if col in categorical_features:continue
        result_train = train[col].values
        result_test = test[col].values
        logger.info(f"COL: {col} | LENGTH: {len(result_train)}")
        utils.to_pkl_gzip(obj=train[col].values, path=f'../features/{filekey}/train_{col}')
        if col != target:
            utils.to_pkl_gzip(obj=test[col].values, path=f'../features/{filekey}/test_{col}')


def make_cat_features(df, filekey):
    mkdir_func(f'../features/{filekey}')
    train = df[~df[target].isnull()]
    test = df[df[target].isnull()]
    categorical_features = get_categorical_features(df=train, ignore_list=ignore_list)

    #========================================================================
    # Categorical Feature Encode
    #========================================================================
    # Factorize
    logger.info("Factorize Start!!")
    for col in categorical_features:
        for col in categorical_features:
            train[f"lbl_{col}@"], indexer = pd.factorize(train[col])
            test[f"lbl_{col}@"] = indexer.get_indexer(test[col])

    # Count Encoding
    logger.info("Count Encoding Start!!")
    for col in categorical_features:
        train = cnt_encoding(train, col, ignore_list=ignore_list)
        test = cnt_encoding(test, col, ignore_list=ignore_list)

    #========================================================================
    # Categorical Feature Save
    #========================================================================
    for col in train.columns:
        logger.info("Saving Features...")
        if col.count('@'):
            result_train = train[col].values
            result_test = test[col].values
            logger.info(f"COL: {col} | LENGTH: {len(result_train)}")
            utils.to_pkl_gzip(obj=result_train, path=f'../features/{filekey}/train_{col}')
            utils.to_pkl_gzip(obj=result_test, path=f'../features/{filekey}/test_{col}')

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

#========================================================================
# Start
#========================================================================
utils.start(sys.argv[0])

app = utils.read_df_pkl(path='../input/clean_app*.p')[[key, target]]

fname_list = [
    'bureau'
    #  'prev'
    #  ,'install'
    #  ,'pos'
    #  ,'ccb'
]
for fname in fname_list:
    logger.info(f"{fname} Start!")
    df_feat = utils.read_df_pkl(path=f'../input/clean_{fname}*.p')
    # Target Join
    df_feat = df_feat.merge(app, on=key, how='inner')
    #  df_feat = df_feat.head(10000)

    #  make_num_features(df=df_feat, filekey=fname)
    make_cat_features(df=df_feat, filekey=fname)

#  pre = utils.read_df_pkl(path='../input/clean_prev*.p')
#  pos = utils.read_df_pkl(path='../input/clean_pos*.p')
#  ins = utils.read_df_pkl(path='../input/clean_ins*.p')
#  ccb = utils.read_df_pkl(path='../input/clean_ccb*.p')

utils.end(sys.argv[0])

#  pre_eda = eda.df_info(pre)
