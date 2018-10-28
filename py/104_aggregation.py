feat_no = '104_'
pararell = True
pararell = False
arithmetic=True
ext_feat=False

import numpy as np
import pandas as pd
import datetime
import sys
import multiprocessing
from tqdm import tqdm

import os
HOME = os.path.expanduser('~')
sys.path.append(f"{HOME}/kaggle/data_analysis/library/")
import utils
from utils import logger_func, get_categorical_features, get_numeric_features, pararell_process
logger = logger_func()
pd.set_option('max_columns', 200)
pd.set_option('max_rows', 200)
from feature_engineering import base_aggregation, diff_feature, division_feature, product_feature


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


def pararell_arith(feat_combi):
    f1 = feat_combi[0]
    f2 = feat_combi[1]
    feat1 = diff_feature(df=df, first=f1, second=f2)
    feat2 = division_feature(df=df, first=f1, second=f2)
    feat3 = product_feature(df=df, first=f1, second=f2)
    feat = pd.concat([feat1, feat2, feat3], axis=1)
    return feat

if arithmetic:
    used_list = []
    '''
    CALICULATION
    複数カラムを四則演算し新たな特徴を作成する
    '''
    num_list = get_numeric_features(df=df, ignore_list=ignore_list)
    amt_list = [col for col in num_list if col.count('AMT_')]
    days_list = [col for col in num_list if col.count('DAYS_') or col.count('OWN')]
    num_list = amt_list + days_list

    ext_list = [col for col in df.columns if col.count('EXT_')]

    f1_list = num_list
    f2_list = num_list
    #  f1_list = ext_list
    #  f2_list = ext_list
    #  f2_list = ['EXT_SOURCE_2']
    used_lsit = []
    result_feat = pd.DataFrame()
    for f1 in tqdm(f1_list):
        for f2 in f2_list:
            ' 同じ組み合わせの特徴を計算しない '
            if f1 == f2:
                continue
            if sorted([f1, f2]) in used_list:
                continue
            used_list.append(sorted([f1, f2]))

            if not(pararell):

                tmp = df[[f1, f2]]
                feat1 = diff_feature(df=tmp, first=f1, second=f2, only_feat=True)
                feat2 = division_feature(df=tmp, first=f1, second=f2, only_feat=True)
                feat3 = product_feature(df=tmp, first=f1, second=f2, only_feat=True)
                tmp_feat = pd.concat([feat1, feat2, feat3], axis=1)
                #  tmp_feat = pd.concat([feat1, feat3], axis=1)
                #  tmp_feat = feat2.to_frame()

                if len(result_feat):
                    result_feat = pd.concat([result_feat, tmp_feat], axis=1)
                else:
                    result_feat = pd.concat([df[[key, target]], tmp_feat], axis=1)

    if pararell:
        cpu_cnt=multiprocessing.cpu_count()
        if len(used_list)>cpu_cnt:
            for i in range(0, int(len(used_list)/cpu_cnt)+1, 1):
                if len(used_list)>=cpu_cnt*(i+1):
                    feat_list  =pararell_process(pararell_arith, used_list[i*cpu_cnt : cpu_cnt*(i+1)])
                else:
                    feat_list  =pararell_process(pararell_arith, used_list[i*cpu_cnt : len(used_list)])

                df_feat = pd.concat(feat_list, axis=1)
                if i==0:
                    df = pd.concat([df[[key, target]], df_feat], axis=1)
                else:
                    df = pd.concat([df, df_feat], axis=1)
        else:
            feat_list  =pararell_process(pararell_arith, used_list)
            df_feat = pd.concat(feat_list, axis=1)
            df = pd.concat([df[[key, target]], df_feat], axis=1)
    else:
        df = result_feat

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

