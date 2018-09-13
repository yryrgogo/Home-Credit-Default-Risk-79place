import numpy as np
import pandas as pd
import datetime
import glob
import sys
import re
from multiprocessing import Pool
import multiprocessing
from itertools import combinations

import gc
import os
HOME = os.path.expanduser('~')
sys.path.append(f"{HOME}/kaggle/github/library/")
import utils
from utils import logger_func, get_categorical_features, get_numeric_features, pararell_process
logger = logger_func()
pd.set_option('max_columns', 200)
pd.set_option('max_rows', 200)
from preprocessing import set_validation, split_dataset, get_dummies, factorize_categoricals
from feature_engineering import base_aggregation, diff_feature, division_feature, product_feature, cnt_encoding, select_category_value_agg, exclude_feature, target_encoding
from make_file import make_npy, make_feature_set, make_raw_feature


start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

# ===========================================================================
# global variables
# ===========================================================================
dir = "../features/1_first_valid"
key = 'SK_ID_CURR'
target = 'TARGET'
ignore_list = [key, target, 'SK_ID_BUREAU', 'SK_ID_PREV']

# ===========================================================================
# DATA LOAD
# ===========================================================================

# ===========================================================================
# 集計方法を選択
# ===========================================================================
agg_code = ['base', 'raw', 'tgec', 'caliculate', 'cnt', 'category', 'combi', 'dummie'][2]
diff = [True, False][0]
div = [True, False][0]
pro = [True, False][0]
method_list = ['sum', 'mean', 'var', 'max', 'min']
#  path_list = glob.glob('../features/1_first_valid/*.gz')


def main():

    #  path = f'../input/add_clean_app*'
    #  prefix = 'app_'
    #  df = utils.read_df_pickle(path=path)
    path = f'../input/{sys.argv[1]}*'
    prefix = sys.argv[2]
    enc_feat_list = ['EXT_SOURCE_1', target]
    #  enc_feat_list = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    base = pd.read_csv('../input/base.csv')[[key, target]]
    df = utils.read_df_pickle(path=path)
    gc.collect()

    '''
    BASE AGGRIGATION
    単一カラムをlevelで粒度指定して基礎集計
    '''
    if agg_code == 'base':
        one_base_agg(df=df, prefix=prefix)
    elif agg_code == 'raw':
        make_raw_feature(df, prefix, ignore_list=ignore_list)
    elif agg_code == 'tgec':
        app = utils.read_df_pickle(path='../input/add_clean_app*')[[key] + enc_feat_list]
        cat_list = get_categorical_features(df=df, ignore=ignore_list)
        for enc_feat in enc_feat_list:
            if enc_feat.count(target):
                base = base[key].to_frame().merge(app[[key, enc_feat]], on=key, how='inner')
            else:
                base = base[key].to_frame().merge(app[[key, enc_feat, target]], on=key, how='inner')
            for cat in cat_list:
                target_encoding(logger=logger, base=base, df=df, key=key, level=cat, target=target, enc_feat=enc_feat, prefix=prefix, ignore_list=ignore_list)
    elif agg_code == 'caliculate':
        df = two_calicurate(df=df)
        if prefix!='app_':
            one_base_agg(df=df, prefix=prefix)
        else:
            for col in df.columns:
                file_path = f'{prefix}{col}.fp'
                utils.to_pkl_gzip(obj=df[col].values, path=f"{dir}/{prefix}{col}.fp")
                logger.info(f'''
#========================================================================
# COMPLETE MAKE CALICURATE FEATURE : {file_path}
#========================================================================''')

    elif agg_code == 'cnt':
        '''
        COUNT ENCODING
        level粒度で集計し、cnt_valを重複有りでカウント
        '''
        cat_list = get_categorical_features(df=df, ignore=ignore_list)

        for category_col in cat_list:
            df = cnt_encoding(df, category_col, ignore_list)
        df = base.merge(df, on=key, how='inner')
        cnt_cols = [col for col in df.columns if col.count('cntec')]
        for col in cnt_cols:
            if exclude_feature(col, df[col].values): continue
            utils.to_pickle(path=f"{dir}/{col}.fp", obj=df[col].values)

    elif agg_code == 'category':

        ' カテゴリカラムの中のvalue毎に集計する '
        arg_list = []
        cat_list = get_categorical_features(df=df, ignore=ignore_list)
        num_list = get_numeric_features(df=df, ignore=ignore_list)
        for cat in cat_list:
            for value in num_list:
                for method in method_list:
                    select_category_value_agg(base, df=df, key=key, category_col=cat, value=value, method=method, ignore_list=ignore_list, prefix=prefix)
                    #  arg_list.append(base, df, key, cat, value, method, ignore_list, prefix)

        #  pararell_process(select_cat_wrapper, arg_list)

    elif agg_code == 'combi':
        combi_num=[1, 2, 3][0]
        cat_combi=list(combinations(categorical, combi_num))

    elif agg_code == 'dummie':

        ' データセットのカテゴリカラムをOneHotエンコーディングし、その平均をとる '
        cat_list=get_categorical_features(data, ignore_features)
        df=get_dummies(df = df, cat_list = cat_list)


def base_agg_wrapper(args):
    return base_aggregation(*args)
def select_cat_wrapper(args):
    return select_category_value_agg(*args)
def one_base_agg(df, prefix):
    # =======================================================================
    # 集計するカラムリストを用意
    # =======================================================================
    num_list = get_numeric_features(df=df, ignore=ignore_list)

    # 並列処理→DFが重いと回らないかも
    #  arg_list = []
    #  for num in num_list:
    #      for method in method_list:
    #          tmp = df[[key, num]]
    #          arg_list.append([tmp, key, num, method, prefix, '', base])

    #  ' データセットにおけるカテゴリカラムのvalue毎にエンコーディングする '
    #  call_list = pararell_process(base_agg_wrapper, arg_list)
    #  result = pd.concat(call_list, axis=1)

    #  for col in result.columns:
    #      if not(col.count('@')) or col in ignore_list:
    #          continue
    #      #  utils.to_pickle(path=f"{dir}/{col}.fp", obj=result[col].values)
    #  sys.exit()

    # 直列処理
    for num in num_list:
        for method in method_list:
            tmp = df[[key, num]]
            tmp_result = base_aggregation(
                df=tmp, level=key, method=method, prefix=prefix, feature=num)
            result = base.merge(tmp_result, on=key, how='left')
            renu = result[result[target].isnull()]
            for col in result.columns:
                if not(col.count('@')) or col in ignore_list:
                    continue
                if exclude_feature(col, result[col].values): continue
                if exclude_feature(col, renu[col].values): continue

                file_path = f"{dir}/{col}.fp"
                #  utils.to_pickle(path=file_path, obj=result[col].values)
                utils.to_pkl_gzip(obj=result[col].values, path=file_path)
                logger.info(f'''
                #========================================================================
                # COMPLETE MAKE FEATURE : {file_path}
                #========================================================================''')
            del result, renu, tmp_result
            gc.collect()

def two_calicurate(df):
    used_list = []
    '''
    CALICULATION
    複数カラムを四則演算し新たな特徴を作成する
    '''
    f1_list = get_numeric_features(df=df, ignore=ignore_list)
    f2_list = get_numeric_features(df=df, ignore=ignore_list)
    used_lsit = []
    for f1 in f1_list:
        for f2 in f2_list:
            ' 同じ組み合わせの特徴を計算しない '
            if f1 == f2:
                continue
            if sorted([f1, f2]) in used_list:
                continue
            used_list.append(sorted([f1, f2]))

            ' For home-credit'
            if (not(f1.count('revo')) and f2.count('revo')) or (f1.count('revo') and not(f2.count('revo'))):
                continue
            if not(f1.count('AMT')) and not(f1.count('DAYS')) and not(f1.count('OWN')):
                continue
            if not(f2.count('AMT')) and not(f2.count('DAYS')) and not(f2.count('OWN')):
                continue

            if diff:
                df = diff_feature(df=df, first=f1, second=f2)
            if div:
                df = division_feature(df=df, first=f1, second=f2)
            if pro:
                df = product_feature(df=df, first=f1, second=f2)

    use_cols = []
    for col in df.columns:
        if col.count('_div_') or col.count('_diff_') or col.count('_pro_'):
            use_cols.append(col)
    df = df[[key]+use_cols]
    return df


if __name__ == '__main__':

    main()
