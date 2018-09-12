import numpy as np
import pandas as pd
import datetime
import glob
import sys
import re
from multiprocessing import Pool
import multiprocessing
from itertools import combinations

import os
HOME = os.path.expanduser('~')
sys.path.append(f"{HOME}/kaggle/github/library/")
import utils
from utils import logger_func, get_categorical_features, get_numeric_features, pararell_process
logger = logger_func()
pd.set_option('max_columns', 200)
pd.set_option('max_rows', 200)
from preprocessing import set_validation, split_dataset, get_dummies, factorize_categoricals
from feature_engineering import base_aggregation, diff_feature, division_feature, product_feature, cnt_encoding, select_category_value_agg, exclude_feature
from make_file import make_npy, make_feature_set


start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())


# ===========================================================================
# global variables
# ===========================================================================
dir = "../features/1_first_valid"
key = 'SK_ID_CURR'
target = 'TARGET'
ignore_list = [key, target, 'SK_ID_BUREAU', 'SK_ID_PREV']
prefix = 'b_'

# ===========================================================================
# DATA LOAD
# ===========================================================================
base = pd.read_csv('../input/base.csv')[[key, target]]
#  base = pd.read_csv( '../../../kaggle/kaggle_private/home-credit-default-risk/data/base.csv')

# ===========================================================================
# 集計方法を選択
# ===========================================================================
agg_code = ['base', 'caliculate', 'cnt', 'category', 'combi', 'dummie'][0]
diff = [True, False][0]
div = [True, False][0]
pro = [True, False][0]
method_list = ['sum', 'mean', 'var', 'max', 'min']

#  path_list = glob.glob('../features/1_first_valid/*.gz')


def main():

    df = utils.read_df_pickle(path='../input/clean_bureau*')
    '''
    BASE AGGRIGATION
    単一カラムをlevelで粒度指定して基礎集計
    '''
    if agg_code == 'base':

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
                del result, renu, tmp_result
                gc.collect()

    elif agg_code == 'caliculate':

        '''
        CALICULATION
        複数カラムを四則演算し新たな特徴を作成する
        '''
        f1_list = []
        f2_list = []
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

                if diff:
                    df = diff_feature(df=df, first=f1, second=f2)
                elif div:
                    df = division_feature(df=df, first=f1, second=f2)
                elif pro:
                    df = product_feature(df=df, first=f1, second=f2)

        for col in df.columns:
            utils.to_pickle(path=f"{dir}/{col}.fp", obj=df[col].values)

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
                    select_category_value_agg(base, df=df, key=key, category_col=cat, value=value, method, ignore_list, prefix)
                    #  arg_list.append(base, df, key, cat, value,
                                    method, ignore_list, prefix)

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


if __name__ == '__main__':

    main()
