import numpy as np
import pandas as pd
import datetime
import glob
import sys
import re
from itertools import combinations
from multiprocessing import Pool
import multiprocessing
from mv_wg_avg import exp_weight_avg

sys.path.append('../../../github/module/')
from load_data import pararell_load_data
from feature_engineering import diff_feature, division_feature, product_feature
from make_file import make_npy
from logger import logger_func
from convinience_function import pararell_process, get_categorical_features

logger = logger_func()

start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

unique_id = 'SK_ID_CURR'
p_id = 'SK_ID_CURR'
target = 'TARGET'
ignore_features = [unique_id, target, 'valid_no', 'valid_no_2', 'is_train', 'is_test', 'SK_ID_PREV']
base = pd.read_csv('../data/base.csv')


' 並列処理 '
def agg_wrapper(args):
    return pararell_wavg(*args)
def pararell_wavg(data, num, prefix):

    weight_list = [0.99, 0.95]
    for weight in weight_list:
        wavg = exp_weight_avg(data=data, level=unique_id, weight=weight, label='DAYS_DECISION', value=num)
        result = base.merge(wavg.to_frame().reset_index(), on=unique_id, how='left').fillna(0)

        make_npy(logger=logger, result=result, ignore_list=ignore_features, prefix=prefix)


def previous_wavg(data, num_list, prefix='p_'):

    ' 並列処理（少ないとオーバーヘッドの方がでかそう） '
    #  arg_list = []
    #  for num in num_list:
    #      arg_list.append([data, num, prefix])
    #  pararell_process(agg_wrapper, arg_list)

    ' 重み付き平均 '
    weight_list = [0.99]
    for num in num_list:
        for weight in weight_list:
            wavg = exp_weight_avg(data=data, level=unique_id, weight=weight, label='DAYS_DECISION', value=num)
            result = base.merge(wavg.to_frame().reset_index(), on=unique_id, how='left').fillna(0)

            make_npy(logger=logger, result=result, ignore_list=ignore_features, prefix=prefix)



def select_level(df, status='', category='', num_list=[]):

    if len(status) > 0:
        df_status = df.query(f"NAME_CONTRACT_STATUS=='{status}'")
    else:
        df_status = df

    if len(category) > 0:
        for cat in df[category].drop_duplicates():
            tmp = df_status.query(f"{category}=='{cat}'")

            print(f'{status} | {category} | {cat}')

            prefix = f'p_{status}_{category}_{cat}_'

            previous_wavg(data=tmp, num_list=num_list, prefix=prefix)


def main():
    '''
    集計粒度であるカテゴリカラムをfeature_ext_sourceにわたし、
    そのカテゴリ粒度をext_sourceでターゲットエンコーディングする
    '''

    data = pd.read_csv('../data/previous_application_after.csv')
    data['DAYS_LAST_ORIGINAL'] = data.apply(lambda x: x['DAYS_LAST_DUE'] if x['DAYS_LAST_DUE'] < 99999 else x['DAYS_LAST_DUE_1ST_VERSION'], axis=1)
    #  data.set_index('SK_ID_CURR', inplace=True)
    category_list = get_categorical_features(data=data, ignore=ignore_features)
    category_list = [col for col in category_list if not(col.count('WEEK')) and not(col.count('FLAG'))]

    amt_list = [
        'AMT_CREDIT',
        'AMT_ANNUITY',
        'AMT_APPLICATION',
        'AMT_DOWN_PAYMENT',
        'CNT_ANNUITY',
        'AMT_YIELD'
    ]

    #  days_list = [
    #      'DAYS_DECISION',
    #      'DAYS_LAST_ORIGINAL',
    #      'PREV_CREDIT_TERM'
    #  ]

    feature_list = amt_list

    data['CNT_ANNUITY'] = data['AMT_CREDIT'] / data['AMT_ANNUITY']
    data['AMT_YIELD'] = data['AMT_CREDIT'] / data['AMT_APPLICATION']
    data['PREV_CREDIT_TERM'] = data['DAYS_LAST_ORIGINAL'] / data['DAYS_DECISION']

    status_list = [
        'Approved',
        'Refused'
    ]

    ' カテゴリ振り直しをしたのは上の５つ'
    #  category_list = [
    #      'SELLER_CODE',
    #      'REJECT_CODE',
    #      'COMBI_CODE',
    #      'PURPOSE_CODE',
    #      'GOODS_CODE',
    #      'PORTFOLIO_CODE',
    #      'YIELD_CODE',
    #      'CONTRACT_CODE',
    #      'NAME_PRODUCT_TYPE',
    #      'CHANNEL_CODE',

    #      #  'CNT_PAYMENT',
    #      #  'NAME_PAYMENT_TYPE',
    #      #  'NAME_CLIENT_TYPE',
    #  ]

    for status in status_list:
        for category in category_list:
            select_level(data, status=status, category=category, num_list=feature_list)


if __name__ == '__main__':
    main()
