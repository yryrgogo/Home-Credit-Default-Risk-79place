import numpy as np
import pandas as pd
import datetime
import glob
import sys
import re
from tqdm import tqdm
from itertools import combinations
from multiprocessing import Pool
import multiprocessing
from mv_wg_avg import exp_weight_avg
from target_encoding import target_encoding

sys.path.append('../../../github/module/')
from load_data import pararell_load_data
from feature_engineering import diff_feature, division_feature, product_feature, cat_to_target_bin_enc
from make_file import make_npy, make_feature_set
from logger import logger_func
from convinience_function import pararell_process, get_categorical_features

logger = logger_func()

start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

unique_id = 'SK_ID_CURR'
p_id = 'SK_ID_PREV'
target = 'TARGET'
ignore_features = [unique_id, target, 'valid_no', 'valid_no_2', 'is_train', 'is_test', 'SK_ID_PREV']
base = pd.read_csv('../data/base.csv')

#  cat = np.load('../features/impute/pure_submission_impute.npy')
#  base['pure_submission_impute'] = cat
#  base['bin13_pure_submission_impute'] = pd.qcut(x=base[ 'pure_submission_impute' 
#  ], q=13, duplicates='drop')
#  np.save(file='bin13_pure_submission_impute', arr=base['bin13_pure_submission_impute'].values)
#  sys.exit()

def install_target_encoding(base, df, bins=0, level_num=0):

    #  label_list = ['a_HOUSE_HOLD_CODE@', 'a_REGION_RATING_CLIENT_W_CITY']
    label_list = ['a_HOUSE_HOLD_CODE@', 'a_REGION_RATING_CLIENT_W_CITY']
    org_occ_list = ['bin10_a_ORGANIZATION_TYPE', 'bin10_a_OCCUPATION_TYPE']

    #  cat_list = get_categorical_features(data=df, ignore=ignore_features) + label_list
    #  cat_list = [cat for cat in cat_list if not(cat.count('FLAG')) and not(cat.count('bin'))]

    num_list = [col for col in df.columns if (col.count('ccb_') or col.count('is_')) and (not(col.count('train')) and not(col.count('test')) )]
    for cat in num_list:
        if cat.count('bin'):
            continue
        df[cat] = pd.qcut(x=df[cat], q=bins, duplicates='drop')
        df.rename(columns={cat:f'bin{bins}_{cat}'}, inplace=True)

    cat_list = [col for col in df.columns if col.count('bin')]

    #  bin_list = [col for col in df.columns if (col.count('bin20') or col.count('bin10') )]
    bin_list = [col for col in df.columns if col.count('bin')]

    elem = 'a_FLAG_OWN_REALTY'
    elem_2 = 'a_FLAG_OWN_CAR'
    elem_3 = 'bin10_a_OCCUPATION_TYPE'
    categorical_list = []

    #  for num in bin_list:
    for num in cat_list:
        for cat_2 in cat_list:
            if level_num==2:
                if not(num.count('is_')) and not(cat_2.count('ccb_')):
                    continue
                encode_list = [num, cat_2]
                length = len(df[encode_list].drop_duplicates())
                cnt_id = len(df[unique_id].drop_duplicates())
                if length>100 or length<50 or cnt_id/length<3000:
                    continue
                categorical_list.append(encode_list)

            elif level_num==3:
                for cat_3 in cat_list:
                    if cat_2==cat_3:
                        continue
                    encode_list = [num, cat_2, cat_3]

                    length = len(df[encode_list].drop_duplicates())
                    cnt_id = len(df[unique_id].drop_duplicates())
                    if length>100 or length<50 or cnt_id/length<3000:
                        continue
                    categorical_list.append(encode_list)

    method_list = ['mean', 'std']
    select_list = []
    val_col = 'valid_no_4'

    for cat in tqdm(categorical_list):
        length = len(df[cat].drop_duplicates())
        prefix = f'is_len{length}_'
        prefix = f'ccb_len{length}_'
        target_encoding(base=base, data=df, unique_id=unique_id, level=cat, method_list=method_list,
                        prefix=prefix, select_list=select_list, test=1, impute=1208, val_col=val_col, npy_key=target)


def previous_wavg(data, num_list, prefix='p_'):

    ' 並列処理（少ないとオーバーヘッドの方がでかそう） '
    #  arg_list = []
    #  for num in num_list:
    #      arg_list.append([data, num, prefix])
    #  pararell_process(agg_wrapper, arg_list)

    ' 重み付き平均 '
    weight_list = [0.97]
    for num in num_list:
        for weight in weight_list:
            wavg = exp_weight_avg(data=data, level=unique_id, weight=weight, label='DAYS_DECISION', value=num)
            result = base.merge(wavg.to_frame().reset_index(), on=unique_id, how='left').fillna(0)

            make_npy(logger=logger, result=result, ignore_list=ignore_features, prefix=prefix)


def make_installment():
    inst = pd.read_csv('../data/installments_payments.csv')
    inst.sort_values(by=[unique_id, p_id], inplace=True)

    inst['DAYS_INSTALMENT'] = (inst['DAYS_INSTALMENT']/30).astype('int')
    inst['IMPUTE_ENTRY_FLG'] = inst['DAYS_ENTRY_PAYMENT'].map(lambda x: 0 if x==x else 1)
    inst['IMPUTE_ENTRY_VALUE'] = inst['DAYS_INSTALMENT'] * inst['IMPUTE_ENTRY_FLG']
    inst['DAYS_ENTRY_PAYMENT'] = inst['DAYS_ENTRY_PAYMENT'].fillna(0) + inst['IMPUTE_ENTRY_VALUE']
    inst['DAYS_ENTRY_PAYMENT'] = (inst['DAYS_ENTRY_PAYMENT']/30).astype('int')
    drop_col = [col for col in inst.columns if col.count('IMPUTE')]
    inst.drop(drop_col, axis=1, inplace=True)

    # 集約する
    inst['INST_VERSION'] = inst['NUM_INSTALMENT_VERSION'].map(lambda x:9 if x>=8 else 6  if x==7 else x)

    inst['DIFF_AMT'] = inst['AMT_INSTALMENT'] - inst['AMT_PAYMENT']
    inst['DIFF_DAYS'] = inst['DAYS_INSTALMENT'] - inst['DAYS_ENTRY_PAYMENT']
    inst['DIFF_DAYS'].value_counts()

    # 前の行との差分
    inst.sort_values(by=[unique_id, p_id, 'DAYS_INSTALMENT'], inplace=True)
    inst['LAG_DAYS_INSTALMENT'] = inst.groupby([unique_id, p_id])['DAYS_INSTALMENT'].shift(-1)
    inst.sort_values(by=[unique_id, p_id, 'DAYS_ENTRY_PAYMENT'], inplace=True)
    inst['LAG_DAYS_ENTRY'] = inst.groupby([unique_id, p_id])['DAYS_ENTRY_PAYMENT'].shift(-1)
    inst['DIFF_DAYS_INSTALMENT'] = inst['LAG_DAYS_INSTALMENT'] - inst['DAYS_INSTALMENT']
    inst['DIFF_DAYS_ENTRY'] = inst['LAG_DAYS_ENTRY'] - inst['DAYS_ENTRY_PAYMENT']

    drop_col = [col for col in inst.columns if col.count('LAG_')]
    inst.drop(drop_col, axis=1, inplace=True)

    inst.to_csv('../data/installments_payments_after.csv', index=True)


def main():
    '''
    集計粒度であるカテゴリカラムをfeature_ext_sourceにわたし、
    そのカテゴリ粒度をext_sourceでターゲットエンコーディングする
    '''
    pd.set_option("display.max_rows", 130)

    base = pd.read_csv('../data/base.csv')
    #  data = pd.read_csv('../data/installments_payments.csv')
    #  data = make_feature_set(base[unique_id].to_frame(), '../features/valid_feature/*.npy')
    data = make_feature_set(base, '../features/valid_feature/is_*.npy')

    win = make_feature_set(base[unique_id].to_frame(), '../features/valid_feature/ccb_*.npy')
    #  win = pd.read_csv('../data/application_summary_set.csv')

    data  = win.merge(data, on=unique_id, how='left')

    #  bins = 10
    bins_list = [10, 20, 30]
    level_num=2
    #  bins_list = [20, 30]
    for bins in bins_list:
        install_target_encoding(base, data, bins, level_num)

if __name__ == '__main__':
    main()
