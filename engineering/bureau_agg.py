import numpy as np
import pandas as pd
import datetime
import glob
import sys
import re
from multiprocessing import Pool
import multiprocessing
from itertools import combinations

sys.path.append('../../../github/module/')
from preprocessing import set_validation, split_dataset, factorize_categoricals
from load_data import pararell_load_data, x_y_split
from convinience_function import get_categorical_features, get_numeric_features, row_number
from make_file import make_feature_set, make_npy
from logger import logger_func
from feature_engineering import base_aggregation, diff_feature, division_feature, cat_to_target_bin_enc

logger = logger_func()


#  logger = logger_func()
start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

path_data = '../data/'

unique_id = 'SK_ID_CURR'
b_id = 'SK_ID_BUREAU'
p_id = 'SK_ID_PREV'
target = 'TARGET'
ignore_features = [unique_id, target, 'valid_no', 'valid_no_1', 'valid_no_2',
                   'valid_no_3', 'valid_no_4', 'is_train', 'is_test', 'SK_ID_PREV', 'SK_ID_BUREAU']


def target_encoding(base, data, unique_id, level, method_list, prefix='', test=0, select_list=[], impute=1208, val_col='valid_no'):
    '''
    Explain:
        TARGET関連の特徴量を4partisionに分割したデータセットから作る.
        1partisionの特徴量は、残り3partisionの集計から作成する。
        test対する特徴量は、train全てを使って作成する
    Args:
        data(DF)             : 入力データ。カラムにはunique_idとvalid_noがある前提
        level(str/list/taple): 目的変数を集計する粒度。
        method(str)          : 集計のメソッド
        select_list(list)    : 特定のfeatureのみ保存したい場合はこちらにリストでfeature名を格納
    Return:
        カラム名は{prefix}{target}@{level}
    '''

    if str(type(level)).count('str'):
        level = [level]
    elif str(type(level)).count('tuple'):
        level = list(level)

    tmp_base = data[[unique_id, val_col] + level].drop_duplicates()
    if len(base) > 0:
        base = base[unique_id].to_frame().merge(
            tmp_base, on=unique_id, how='left')

    for method in method_list:
        result = pd.DataFrame([])
        valid_list = data[val_col].drop_duplicates().values
        if test == 0:
            valid_list.remove(-1)

        for valid_no in valid_list:

            if valid_no == -1:
                df = data
            else:
                df = data.query('is_train==1')
            '''
            集計に含めないpartisionのDFをdf_val.
            集計するpartisionのDFをdf_aggとして作成
            '''
            df_val = df[df[val_col] == valid_no][level].drop_duplicates()
            #  logger.info(f"\ndf_val: {df_val.shape}")

            df_agg = df[df[val_col] != valid_no][level+[target]]
            #  logger.info(f"\ndf_agg: {df_agg.shape}")

            #  logger.info(f'\nlevel: {level}\nvalid_no: {valid_no}')
            df_agg = base_aggregation(df_agg, level, target, method)

            ' リークしないようにvalidation側のデータにJOIN '
            tmp_result = df_val.merge(df_agg, on=level, how='left')
            tmp_result[val_col] = valid_no

            if len(result) == 0:
                result = tmp_result
            else:
                result = pd.concat([result, tmp_result], axis=0)
            #  logger.info(f'\ntmp_result shape: {result.shape}')

        result = base.merge(result, on=level+[val_col], how='left')

        for col in result.columns:
            if col.count('bin') and not(col.count(target)):
                result.drop(col, axis=1, inplace=True)

        if impute != 1208:
            print(result.head())
            result.fillna(impute, inplace=True)

        #  logger.info(f'\nresult shape: {result.shape}')
        #  logger.info(f'\n{result.head()}')

        make_npy(result, ignore_features, prefix, select_list=select_list)


def make_bureau():
    ' bureau '
    bureau = pd.read_csv('../data/bureau_cleansing.csv')
    bureau.sort_values(by=[unique_id, 'SK_ID_BUREAU',
                           'DAYS_CREDIT'], ascending=False, inplace=True)
    bureau = row_number(bureau, unique_id)

    base_bureau = bureau[['SK_ID_CURR', 'SK_ID_BUREAU']].drop_duplicates()

    ' bureau_balanceで遅延したことのあるIDのみJOINする '
    balance = pd.read_csv('../data/bureau_balance.csv')
    balance['STATUS'] = balance['STATUS'].map(
        lambda x: '-1' if x == 'C' or x == 'X' else x).astype('int')
    b_status = balance.groupby(b_id)['STATUS'].max()

    b_status = b_status.reset_index().query('STATUS>=0')
    bureau = bureau.merge(b_status, on=b_id, how='left')

    bureau['STATUS'] = bureau['STATUS'] + 1
    bureau['DAYS_ENDDATE_ORIGINAL'] = bureau.apply(lambda x: x['DAYS_ENDDATE_FACT'] if x['DAYS_ENDDATE_FACT'] < 99999 else x['DAYS_CREDIT_ENDDATE'], axis=1)

    ' ENDDATE '
    bureau['CREDIT_TERM'] = bureau['DAYS_ENDDATE_ORIGINAL'] - bureau['DAYS_CREDIT']
    bureau['CREDIT_TERM'] = bureau['CREDIT_TERM'].map(
        lambda x: 0 if x < 0 else x)
    bureau['LOAN_MONTHLY'] = bureau['AMT_CREDIT_SUM'] / bureau['CREDIT_TERM']

    ' AMT '
    bureau = diff_feature(df=bureau, first='AMT_CREDIT_SUM',
                          second='AMT_CREDIT_SUM_DEBT')
    bureau = diff_feature(df=bureau, first='AMT_ANNUITY',
                          second='LOAN_MONTHLY')

    ' CREDIT_TYPE'
    category = 'CREDIT_TYPE'
    bureau[category] = bureau[category].map(lambda x:
                                            'high_CREDIT_TYPE' if x.count('Another') or x.count('card') or x.count('Micro') or x.count('equipment') or x.count('replenish')
                                            else 'middle_CREDIT_TYPE' if x.count('Consumer') or x.count('Unknown')
                                            else 'row_CREDIT_TYPE' if x.count('Car') or x.count('Mortgage') or x.count('develop') or x.count('earmarked') or x.count('Real') or x.count('margin') or x.count('lending') or x.count('Inter') or x.count('Mobile')
                                            else 'other'
                                            )

    'row_no'
    bureau.sort_values(by=[unique_id, 'SK_ID_BUREAU', 'DAYS_CREDIT'], ascending=False, inplace=True)
    bureau = row_number(bureau, [unique_id, 'CREDIT_TYPE'])

    return bureau


def make_win_set(base):
    path = '../features/5_engineering/*.npy'
    df = make_feature_set(base, path)

    'positive率とサンプル数によってラベリングする'

    ' SUITE'
    df['a_NAME_TYPE_SUITE'] = df['a_NAME_TYPE_SUITE'].map({
        'Children': 'middle_row_risk', 'Family': 'middle_row_risk', 'Unaccompanied': 'middle_risk', 'Groupofpeople': 'hight_risk', 'Other_A': 'high_risk', 'Other_B': 'high_risk', 'Spouse,partner': 'row_risk'
    })

    ' INCOME_TYPE'
    df['a_NAME_INCOME_TYPE'] = df['a_NAME_INCOME_TYPE'].map(lambda x: 'Pensioner' if x.count(
        'Business') or x.count('Student') else 'Working' if x.count('Unemployed') or x.count('Maternity') else x)

    ' EDUCATION_TYPE'
    df['a_NAME_EDUCATION_TYPE'] = df['a_NAME_EDUCATION_TYPE'].map(
        lambda x: 'Secondary / secondary special' if x.count('Lower secondary') else 'Higher education' if x.count('Academic') else x)

    ' EDUCATION_TYPE'
    df['a_NAME_FAMILY_STATUS'] = df['a_NAME_FAMILY_STATUS'].map(lambda x: 'Married' if x.count(
        'Widow') or x.count('Unknown') else 'Civil marriage' if x.count('Single') or x.count('Separated') else x)

    ' ORGANIZATION '
    tmp = df.query('TARGET != -1')
    for category in ['a_ORGANIZATION_TYPE', 'a_OCCUPATION_TYPE']:
        df = cat_to_target_bin_enc(tmp, category)

    return df


def bureau_extract_agg(base, data, win, bins=0, val='', status='', term='', freq=0):
    ' bureauのデータセットをカテゴリや期間で絞る '


    ' カテゴリー区分 '
    if val.count('high'):
        data = data.query(f"CREDIT_TYPE=='{val}'")
    elif val.count('middle'):
        data = data.query(f"CREDIT_TYPE=='{val}'")
    elif val.count('row'):
        data = data.query(f"CREDIT_TYPE=='{val}'")

    ' ステータス区分 '
    if status.count('Act'):
        data = data.query(f"CREDIT_ACTIVE=='{status}'")
    elif status.count('Clo'):
        data = data.query(f"CREDIT_ACTIVE=='{status}'")
    elif status.count('og_a'):
        data = data.query(f"DAYS_ENDDATE_ORIGINAL != DAYS_ENDDATE_ORIGINAL or DAYS_ENDDATE_ORIGINAL>0")
    elif status.count('og_c'):
        data = data.query(f"DAYS_ENDDATE_ORIGINAL<=0")

    if freq > 0:
        data = data.query(f'row_no<={freq}')

    ' dataチェック '
#     print(data.shape)
#     print('*** SK_ID_CURR ***')
#     print(data[unique_id].drop_duplicates().shape)
    ' ユニークIDが10万に満たない場合は特徴量としない '
    if data[unique_id].drop_duplicates().count() < 100000:
        print( f'less than 10000 unique_id. combi: {bins}_{val}_{status}_{term}_{freq}')
        return

    print(f'combi: {bins}_{val}_{status}_{term}_{freq}')
    print(f'unique_id count: {data[unique_id].drop_duplicates().count()}')

    days_list = [col for col in data.columns if col.count('DAYS')]
    amt_list = [col for col in data.columns if col.count('AMT_')]

    amt = data[[unique_id] + amt_list].groupby(unique_id).agg('sum')
    days = data[[unique_id] + days_list].groupby(unique_id).agg('max')
    result = amt.join(days).reset_index()

    for col in result.columns:
        if col in ignore_features:
            continue
        elif col.count('AMT_'):
            result.rename(columns={col: f'{col}_sum@'}, inplace=True)
        elif col.count('DAYS_'):
            result.rename(columns={col: f'{col}_max@'}, inplace=True)

    ' ここからターゲットエンコーディング '
    method_list = ['mean', 'std']
    # method_list = ['sum', 'mean', 'std', 'max', 'min']

    val_col = 'valid_no_4'
    bin_list = [col for col in result.columns if col.count('@')]
    result = result.merge(win, on=[unique_id], how='inner')

    cat_list_1 = []
    # cat_list_1 = [
    #     'a_OCCUPATION_TYPE'
    #     ,'a_ORGANIZATION_TYPE'
    #     ,'a_NAME_EDUCATION_TYPE'
    #     ,'a_NAME_INCOME_TYPE'
    #     ,'a_CODE_GENDER'
    #     ,'a_NAME_TYPE_SUITE'
    #     ,'a_NAME_FAMILY_STATUS'
    #     ,'a_REGION_RATING_CLIENT_W_CITY'
    #     ,'a_HOUSE_HOLD_CODE@'
    #     ,'a_FLAG_WORK_PHONE'
    # ]

    cat_list_2 = [
        'a_OCCUPATION_TYPE', 'a_ORGANIZATION_TYPE', 'a_NAME_EDUCATION_TYPE', 'a_NAME_INCOME_TYPE', 'a_CODE_GENDER', 'a_NAME_FAMILY_STATUS', 'a_NAME_TYPE_SUITE', 'a_REGION_RATING_CLIENT_W_CITY', 'a_HOUSE_HOLD_CODE@', 'a_FLAG_WORK_PHONE'
    ]

    ' binをカテゴリとする場合はこちら '
    for col in bin_list:
        ' binより値の種類が少ない時は計算しない '
        if len(result[col].drop_duplicates()) < bins:
            cat_list_1.append(col)
        else:
            logger.info(f'bin:{bins}_col:{col}')
            logger.info(f'{result.head()}')
            result[f'bin{bins}_{col}'] = pd.qcut(x=result[col], q=bins, duplicates='drop')
            cat_list_1.append(f'bin{bins}_{col}')

    categorical = []
    for cat1 in cat_list_1:
        for cat2 in cat_list_2:
            if cat1 == cat2:
                continue
            cnt_id = len(result[unique_id].drop_duplicates())
            length = len(result[[cat1, cat2]].drop_duplicates())
            if length>100 or length<30 or cnt_id/length<3000:
                continue
            categorical.append([cat1, cat2])

    select_list = []

    for cat in categorical:
        length = len(result[cat].drop_duplicates())
        prefix = f'b_{val}_{status}_latest{freq}_len{length}_'
        target_encoding(base=base, data=result, unique_id=unique_id, level=cat, method_list=method_list, prefix=prefix, select_list=select_list, test=1, impute=1208, val_col=val_col)


def bureau_target_encoding(base, bureau, win):
    '集計前の絞り込み条件'
    'SK_ID_CURRが1万件以上のデータに限る'
    bins_list = [20, 15, 10]

    val_list = ['', 'high_CREDIT_TYPE', 'middle_CREDIT_TYPE', 'row_CREDIT_TYPE']

    status_list = ['Active', 'Closed', 'og_a', 'og_c']

    freq_list = [5]

    for bins in bins_list:
        for val in val_list:
            for status in status_list:
                for freq in freq_list:
                    bureau_extract_agg(base=base, data=bureau, win=win, bins=bins, val=val, status=status, freq=freq)
    sys.exit()
    for bins in bins_list:
        for val in val_list:
            for status in status_list:
                for term in term_list:
                    bureau_extract_agg(data=data, bins=bins,
                                       val=val, status=status, term=term)


def main():

    base = pd.read_csv('../data/base.csv')[unique_id].to_frame()
    #  bureau = make_bureau()
    #  bureau.to_csv('../data/bureau_summary_set.csv', index=False)
    #  sys.exit()
    bureau = pd.read_csv('../data/bureau_summary_set.csv')
    win = pd.read_csv('../data/application_summary_set.csv')

    bureau_target_encoding(base, bureau, win)


if __name__ == '__main__':

    main()
