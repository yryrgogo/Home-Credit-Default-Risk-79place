import numpy as np
import pandas as pd
import datetime
import glob
import sys
import re
from itertools import combinations
from multiprocessing import Pool
import multiprocessing
from categorical_encoding import make_win_set, cat_to_target_bin_enc
from target_encoding import target_encoding
from tqdm import tqdm

sys.path.append('../../../github/module/')
from load_data import pararell_load_data
from feature_engineering import diff_feature, division_feature, product_feature, cat_to_target_bin_enc
from convinience_function import get_categorical_features, get_numeric_features, row_number
from make_file import make_feature_set, make_npy
from logger import logger_func

logger = logger_func()
start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

unique_id = 'SK_ID_CURR'
target = 'TARGET'
ignore_features = [unique_id, target, 'valid_no', 'valid_no_1', 'valid_no_2', 'valid_no_3', 'valid_no_4', 'is_train', 'is_test', 'SK_ID_PREV', 'SK_ID_BUREAU']


def feature_ext_source(base, df):
    '''
    Explain:
        ext_source関連の特徴量を作成
    Args:
    Return:
    '''

    ext_source = [
        'EXT_SOURCE_1',
        'EXT_SOURCE_2',
        'EXT_SOURCE_3'
    ]

    ext_combi = list(combinations(ext_source, 2))

    for ext in ext_combi:
        diff_feature(df, ext[0], ext[1])
        product_feature(df, ext[0], ext[1])

    df[f'a_ext_1-2-3@{unique_id}'] = df['EXT_SOURCE_1'] - \
        df['EXT_SOURCE_2'] - df['EXT_SOURCE_3']
    df[f'a_ext_1%2%3@{unique_id}'] = df['EXT_SOURCE_1'] / \
        df['EXT_SOURCE_2'] / df['EXT_SOURCE_3']

    ' 縦持ちにしてext_source1/2/3を集計する '
    tate = df[[unique_id]+ext_source].set_index(unique_id)
    tate = tate.stack().reset_index().rename(
        columns={'level_1': 'category', 0: 'EXT_SOURCE'})
    ext_agg = tate.groupby(unique_id, as_index=False)['EXT_SOURCE'].agg(
        {
            f'a_EXT_SOURCE_avg@': 'mean',
            f'a_EXT_SOURCE_max@': 'max',
            f'a_EXT_SOURCE_min@': 'min',
            f'a_EXT_SOURCE_std@': 'std'
        }
    )

    result = df.merge(ext_agg, on=unique_id, how='left')

    result = base.merge(result, on=unique_id, how='left')

    for col in result.columns:
        if col.count('@'):
            np.save(f'../features/1_first_valid/{col}', result[col].values)

    print(result)
    sys.exit()


def household_code(df):

    # 世帯人数
    df['CNT_CHILDREN'] = df['CNT_CHILDREN'].fillna(0)
    df['CNT_FAM_MEMBERS'] = df['CNT_FAM_MEMBERS'].fillna(0)

    df['HOUSE_HOLD_CODE@'] = df[['CNT_FAM_MEMBERS', 'CNT_CHILDREN', 'CODE_GENDER']].apply(
        # シングルファザー
        lambda x: 2 if (x['CNT_FAM_MEMBERS'] - x['CNT_CHILDREN'] == 1) and x['CNT_CHILDREN'] > 0 and x['CODE_GENDER'] == 'M'
        # 独身男性
        else 1 if x['CNT_FAM_MEMBERS'] == 1 and x['CODE_GENDER'] == 'M'
        # シングルマザー
        else 0 if (x['CNT_FAM_MEMBERS'] - x['CNT_CHILDREN'] == 1) and x['CNT_CHILDREN'] > 0 and x['CODE_GENDER'] == 'F'
        # 既婚男性
        else -1 if x['CNT_FAM_MEMBERS'] > 1 and x['CODE_GENDER'] == 'M'
        # 既婚女性
        else -2 if x['CNT_FAM_MEMBERS'] > 1 and x['CODE_GENDER'] == 'F'
        # 独身女性
        else -3 if x['CNT_FAM_MEMBERS'] == 1 and x['CODE_GENDER'] == 'F'
        # 不明
        else -4, axis=1)

    return df


def null_cnt_feature(base, df, unique_id):
    ' Nullの数を数えて変数にする '
    df = df.fillna(19891208)

    one = np.zeros(len(df.columns) * len(df)) + 1
    nans = pd.DataFrame(one.reshape(df.shape),
                        dtype='int32', columns=df.columns)
    nans = nans[df == 19891208]
    nans[unique_id] = df[unique_id]

    df_nans = nans.set_index(unique_id).stack()
    df_nans = df_nans.groupby(unique_id).sum().reset_index().rename(
        columns={0: 'nan_app_train@'})
    df = base.merge(df_nans, on=unique_id, how='left')

    return df


def document_number(df):

    doc_list = []
    for col in df.columns:
        if col.count('DOCUMENT'):
            #  if col.count('3') or col.count('16') or col.count('18'):
            doc_list.append(col)

    #  df['a_DOCUMENT_NUM@'] = 
    df_doc = df[doc_list]
    df_doc['FLAG_DOCUMENT_3'] = df_doc['FLAG_DOCUMENT_3']*-1
    df_doc = df_doc.stack().reset_index()
    result = df_doc.groupby(unique_id)[0].sum()

    np.save('../features/1_first_valid/a_all_document_number_-3', result)


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


def application_target_encoding(base, df, bins=0):

    if bins>0:
        abp = make_feature_set(base[unique_id].to_frame(), '../features/dima/*.npy')
        #  abp = make_feature_set(base, '../features/f_abp/*.npy')
        #  val_list = [col for col in abp.columns if col.count('valid')]
        #  abp.drop(val_list+[target, 'is_train', 'is_test'], axis=1, inplace=True)

        bin_list = [col for col in abp.columns if col.count('abp_')]

        for col in bin_list:
            ' NULLが多いfeatureは除く '
            #  null_len = len(abp[abp[col].isnull()])
            #  if null_len>100000:
            #      abp.drop(col, axis=1, inplace=True)
            #      continue
            abp[col] = abp[col].fillna(0)
            abp[col] = pd.qcut(x=abp[col], q=bins, duplicates='drop')
            abp.rename(columns={col:f'bin{bin}_{col}'}, inplace=True)

        df = df.merge(abp, on=unique_id, how='left')

    #  label_list = ['a_HOUSE_HOLD_CODE@', 'a_REGION_RATING_CLIENT_W_CITY']
    label_list = ['a_REGION_RATING_CLIENT_W_CITY']
    #  cat_list = ['bin10_a_ORGANIZATION_TYPE']
    cat_list = get_categorical_features(data=df, ignore=ignore_features) + label_list
    cat_list = [cat for cat in cat_list if not(cat.count('bin'))]
    cat_list = [cat for cat in cat_list if not(cat.count('GENDER')) and not(cat.count('FLAG')) and not(cat.count('OCC'))]
    #  bin_list = [col for col in df.columns if col.count('bin') and (not(col.count('ORG')) and not(col.count('OCC')) )]
    #  bin_list = [col for col in df.columns if (col.count('bin20') or col.count('bin30') ) and (col.count('AMT') or col.count('EXT') )]
    #  bin_list = [col for col in df.columns if (col.count('bin20') or col.count('bin30') ) and col.count('DAY')]
    bin_list = [col for col in df.columns if (col.count('bin20') or col.count('bin10') )]
    bin_list = [num for num in bin_list if not(num.count('ATION_TYPE'))]

    if bins>0:
        bin_list = [col for col in df.columns if (col.count('bin') and col.count('abp_'))]
        cat_list = cat_list + bin_list

    elem = 'a_FLAG_OWN_REALTY'
    elem_2 = 'a_FLAG_OWN_CAR'
    elem_3 = 'bin10_a_OCCUPATION_TYPE'
    bin_list = [1]
    categorical_list = []
    for cat in cat_list:
        for num in bin_list:
            encode_list = [cat, elem_3, elem, elem_2]
            #  ' 特定のカテゴリが含まれていたら組み合わせに含める '
            #  ok_flg = 0
            #  ok_flg_1 = 0
            #  ok_flg_2 = 0
            #  ok_flg_3 = 0
            #  for elem in cat_list:
            #      elem_1 = 'EMPLOYED'
            #      elem_2 = 'ORG'
            #      elem_3 = '@@@'
            #      #  if elem.count('HOUSE_HOLD') or elem.count('W_CITY'):
            #      if elem.count(elem_1):
            #          ok_flg_1 = 1
            #      if elem.count(elem_2):
            #          ok_flg_2 = 1
            #      if elem.count(elem_3):
            #          ok_flg_3 = 1
            #      ok_flg = ok_flg_1 + ok_flg_2 + ok_flg_3
            #  if ok_flg<=1:
            #      continue

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
        prefix = f'a_len{length}_'
        #  prefix = f'abp_vc{length}_'
        target_encoding(base=base, data=df, unique_id=unique_id, level=cat, method_list=method_list,
                        prefix=prefix, select_list=select_list, test=1, impute=1208, val_col=val_col, npy_key=target)


def application_division(df, num_list_1, num_list_2):

    for num_1 in num_list_1:
        for num_2 in num_list_2:
            if num_1==num_2:continue
            df = division_feature(df=df, first=num_1, second=num_2)

    return df


def make_tmp():
    num_list = get_numeric_features(data, ignore=ignore_features)
    num_list = [num for num in num_list if num.count('impute') or num.count('diff') or num.count('div')]

    data['a_AMT_CREDIT_diff_AMT_ANNUITY_impute_diff_AMT_GOODS_PRICE_impute@'] = (data['a_AMT_CREDIT'] - data['a_AMT_ANNUITY_impute']) - data['a_AMT_GOODS_PRICE_impute']
    data['a_AMT_INCOME_TOTAL_div_DAYS_EMPLOYED@_div_DAYS_REGISTRATION@'] = data['a_AMT_INCOME_TOTAL'] / data['a_DAYS_EMPLOYED_impute'] / data['a_DAYS_REGISTRATION']
    data['a_DAYS_EMPLOYED_impute_div_a_DAYS_REGISTRATION@'] = data['a_DAYS_EMPLOYED_impute'] / data['a_DAYS_REGISTRATION']
    data['a_DAYS_ID_PUBLISH_div_DAYS_EMPLOYED_impute@'] =  data['a_DAYS_ID_PUBLISH'] / data['a_DAYS_EMPLOYED_impute']
    data['a_DAYS_ID_PUBLISH_div_DAYS_EMPLOYED@_diff_DAYS_LAST_PHONE_CHANGE_div_DAYS_ID_PUBLISH@'] =  (data['a_DAYS_ID_PUBLISH'] / data['a_DAYS_EMPLOYED_impute']) - (data['a_DAYS_LAST_PHONE_CHANGE_impute'] / data['a_DAYS_ID_PUBLISH'])

    days_list = ['a_DAYS_BIRTH', 'a_DAYS_EMPLOYED_impute', 'a_DAYS_ID_PUBLISH', 'a_DAYS_REGISTRATION', 'a_DAYS_LAST_PHONE_CHANGE_impute']
    for num in days_list:
        data = division_feature(df=data, first='a_OWN_CAR_AGE_impute', second=num)
        if not(num.count('PHONE')):
            data = division_feature(df=data, first='a_DAYS_LAST_PHONE_CHANGE_impute', second=num)

    for num in num_list:
        data.rename(columns={num:f'{num}@'}, inplace=True)

    result = data
    make_npy(result, ignore_features, '', '', logger=logger)
    sys.exit()


def main():
    '''
    集計粒度であるカテゴリカラムをfeature_ext_sourceにわたし、
    そのカテゴリ粒度をext_sourceでターゲットエンコーディングする
    '''

    base = pd.read_csv('../data/base.csv')
    #  data = pd.read_csv('../data/application_train_test_after.csv')
    data = pd.read_csv('../data/application_summary_set.csv')



    #  data.set_index('SK_ID_CURR', inplace=True)

    #  bins = int(sys.argv[1])
    bins = 0
    application_target_encoding(base, data, bins)

    #  document_number(data)
    #  feature_ext_source(base, data)


if __name__ == '__main__':
    main()
