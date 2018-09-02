agg_flg = 3
import numpy as np
import pandas as pd
import datetime
import glob
import sys
import re
from multiprocessing import Pool
import multiprocessing
from itertools import combinations
from previous_agg import previous_wavg
from categorical_encoding import make_win_set, cat_to_target_bin_enc
from target_encoding import target_encoding

sys.path.append('../../../github/module/')
from preprocessing import set_validation, split_dataset, factorize_categoricals
from load_data import pararell_load_data, x_y_split
from convinience_function import get_categorical_features, get_numeric_features, row_number
from make_file import make_feature_set, make_npy
from logger import logger_func
from feature_engineering import base_aggregation, diff_feature, division_feature


logger = logger_func()
start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

unique_id = 'SK_ID_CURR'
b_id = 'SK_ID_BUREAU'
p_id = 'SK_ID_PREV'
target = 'TARGET'
ignore_features = [unique_id, target, 'valid_no', 'valid_no_1', 'valid_no_2',
                   'valid_no_3', 'valid_no_4', 'is_train', 'is_test', 'SK_ID_PREV', 'SK_ID_BUREAU']

prev_cat_list = [
    'CHANNEL_TYPE'
    ,'NAME_CONTRACT_TYPE'
    ,'NAME_YIELD_GROUP'
    ,'NAME_PORTFOLIO'
    ,'CODE_REJECT_REASON'
    ,'PRODUCT_COMBINATION'
    ,'NAME_GOODS_CATEGORY'
    ,'NAME_CASH_LOAN_PURPOSE'
    ,'NAME_SELLER_INDUSTRY'
]


def make_previous(data):

    seller_code = {
        'Autotechnology': 'high_seller',
        'Connectivity': 'high_seller',
        'Jewelry': 'high_seller',
        'XNA': 'high_seller',

        'Consumerelectronics': 'low_seller',
        'Industry': 'low_seller',
        'Furniture': 'low_seller',
        'Clothing': 'low_seller',
        'Construction': 'low_seller',
        'MLMpartners': 'low_seller',
        'Tourism': 'low_seller'
    }
    data['NAME_SELLER_INDUSTRY'] = data['NAME_SELLER_INDUSTRY'].map(
        seller_code)

    purpose_code = {
        'Hobby': 'high_pp_risk',
        'Car repairs': 'high_pp_risk',
        'Refusal to name the goal': 'high_pp_risk',
        'Money for a third person': 'high_pp_risk',
        'Gasification / water supply': 'high_pp_risk',
        'Payments on other loans': 'high_pp_risk',
        'Urgent needs': 'high_pp_risk',
        'Building a house or an annex': 'high_pp_risk',
        'Medicine': 'high_pp_risk',
        'Wedding / gift / holiday': 'high_pp_risk',
        'Buying a used car': 'high_pp_risk',
        'Furniture': 'high_pp_risk',
        'Repairs': 'high_pp_risk',
        'Business development': 'high_pp_risk',
        'Other': 'high_pp_risk',
        'Purchase of electronic equipment': 'high_pp_risk',
        'Buying a holiday home / land': 'high_pp_risk',
        'Education': 'high_pp_risk',
        'Journey': 'high_pp_risk',
        'Everyday expenses': 'high_pp_risk',

        'XAP': 'low_pp_risk',
        'XNA': 'low_pp_risk',
        'Buying a garage': 'low_pp_risk',
        'Buying a new car': 'low_pp_risk',
        'Buying a home': 'low_pp_risk',

    }
    data['NAME_CASH_LOAN_PURPOSE'] = data['NAME_CASH_LOAN_PURPOSE'].map(
        purpose_code)

    combi_code = {
        'Cash Street: middle': 'high_combi',
        'Cash X-Sell: high': 'high_combi',
        'Cash Street: high': 'high_combi',
        'Card Street': 'high_combi',

        'Cash Street: low': 'middle_combi',
        'Cash': 'middle_combi',
        'Card X-Sell': 'middle_combi',
        'POS mobile with interest': 'middle_combi',

        'POS other with interest': 'low_combi',
        'POS mobile without interest': 'low_combi',
        'Cash X-Sell: low': 'low_combi',
        'POS household with interest': 'low_combi',
        'POS others without interest': 'low_combi',

        'POS household without interest': 'low_combi',
        'Cash X-Sell: low': 'low_combi',
        'POS industry with interest': 'low_combi',
        'POS industry without interest': 'low_combi',
    }
    data['PRODUCT_COMBINATION'] = data['PRODUCT_COMBINATION'].map(combi_code)

    goods_code = {
        'Insurance': 'high_goods',
        'Vehicles': 'high_goods',
        'XNA': 'high_goods',
        'Auto Accessories': 'high_goods',
        'Jewelry': 'high_goods',
        'Mobile': 'high_goods',
        'Office Appliances': 'high_goods',
        'Direct Sales': 'high_goods',

        'Computers': 'middle_goods',
        'Audio/Video': 'middle_goods',
        'Photo /CinemaEquipment': 'middle_goods',
        'Sport and Leisure': 'middle_goods',
        'Consumer Electronics': 'middle_goods',
        'Construction Materials': 'middle_goods',
        'Gardening': 'middle_goods',
        'Homewares': 'middle_goods',

        'Additional Service': 'low_goods',
        'Medicine': 'low_goods',
        'Weapon': 'low_goods',
        'Furniture': 'low_goods',
        'other': 'low_goods',
        'clothing and Accessories': 'low_goods',
        'education': 'low_goods',
        'medical Supplies': 'low_goods',
        'tourism': 'low_goods',
        'fitness': 'low_goods',
        'animals': 'low_goods'
    }
    data['NAME_GOODS_CATEGORY'] = data['NAME_GOODS_CATEGORY'].map(goods_code)

    reject_code = {
        'SCOFR': 'middle_reject',
        'LIMIT': 'middle_reject',
        'HC': 'middle_reject',
        'XNA': 'middle_reject',
        'SCO': 'middle_reject',
        'VERIF': 'middle_reject',

        'CLIENT': 'low_reject',
        'XAP': 'low_reject',
        'SYSTEM': 'low_reject',
    }

    ' SUITE'
    data['NAME_TYPE_SUITE'] = data['NAME_TYPE_SUITE'].fillna('XNA')
    data['NAME_TYPE_SUITE'] = data['NAME_TYPE_SUITE'].map(lambda x: 'row_risk' if x.count('Child') or x.count('Family') else 'high_risk' if x.count('XNA') else 'middle_risk')

    data['CODE_REJECT_REASON'] = data['CODE_REJECT_REASON'].map(reject_code)

    data['NAME_PORTFOLIO'] = data['NAME_PORTFOLIO'].map(
        lambda x: 'POS' if x == 'Cars' else 'Cards' if x == 'XNA' else x)

    data['NAME_YIELD_GROUP'] = data['NAME_YIELD_GROUP'].map(
        lambda x: 'low_normal' if x == 'low_action' else 'high' if x.count('XNA') else x)

    data['NAME_CONTRACT_TYPE'] = data['NAME_CONTRACT_TYPE'].map(
        lambda x: 'Revolving loans' if x == 'XNA' else 'Consumer loans' if x.count('Cash') else x)

    data['CHANNEL_TYPE'] = data['CHANNEL_TYPE'].map(
        lambda x: 'Stone' if x.count('corporate') or x.count('dealer') or x.count('Regional') else 'Contact center' if x.count('AP') else 'Country-wide' if x.count('office') else x)

    ' positive率の可視化 '
    #  print(data[[unique_id, target, 'CHANNEL_TYPE']].drop_duplicates().groupby(['CHANNEL_TYPE']).mean())

    days_list = [
        'DAYS_DECISION',
        'DAYS_FIRST_DRAWING',
        'DAYS_FIRST_DUE',
        'DAYS_LAST_DUE_1ST_VERSION',
        'DAYS_LAST_DUE',
        'DAYS_TERMINATION',
        'DAYS_LASTPLAN_MONTH',
        'DAYS_ORIGINAL_LAST_DUE',
        'DAYS_LAST_DUE_DIFF_LASTPLAN_MONTH',
        'DAYS_TERM_MONTH',
        'DAYS_TERM_MONTH_1ST'
    ]

    ' first_due - decision '
    data = diff_feature(df=data, first=days_list[0], second=days_list[2])
    ' last_due - due_1st_version '
    data = diff_feature(df=data, first=days_list[3], second=days_list[4])
    ' termination - last_due '
    data = diff_feature(df=data, first=days_list[4], second=days_list[5])
    ' decision - last_due '
    data['DAYS_TERM_MONTH'] = data[days_list[4]] - data[days_list[0]]
    ' decision - last_due_1st '
    data['DAYS_TERM_MONTH_1ST'] = data[days_list[3]] - data[days_list[0]]

    ' credit / application '
    data = division_feature(df=data, first='AMT_CREDIT',
                            second='AMT_APPLICATION')
    ' credit / annuity '
    data['CNT_ANNUITY'] = data['AMT_CREDIT'] / data['AMT_ANNUITY']
    ' cnt_annuity - cnt_payment  '
    data = diff_feature(df=data, first='CNT_ANNUITY', second='CNT_PAYMENT')

    " last_due - (decision + (amt_credit_div_amt_annuity)) 実際の完了日と支払額から月毎の返済として逆算した場合の月数の差 "
    data['DAYS_LASTPLAN_MONTH'] = data[days_list[0]] + \
        (data['AMT_CREDIT'] / data['AMT_ANNUITY'])
    data['DAYS_LAST_DUE_DIFF_LASTPLAN_MONTH'] = data[days_list[4]] - \
        data['DAYS_LASTPLAN_MONTH']
    data['DAYS_ORIGINAL_LAST_DUE'] = data[days_list[0:7]].apply(
        lambda x: x[4] if x[4] == x[4] else x[3] if x[3] == x[3] else x[6], axis=1)

    data.sort_values(by=[unique_id, 'SK_ID_PREV', 'DAYS_DECISION'], ascending=False, inplace=True)
    data = row_number(data, unique_id)

    return data


def previous_target_encoding(base, data, win, bins=0, prefix='', cnt_id=0):

    categorical_list = get_categorical_features(data=data, ignore=ignore_features)
    days_list = [col for col in data.columns if col.count('DAYS')]
    days_list.remove('DAYS_LAST_DUE')
    amt_list = [col for col in data.columns if col.count('AMT_')]
    num_list = get_numeric_features(data=data, ignore=ignore_features)
    oth_list = [col for col in num_list if (not(col.count('AMT_')) or not(col.count('DAYS_'))) and (col.count('HOUR') or col.count('SELL'))]
    pay_list = [col for col in num_list if (not(col.count('AMT_')) or not(col.count('DAYS_'))) and (col.count('CNT_') or col.count('RATE_DOWN'))]

    amt = data[[unique_id] + amt_list].groupby(unique_id).agg('sum')
    days = data[[unique_id] + days_list].groupby(unique_id).agg('max')
    oth = data[[unique_id] + oth_list].groupby(unique_id).agg('mean')
    pay = data[[unique_id] + pay_list].groupby(unique_id).agg('max')
    result = amt.join(days).join(oth).join(pay)

    for col in result.columns:
        if col in ignore_features:
            continue
        elif col.count('AMT_'):
            result.rename(columns={col: f'{col}_sum@'}, inplace=True)
        elif col.count('DAYS_'):
            result.rename(columns={col: f'{col}_max@'}, inplace=True)
        elif col.count('CNT_') or col.count('RATE'):
            result.rename(columns={col: f'{col}_max@'}, inplace=True)
        elif col.count('HOUR') or col.count('SELL') or col.count('NFLAG'):
            result.rename(columns={col: f'{col}_mean@'}, inplace=True)

    ' ここからターゲットエンコーディング '
    method_list = ['mean', 'std']
    # method_list = ['sum', 'mean', 'std', 'max', 'min']

    val_col = 'valid_no_4'
    bin_list = [col for col in result.columns if col.count('@') or col.count('CNT')]
    result = result.merge(win, on=[unique_id], how='inner')

    #  cat_list = get_categorical_features(data=win, ignore=ignore_features)
    cat_list = ['a_HOUSE_HOLD_CODE@', 'a_OCCUPATION_TYPE', 'a_ORGANIZATION_TYPE', 'a_REGION_RATING_CLIENT_W_CITY']
    cat_list_1 = []
    cat_list_2 = [col for col in cat_list if not(col.count('bin'))]

    ' binをカテゴリとする場合はこちら '
    for col in bin_list:
        ' binより値の種類が少ない時は計算しない '
        if len(result[col].drop_duplicates()) < bins:
            cat_list_1.append(col)
        else:
            logger.info(f'bin:{bins}_col:{col}')
            logger.info(f'{result.head()}')
            result[f'bin{bins}_{col}'] = pd.qcut( x=result[col], q=bins, duplicates='drop')
            cat_list_1.append(f'bin{bins}_{col}')

    categorical = []
    for cat1 in cat_list_1:
        for cat2 in cat_list_2:
            if cat1 == cat2:
                continue
            length = len(result[[cat1, cat2]].drop_duplicates())
            if length>100 or length<40 or (cnt_id/length)<3000:
                logger.info(f'Overfitting Risk. cat1: {cat1} |cat2: {cat2} |length: {length} |cnt_id: {cnt_id}')
                continue
            categorical.append([cat1, cat2])

    select_list = []

    for cat in categorical:
        length = len(result[cat].drop_duplicates())
        target_encoding(base=base, data=result, unique_id=unique_id, level=cat, method_list=method_list,
                        prefix=f'{prefix}len{length}_' , select_list=select_list, test=1, impute=1208, val_col=val_col)


def prev_extract_data(base, previous, win, cat, bins_list, val, status, apr, freq, term):
    '集計前の絞り込み条件'
    'SK_ID_CURRが1万件以上のデータに限る'

    ' カテゴリの1つのvalueに絞る '
    data = previous.query(f"{cat}=='{val}'")

    ' previousのデータセットをカテゴリや期間で絞り、集計に回す '
    if freq > 0:
        #  prefix = f'p_less100k_{apr}_{cat}_{val}_{status}_latest{freq}_'
        prefix = f'p_{apr}_{cat}_{val}_{status}_latest{freq}_'
    elif freq == 0:
        prefix = f'p_{apr}_{cat}_{val}_{status}_{term}_'

    ' ステータス区分 '
    if status.count('og_a'):
        data = data.query(
            f"DAYS_LAST_DUE!=DAYS_LAST_DUE or DAYS_LAST_DUE>0")
    elif status.count('og_c'):
        data = data.query(f"DAYS_LAST_DUE<=0")

    ' 承認区分 '
    if apr.count('Approved') or apr.count('Refused'):
        data = data.query(f"NAME_CONTRACT_STATUS=='{apr}'")

    ' 期間 '
    if term.count('3mon'):
        data = data.query(f"DAYS_DECISION>=-3")
    elif term.count('6mon'):
        data = data.query(f"DAYS_DECISION>=-6")
    elif term.count('1year'):
        data = data.query(f"DAYS_DECISION>=-12")
    elif term.count('2year'):
        data = data.query(f"DAYS_DECISION>=-24")
    elif term.count('3year'):
        data = data.query(f"DAYS_DECISION>=-36")

    if freq > 0:
        data = data.query(f'row_no<={freq}')

    ' ユニークIDが10万に満たない場合は特徴量としない '
    cnt_id = data[unique_id].drop_duplicates().count()
    if cnt_id < 100000:
    #  if data[unique_id].drop_duplicates().count() >= 100000:
    #  if (data[unique_id].drop_duplicates().count() >= 100000) or (data[unique_id].drop_duplicates().count() < 30000):
        #  print( f'more than 100000 unique_id. combi: {val}_{status}_{term}_{freq}')
        print( f'less than 100000 unique_id. combi: {val}_{status}_{term}_{freq}')
        return

    print(f'combi: {val}_{status}_{term}_{freq}')
    print( f'unique_id count: {data[unique_id].drop_duplicates().count()}')

    if agg_flg==1 or agg_flg==2:

        ' 集計するfeatureをピックアップ '
        days_list = [col for col in data.columns if col.count('DAYS')]
        days_list.remove('DAYS_TERMINATION')
        days_list.remove('DAYS_TERM_MONTH')
        days_list.remove('DAYS_TERM_MONTH_1ST')
        days_list.remove('DAYS_FIRST_DUE')
        days_list.remove('DAYS_FIRST_DRAWING')
        amt_list = [col for col in data.columns if col.count('AMT_')]
        num_list = get_numeric_features(data=data, ignore = ignore_features)
        oth_list = [col for col in num_list if (not(col.count('AMT_')) and not(col.count('DAYS_'))) and (col.count('HOUR') or col.count('SELL'))]
        pay_list = [col for col in num_list if (not(col.count('AMT_')) and not(col.count('DAYS_'))) and (col.count('CNT_') or col.count('RATE_DOWN'))]
        method_list = ['sum', 'mean', 'std', 'max', 'min']
        feature_list = days_list + amt_list + oth_list + pay_list

        if agg_flg==1:

            for method in method_list:
                for feature in feature_list:
                    if feature.count('DAYS'):
                        if method != 'max': continue
                    elif feature.count('AMT') or feature.count('CNT'):
                        if method != 'sum' or method != 'std': continue
                    elif feature.count('HOUR') or feature.count('RATE_DOWN'):
                        if method != 'mean': continue
                    else:continue

                    tmp = base_aggregation(data=data, level=unique_id, feature=feature, method=method)
                    result = base.merge(tmp, on=unique_id, how='left')
                    select_list = []
                    make_npy(result, ignore_features, prefix, select_list=select_list)
        elif agg_flg==2:
            for num in num_list:
                if num.count('SELL'): num_list.remove(num)
            previous_wavg(data=data, num_list=num_list, prefix=prefix)
    elif agg_flg==3:
        for bins in bins_list:
            previous_target_encoding(base=base, data=data, win=win, bins=bins, prefix=prefix, cnt_id=cnt_id)


def pararell_previous(arg_list):
    p = Pool(multiprocessing.cpu_count())
    p.map(agg_wrapper, arg_list)


def agg_wrapper(args):
    return target_encoding(*args)
    #  return previous_extract_agg(*args)


def main():

    base = pd.read_csv('../data/base.csv')[unique_id].to_frame()
    #  prev = pd.read_csv('../data/previous_cleansing.csv')

    #  previous = make_previous(prev)
    #  previous.to_csv('../data/previous_summary_category.csv', index=False)
    #  sys.exit()
    previous = pd.read_csv('../data/previous_summary_category.csv')
    num_list = ['DAYS_LAST_DUE_1ST_VERSION', 'AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_GOODS_PRICE']
    drop_list = [col for col in previous.columns if col.count('FIRST') or col.count('TERM')]
    previous.drop(drop_list+num_list, axis=1, inplace=True)

    win = pd.read_csv('../data/application_summary_set.csv')
    #  for col in win.columns:
    #      print(col)
    #  sys.exit()


    ' データチェック '
    #  for p in prev_cat_list:
    #      for v in previous[p].drop_duplicates():
    #          tmp = previous.query(f"{p}=='{v}'")
    #          print(f'p:{p} v:{v}')
    #          print(tmp[unique_id].drop_duplicates().count())
    #  sys.exit()
    bins_list = [20, 15, 10]
    bins_list = [10, 5]
    ' status関係なしの場合は空にする '
    status_list = ['', 'og_a', 'og_c']
    apr_list = ['Approved', 'Refused']
    term_list = ['3month', '6month', '1year', '2year', '3year']
    #  freq_list = [3, 5, 10]
    freq_list = [10]

    for cat in prev_cat_list:
        val_list = [val for val in previous[cat].drop_duplicates()]
        for val in val_list:
            for status in status_list:
                for apr in apr_list:
                    for freq in freq_list:
                        prev_extract_data(base, previous, win, cat, bins_list=bins_list, val=val, status=status, apr=apr, freq=freq, term='')


if __name__ == '__main__':

    main()
