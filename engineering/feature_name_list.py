
def previous_cat():

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

    return prev_cat_list


def previous_num():

    prev_num_list = [
        'DAYS_DECISION',
        'DAYS_LAST_DUE_max',
        'DAYS_LASTPLAN_MONTH_max',
        'DAYS_ORIGINAL_LAST_DUE',
        'DAYS_LAST_DUE_DIFF_LASTPLAN_MONTH',
        'DAYS_TERM_MONTH',
        'AMT_ANNUITY',
        'AMT_APPLICATION_sum',
        'AMT_CREDIT_sum',
        'AMT_DOWN_PAYMENT',
        'AMT_CREDIT_div_AMT_APPLICATION',
        'HOUR_APPR_PROCESS_START',
        'RATE_DOWN_PAYMENT',
        'SELLERPLACE_AREA'
    ]

    return prev_num_list

def application_cat():

    app_cat_list = [
        'EDUCATION'
        ,'INCOME'
        ,'SUITE'
        ,'WORK_PHONE'
        ,'OCCUPATION'
        ,'ORGANIZATION'
        ,'HOUSE_HOLD'
        ,'GENDER'
        ,'FAMILY'
        ,'REGION'
    ]

    return app_cat_list


def main():
    bin_list = [
        'AMT_ANNUITY',
        'AMT_CREDIT_diff_AMT_ANNUITY@_div_AMT_ANNUITY@',
        'AMT_CREDIT_diff_AMT_ANNUITY@_div_AMT_GOODS_PRICE@',
        'AMT_CREDIT_div_AMT_ANNUITY@',
        'AMT_GOODS_PRICE',
        'a_impute_EXT_SOURCE_1@',
        'a_impute_EXT_SOURCE_2@',
        'a_impute_EXT_SOURCE_3@'
    ]

    categorical = [
        #      [f'bin{bins}_AMT_CREDIT_div_AMT_ANNUITY@', f'bin{bins}_a_impute_EXT_SOURCE_1@'],
        #      [f'bin{bins}_AMT_CREDIT_diff_AMT_ANNUITY@_div_AMT_ANNUITY@', f'bin{bins}_a_impute_EXT_SOURCE_2@'],
        #      [f'bin{bins}_AMT_CREDIT_diff_AMT_ANNUITY@_div_AMT_ANNUITY@', f'bin{bins}_a_impute_EXT_SOURCE_3@'],
        #      [f'bin{bins}_AMT_CREDIT_diff_AMT_ANNUITY@_div_AMT_GOODS_PRICE@', f'bin{bins}_AMT_GOODS_PRICE'],
        #      [f'bin{bins}_AMT_CREDIT_div_AMT_ANNUITY@', f'bin{bins}_AMT_GOODS_PRICE'],
        #      [f'bin{bins}_AMT_ANNUITY', f'bin{bins}_AMT_CREDIT_diff_AMT_ANNUITY@_div_AMT_ANNUITY@'],
        #      [f'bin{bins}_AMT_CREDIT_diff_AMT_ANNUITY@_div_AMT_ANNUITY@', f'bin{bins}_AMT_GOODS_PRICE'],
        #      [f'bin{bins}_AMT_ANNUITY', f'bin{bins}_AMT_CREDIT_div_AMT_ANNUITY@']
        #  ]
        #      ,
        [f'bin{bins}_a_impute_EXT_SOURCE_1@', 'FLAG_WORK_PHONE'],
        [f'bin{bins}_a_impute_EXT_SOURCE_2@', 'FLAG_WORK_PHONE'],
        [f'bin{bins}_AMT_CREDIT_diff_AMT_ANNUITY@_div_AMT_ANNUITY@',
            'FLAG_WORK_PHONE'],
        [f'bin{bins}_AMT_CREDIT_div_AMT_ANNUITY@', 'FLAG_WORK_PHONE'],
        [f'bin{bins}_a_impute_EXT_SOURCE_3@', 'FLAG_WORK_PHONE']
    ]

    bin_list = [
        'a_impute_EXT_SOURCE_1@',
        'a_impute_EXT_SOURCE_2@',
        'a_impute_EXT_SOURCE_3@',
    ]

    categorical = [
        ['bin_a_impute_EXT_SOURCE_1@', 'CODE_GENDER'],
        ['bin_a_impute_EXT_SOURCE_2@', 'CODE_GENDER'],
        ['bin_a_impute_EXT_SOURCE_3@', 'CODE_GENDER'],
        ['bin_a_impute_EXT_SOURCE_1@', 'REGION_RATING_CLIENT'],
        ['bin_a_impute_EXT_SOURCE_2@', 'REGION_RATING_CLIENT'],
        ['bin_a_impute_EXT_SOURCE_3@', 'REGION_RATING_CLIENT']
    ]

    'AREA系feature'
    data['live'] = data['LIVINGAPARTMENTS_AVG'].fillna(
        0) + data['LIVINGAREA_AVG'].fillna(0) + data['LANDAREA_AVG'].fillna(0)
    live = ['LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 'LANDAREA_AVG']

    data['3AREA_AVG'] = data[['live']+live].fillna(-1).apply(
        lambda x: np.nan if x[1]+x[2]+x[3] == -3 else x[0], axis=1)
    bin_list = ['3AREA_AVG']

    data['3AREA_AVG'] = pd.cut(data['3AREA_AVG'], 5)
    data['LIVINGAPARTMENTS_AVG'] = data['LIVINGAPARTMENTS_AVG'].map(
        lambda x: 1 if x >= 0 else 0)
    data['LIVINGAREA_AVG'] = data['LIVINGAREA_AVG'].map(
        lambda x: 1 if x >= 0 else 0)
    data['LANDAREA_AVG'] = data['LANDAREA_AVG'].map(
        lambda x: 1 if x >= 0 else 0)

    categorical = [
        ['3AREA_AVG', 'REGION_RATING_CLIENT', 'CODE_GENDER'],
        ['LIVINGAPARTMENTS_AVG', 'REGION_RATING_CLIENT', 'CODE_GENDER'],
        ['LIVINGAREA_AVG', 'REGION_RATING_CLIENT', 'CODE_GENDER'],
        ['LANDAREA_AVG', 'REGION_RATING_CLIENT', 'CODE_GENDER'],
    ]
